from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from raft import RAFT
import evaluate
import datasets
from utils import flow_viz

from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler, log_dir=None, image_log_freq=1000, image_log_count=2):
        if image_log_freq < 0 or image_log_count < 0:
            raise ValueError('Image logging frequency and count must be nonnegative')
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.running_count = 0
        self.image_log_freq = image_log_freq
        self.image_log_count = image_log_count
        self.writer = SummaryWriter(log_dir=log_dir)

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/self.running_count for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        for k in self.running_loss:
            self.writer.add_scalar('train/' + k, self.running_loss[k]/self.running_count, self.total_steps)
        self.writer.add_scalar('optimizer/lr', self.scheduler.get_last_lr()[0], self.total_steps)
        self.running_loss = {}
        self.running_count = 0

    def push(self, metrics):
        self.total_steps += 1
        self.running_count += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == 0:
            self._print_training_status()

    def write_dict(self, results):
        for key in results:
            self.writer.add_scalar('val/' + key, results[key], self.total_steps)

    def write_images(self, image1, image2, flow_gt, flow_pred, valid):
        """Log current training samples on step 1 and at the configured interval."""
        if self.image_log_freq == 0 or self.image_log_count == 0:
            return
        if self.total_steps != 1 and self.total_steps % self.image_log_freq != 0:
            return
        self._write_images(image1, image2, flow_gt, flow_pred, valid,
            prefix='train', max_flow=MAX_FLOW)

    def write_validation_images(self, dataset, sample_index, image1, image2, flow_gt, flow_pred, valid):
        """Log the first samples of each validation dataset on every validation pass."""
        if self.image_log_freq == 0 or sample_index >= self.image_log_count:
            return
        self._write_images(image1[None], image2[None], flow_gt[None], flow_pred[None], valid[None],
            prefix='val/' + dataset, sample_offset=sample_index)

    def _write_images(self, image1, image2, flow_gt, flow_pred, valid,
                      prefix, sample_offset=0, max_flow=None):
        count = min(self.image_log_count - sample_offset, image1.shape[0])
        image1 = image1[:count].detach().clamp(0, 255).to(torch.uint8).cpu()
        image2 = image2[:count].detach().clamp(0, 255).to(torch.uint8).cpu()
        flow_gt = flow_gt[:count].detach().float().permute(0, 2, 3, 1).cpu().numpy()
        flow_pred = flow_pred[:count].detach().float().permute(0, 2, 3, 1).cpu().numpy()
        valid = valid[:count].detach().cpu().numpy() >= 0.5
        if max_flow is not None:
            valid &= np.linalg.norm(flow_gt, axis=-1) < max_flow

        for i in range(count):
            tag = '%s/examples/%d/' % (prefix, sample_offset + i)
            self.writer.add_image(tag + 'image1', image1[i], self.total_steps)
            self.writer.add_image(tag + 'image2', image2[i], self.total_steps)

            # One conversion gives GT and prediction the same color scale.
            gt = np.where(valid[i, :, :, None], flow_gt[i], 0)
            pred = flow_pred[i]
            width = gt.shape[1]
            flow_rgb = flow_viz.flow_to_image(np.concatenate([gt, pred], axis=1))
            gt_rgb, pred_rgb = flow_rgb[:, :width].copy(), flow_rgb[:, width:].copy()
            gt_rgb[~valid[i]] = 0
            self.writer.add_image(tag + 'flow_gt', gt_rgb, self.total_steps, dataformats='HWC')
            self.writer.add_image(tag + 'flow_pred', pred_rgb, self.total_steps, dataformats='HWC')
            self.writer.add_image(tag + 'valid', valid[i].astype(np.float32), self.total_steps, dataformats='HW')

    def close(self):
        if self.running_count:
            self._print_training_status()
        self.writer.close()


def train(args):

    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()
    model.train()

    if args.stage != 'chairs':
        model.module.freeze_bn()

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler, log_dir=args.log_dir,
        image_log_freq=args.image_log_freq, image_log_count=args.image_log_count)
    print('TensorBoard logs: %s' % logger.writer.log_dir)

    VAL_FREQ = 5000
    add_noise = True

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            flow_predictions = model(image1, image2, iters=args.iters)            

            loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            metrics['loss'] = loss.detach().item()
            logger.push(metrics)
            logger.write_images(image1, image2, flow, flow_predictions[-1], valid)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                torch.save(model.state_dict(), PATH)

                results = {}
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(model.module,
                            image_callback=logger.write_validation_images))
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(model.module,
                            image_callback=logger.write_validation_images))
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(model.module,
                            image_callback=logger.write_validation_images))

                logger.write_dict(results)
                
                model.train()
                if args.stage != 'chairs':
                    model.module.freeze_bn()
            
            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--log_dir', default=None, help='TensorBoard directory (default: automatic subdirectory of runs/)')
    parser.add_argument('--image_log_freq', type=int, default=1000, help='log training images every N steps, also on step 1 (0 disables)')
    parser.add_argument('--image_log_count', type=int, default=2, help='maximum training samples to visualize per batch (0 disables)')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)
