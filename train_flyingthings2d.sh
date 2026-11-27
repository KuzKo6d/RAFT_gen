#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
exec "${PYTHON:-.venv/bin/python}" train.py --name raft-things2d-all-sources \
    --stage things2d --validation kitti --restore_ckpt models/raft-chairs.pth \
    --gpus 0 --num_steps 120000 --batch_size 5 --lr 0.0001 \
    --image_size 400 720 --wdecay 0.0001 --mixed_precision "$@"
