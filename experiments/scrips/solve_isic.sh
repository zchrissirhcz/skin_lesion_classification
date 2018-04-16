#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

DB=$1
NET=$2
LAB=$3


LOG_NAME="${DB}_${NET}_${LAB}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
LOG="experiments/logs/${LOG_NAME}"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x


#
time ./tools/solve_isic.py \
  --solver models/${DB}/${NET}/${LAB}/solver.pt \
  --weights data/imagenet_models/${NET}.caffemodel \
  --log_dir experiments/vdl_logs/${DB}_${NET}_${LAB}
