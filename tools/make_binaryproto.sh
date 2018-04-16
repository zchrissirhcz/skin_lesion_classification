#!/bin/bash


COMPUTE_IMAGE_MEAN='/home/chris/work/caffe-BVLC/build/tools/compute_image_mean'

db_name='isic2016'

#$COMPUTE_IMAGE_MEAN --backend=lmdb /opt/work/lesion/data/catsdogs/train_lmdb /opt/work/lesion/data/catsdogs/mean.binaryproto
$COMPUTE_IMAGE_MEAN --backend=lmdb /opt/work/lesion/data/${db_name}/train_lmdb /opt/work/lesion/data/${db_name}/mean.binaryproto
#$COMPUTE_IMAGE_MEAN --backend=lmdb /opt/work/deeplearning-cats-dogs-tutorial/input/train_lmdb /opt/work/lesion/data/catsdogs/mean.binaryproto
