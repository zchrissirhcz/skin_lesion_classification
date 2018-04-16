#!/usr/bin/env python2
# coding: utf-8

'''
Title           :make_predictions_1.py
Description     :This script makes predictions using the 1st trained model and generates a submission file.
Author          :Adil Moujahid
Date Created    :20160623
Date Modified   :20160625
version         :0.2
usage           :python make_predictions_1.py
python_version  :2.7.11

adapted from: http://adilmoujahid.com/posts/2016/06/introduction-deep-learning-python-caffe/
'''

from __future__ import print_function
import _init_paths
import os
import glob
import cv2
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2
import perfeval

def transform_img_center(im, im_wt, im_ht):

	im = cv2.resize(im, (im_wt, im_ht), interpolation = cv2.INTER_CUBIC)
    
	#Histogram Equalization
	cx = 50
	cy = 206
	im[cx:cy, cx:cy, 0] = cv2.equalizeHist(im[cx:cy, cx:cy, 0])
	im[cx:cy, cx:cy, 1] = cv2.equalizeHist(im[cx:cy, cx:cy, 1])
	im[cx:cy, cx:cy, 2] = cv2.equalizeHist(im[cx:cy, cx:cy, 2])
    #Image Resizing
	
	return im

def transform_img(img, img_width, img_height):
    '''Image processing helper function'''

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


def binaryproto2npy(binaryproto_pth):
    '''Reading mean image, caffe model and its weights'''
    #Read mean image
    mean_blob = caffe_pb2.BlobProto()
    with open(binaryproto_pth) as f:
        mean_blob.ParseFromString(f.read())
    mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
        (mean_blob.channels, mean_blob.height, mean_blob.width))
    return mean_array


if __name__ == '__main__':
    caffe.set_device(0)
    caffe.set_mode_gpu()

    lab_name = 'lab5'
    #Read model architecture and trained model's weights
    #prototxt = os.path.join(pre, 'caffenet_deploy_1.prototxt')
    prototxt = 'models/isic2016/alex/{:s}/deploy.pt'.format(lab_name)
    #weights = os.path.join(pre, 'caffe_model_1_iter_10000.caffemodel')
    weights = 'output/isic2016/alex_{:s}_iter_2000.caffemodel'.format(lab_name)
    net = caffe.Net(prototxt, weights, caffe.TEST)

    num_classes = 2
    num_test_images_tot = 380
    test_iter = 10
    test_batch_size = 38
    scores = np.zeros((num_classes, num_test_images_tot), dtype=float)
    gt_labels = np.zeros((1, num_test_images_tot), dtype=float).squeeze()
    for t in range(test_iter):
        output = net.forward()
        probs = output['prob']
        labels = net.blobs['label'].data

        gt_labels[t*test_batch_size:(t+1)*test_batch_size] = labels.T.astype(float)
        scores[:,t*test_batch_size:(t+1)*test_batch_size] = probs.T
    # TODO: 处理最后一个batch样本少于num_test_images_per_batch的情况
    
    acc, auc, ap, se, sp = perfeval.isic_cls_eval(scores, gt_labels)
    print('====================================================================\n')
    print('\tDo test on the whole test dataset\n')
    print('>>>>', end='\t')  #设定标记，方便于解析日志获取出数据
    print('acc={:.3f}, auc={:.3f}, ap={:.3f}, se={:.3f}, sp={:.3f}\n'.format(acc, auc, ap, se, sp))
    print('\n====================================================================\n')
