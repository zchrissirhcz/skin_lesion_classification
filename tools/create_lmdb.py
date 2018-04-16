#!/usr/bin/env python2
# coding: utf-8

'''
modified from: https://github.com/adilmoujahid/deeplearning-cats-dogs-tutorial/blob/master/code/create_lmdb.py

Title           :create_lmdb.py
Description     :This script divides the training images into 2 sets and stores them in lmdb databases for training and validation.
Author          :Adil Moujahid
Date Created    :20160619
Date Modified   :20160625
version         :0.2
usage           :python create_lmdb.py
python_version  :2.7.11
'''


import _init_paths
import os
import glob
import random
import numpy as np

import cv2

import caffe
from caffe.proto import caffe_pb2
import lmdb

#Size of images
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

def transform_img_center(im, im_wt=IMAGE_WIDTH, im_ht=IMAGE_HEIGHT):
	"""
	on center region of resize image, do histogram equalization
	"""

	im = cv2.resize(im, (im_wt, im_ht), interpolation = cv2.INTER_CUBIC)
    
	#Histogram Equalization
	cx = 50
	cy = 206
	im[cx:cy, cx:cy, 0] = cv2.equalizeHist(im[cx:cy, cx:cy, 0])
	im[cx:cy, cx:cy, 1] = cv2.equalizeHist(im[cx:cy, cx:cy, 1])
	im[cx:cy, cx:cy, 2] = cv2.equalizeHist(im[cx:cy, cx:cy, 2])
    #Image Resizing
	

	return im

def transform_img(im, im_wt=IMAGE_WIDTH, im_ht=IMAGE_HEIGHT):

    #Histogram Equalization
	"""
	im[:, :, 0] = cv2.equalizeHist(im[:, :, 0])
	im[:, :, 1] = cv2.equalizeHist(im[:, :, 1])
	im[:, :, 2] = cv2.equalizeHist(im[:, :, 2])
	"""
    #Image Resizing
	im = cv2.resize(im, (im_wt, im_ht), interpolation = cv2.INTER_CUBIC)

	return im


def make_datum(img, label):
    #image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring())


def make_lmdb(db_name, mp, im_fd='images', split_fd='splits'):
	"""
	im_fd: image folder
	split_fd: split folder
	"""
	im_dir = '/opt/work/lesion/data/{:s}/{:s}'.format(db_name, im_fd)
	num = {
		'train': 0,
		'val': 0,
		'test': 0
	}

	for split in ['train', 'val', 'test']:
		num_0 = 1
		num_1 = 1

		lmdb_pth = '/opt/work/lesion/data/{:s}/{:s}_lmdb'.format(db_name, split)
		if os.path.exists(lmdb_pth):
			os.system('rm -rf  ' + lmdb_pth)

		split_txt = '/opt/work/lesion/data/{:s}/{:s}/{:s}.txt'.format(db_name, split_fd, split)
		fin = open(split_txt)
		im_name_label_line_list = [_.rstrip('\n') for _ in fin.readlines()]
		fin.close()
		random.shuffle(im_name_label_line_list)
		
		#Shuffle im_names with corresponding labels
		random.shuffle(im_name_label_line_list)

		print ('Creating {:s}_lmdb'.format(split))

		in_db = lmdb.open(lmdb_pth, map_size=int(1e12))
		with in_db.begin(write=True) as in_txn:
			for in_idx, name_label in enumerate(im_name_label_line_list):
				im_name = name_label.split(' ')[0]
				label = int(name_label.split(' ')[1])
				if label == 0:
					num_0 += 1
				else:
					num_1 += 1

				im_pth = os.path.join(im_dir, im_name)
				im = cv2.imread(im_pth, cv2.IMREAD_COLOR)
				im = transform_img_center(im, im_wt=IMAGE_WIDTH, im_ht=IMAGE_HEIGHT)
				datum = make_datum(im, label)
				in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
				print('{:0>5d}'.format(in_idx) + ':' + im_pth + ' labeled as:' + mp[label] +' num_0:num_1='+str(num_0*1.0/num_1))
		in_db.close()

	print('\n===================')
	print('{:d} train images, {:d} val images'.format(num['train'], num['val']))
	print ('\nFinished processing all images')

if __name__ == '__main__':
	mp = ['benign', 'malignant']
	make_lmdb('isic2016', mp)
