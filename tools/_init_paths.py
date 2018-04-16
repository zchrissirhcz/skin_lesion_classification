#!/usr/bine/env python
# coding: utf-8

import sys,os
caffe_dir = '/home/chris/work/caffe-BVLC'
pycaffe_dir = os.path.join(caffe_dir, 'python')
sys.path.insert(0, pycaffe_dir)

lib_dir = os.path.join('/opt/work/lesion/lib')
sys.path.insert(0, lib_dir)