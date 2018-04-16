#!/usr/bin/env python2
# coding: utf-8

"""
inspired and copied from:
    - fcn.berkeleyvision.org
    - py-faster-rcnn
"""

from __future__ import print_function
import _init_paths
import caffe
import argparse
import os
import sys
from datetime import datetime
import cv2

from caffe.proto import caffe_pb2
import google.protobuf as pb2
import google.protobuf.text_format
import numpy as np
import perfeval

from visualdl import LogWriter #for visualization during training

def parse_args():
    """
    Parse input arguments
    """

    parser = argparse.ArgumentParser(description='Train a classification network')
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str, required=True)

    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)

    parser.add_argument('--log_dir', dest='log_dir',
                        help='log dir for VisualDL meta data',
                        default=None, type=str, required=True)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

class SolverWrapper:
    def __init__(self, solver_prototxt, log_dir, pretrained_model=None):
        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print('Loading pretrained model weights from {:s}'.format(pretrained_model))
            self.solver.net.copy_from(pretrained_model)
        
        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)
        self.cur_epoch = 0
        self.test_interval = 30  #用来替代self.solver_param.test_interval
        self.logw = LogWriter(log_dir, sync_cycle=10)
        with self.logw.mode('train') as logger:
            self.sc_train_loss = logger.scalar("loss")
            self.sc_train_acc = logger.scalar("Accuracy")
        with self.logw.mode('val') as logger:
            self.sc_val_acc = logger.scalar("Accuracy(acc)")
            self.sc_val_auc = logger.scalar("Area Under Roc Curve(auc)")
            self.sc_val_ap = logger.scalar("Average Precision(ap)")
            self.sc_val_se = logger.scalar("Sensitivity(se)")
            self.sc_val_sp = logger.scalar("Specificity(sp)")

    def train_model(self):
        """执行训练的整个流程，穿插了validation"""
        cur_iter = 0
        test_batch_size, num_classes = self.solver.test_nets[0].blobs['prob'].shape
        num_test_images_tot = test_batch_size * self.solver_param.test_iter[0]
        while cur_iter < self.solver_param.max_iter:
            #self.solver.step(self.test_interval)
            for i in range(self.test_interval):
                self.solver.step(1)
                loss = self.solver.net.blobs['loss'].data
                acc = self.solver.net.blobs['accuracy'].data
                step = self.solver.iter
                self.sc_train_loss.add_record(step, loss)
                self.sc_train_acc.add_record(step, acc)
            
            self.eval_on_val(num_classes, num_test_images_tot, test_batch_size)
            cur_iter += self.test_interval
        
    def eval_on_val(self, num_classes, num_test_images_tot, test_batch_size):
        """在整个验证集上执行inference和evaluation"""
        self.solver.test_nets[0].share_with(self.solver.net)
        self.cur_epoch += 1
        scores = np.zeros((num_classes, num_test_images_tot), dtype=float)
        gt_labels = np.zeros((1, num_test_images_tot), dtype=float).squeeze()
        for t in range(self.solver_param.test_iter[0]):
            output = self.solver.test_nets[0].forward()
            probs = output['prob']
            labels = self.solver.test_nets[0].blobs['label'].data

            gt_labels[t*test_batch_size:(t+1)*test_batch_size] = labels.T.astype(float)
            scores[:,t*test_batch_size:(t+1)*test_batch_size] = probs.T
        # TODO: 处理最后一个batch样本少于num_test_images_per_batch的情况
        
        acc, auc, ap, se, sp = perfeval.isic_cls_eval(scores, gt_labels)
        print('====================================================================\n')
        print('\tDo validation after the {:d}-th training epoch\n'.format(self.cur_epoch))
        print('>>>>', end='\t')  #设定标记，方便于解析日志获取出数据
        print('acc={:.3f}, auc={:.3f}, ap={:.3f}, se={:.3f}, sp={:.3f}\n'.format(acc, auc, ap, se, sp))
        print('\n====================================================================\n')
        step = self.solver.iter
        self.sc_val_acc.add_record(step, acc)
        self.sc_val_auc.add_record(step, auc)
        self.sc_val_ap.add_record(step, ap)
        self.sc_val_se.add_record(step, se)
        self.sc_val_sp.add_record(step, sp)
        
if __name__ == '__main__':
    args = parse_args()
    solver_prototxt = args.solver
    log_dir = args.log_dir
    pretrained_model = args.pretrained_model

    # init
    caffe.set_mode_gpu()
    caffe.set_device(0)
    
    sw = SolverWrapper(solver_prototxt, log_dir, pretrained_model)
    sw.train_model()