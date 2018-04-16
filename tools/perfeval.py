#!/usr/bin/env python2
# coding: utf-8

"""
Description:
perfeval.py: Performance Evaluation

分classification和detection两种分别计算
"""

from __future__ import print_function
import numpy as np

import sklearn.metrics as metrics


def cls_eval(scores, gt_labels):
    """
    分类任务的evaluation
    @param scores: cxm　np-array, m为样本数量(例如一个epoch)
    @param gt_labels: 1xm np-array, 元素属于{0,1,2,...,K-1}，表示K个类别的索引
    @return ap: Average Precision
    @return acc: Accuracy
    """
    num_classes, num_test_imgs = scores.shape

    pred_labels = scores.argmax(axis=0)

    ap = np.zeros((num_classes, 1), dtype=float).squeeze()
    for i in range(num_classes):
        cls_labels = np.zeros((1, num_test_imgs), dtype=float).squeeze()
        for j in range(num_test_imgs):
            if gt_labels[j]==i:
                cls_labels[j]=1
        ap[i] = metrics.average_precision_score(cls_labels, scores[i])

    acc = metrics.accuracy_score(gt_labels, pred_labels)

    return ap, acc

def isic_cls_eval(scores, gt_labels):
    """
    ISIC2016 Lesion Skin分类数据集的evaluation
    @param scores: cxm　np-array, m为样本数量(例如一个epoch)
    @param gt_labels: 1xm np-array, 元素属于{0,1,2,...,K-1}，表示K个类别的索引
    @return acc: Accuracy
    @return auc: Area Under ROC curve
    @return ap: Average Precision
    @return se: Sensitivity (i.e. Recall, tp / (tp + fn))
    @return sp: Specificity (i.e. tn / (tn + fp))
    """
    num_classes, num_test_imgs = scores.shape

    pred_labels = scores.argmax(axis=0)

    ap = np.zeros((num_classes, 1), dtype=float).squeeze()
    auc = np.zeros((num_classes, 1), dtype=float).squeeze()
    for i in range(num_classes):
        cls_labels = np.zeros((1, num_test_imgs), dtype=int).squeeze()
        for j in range(num_test_imgs):
            if gt_labels[j]==i:
                cls_labels[j]=1
        ap[i] = metrics.average_precision_score(cls_labels, scores[i])
        auc[i] = metrics.roc_auc_score(cls_labels, scores[i])
        """
        if i==0:
            sp = metrics.recall_score(cls_labels, 1-pred_labels)
        if i==1:        
            se = metrics.recall_score(cls_labels, pred_labels)
        """
    se = metrics.recall_score(gt_labels, pred_labels)
    
    sp = metrics.recall_score(1-gt_labels, 1-pred_labels)

    acc = metrics.accuracy_score(gt_labels, pred_labels)

    return acc, auc[1], ap[1], se, sp