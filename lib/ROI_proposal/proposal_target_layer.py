# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 22:30:23 2017
@author: Kevin Liang (modifications)
Adapted from the official Faster R-CNN repo:
https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/rpn/proposal_target_layer.py
"""

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import numpy as np
import numpy.random as npr
import tensorflow as tf

from lib.bbox.bbox_transform import bbox_overlaps
from lib.bbox.bbox_transform import bbox_transform
from config import config as cfg


def proposal_target_layer(rpn_rois, gt_boxes, _num_classes):
    '''
    Make Python version of _proposal_target_layer_py below Tensorflow compatible
    '''
    # rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, \
    # keep_inds, fg_num = _proposal_target_layer_py(rpn_rois, gt_boxes, _num_classes)

    rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, \
    keep_inds, fg_num = tf.py_function(_proposal_target_layer_py, [rpn_rois, gt_boxes, _num_classes],
                                       [tf.float32, tf.int32, tf.float32, tf.float32,
                                        tf.float32, tf.int32, tf.int32])

    rois = tf.reshape(rois, [-1, 5], name='rois')
    labels = tf.convert_to_tensor(value=tf.cast(labels, tf.int32), name='labels')
    bbox_targets = tf.convert_to_tensor(value=bbox_targets, name='bbox_targets')
    bbox_inside_weights = tf.convert_to_tensor(value=bbox_inside_weights, name='bbox_inside_weights')
    bbox_outside_weights = tf.convert_to_tensor(value=bbox_outside_weights, name='bbox_outside_weights')

    return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, keep_inds, fg_num


def _proposal_target_layer_py(rpn_rois, gt_boxes, _num_classes):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
    # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
    all_rois = rpn_rois.numpy()

    # Include ground-truth boxes in the set of candidate rois
    # zeros = np.zeros((gt_boxes.shape[0], 1), dtype='float32')
    # all_rois = np.vstack(
    #     (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
    # )

    num_images = 1

    # Sample rois with classification labels and bounding box regression
    # targets
    labels, rois, bbox_targets, bbox_inside_weights, keep_inds, fg_nums = _sample_rois(
        all_rois, gt_boxes, _num_classes)

    rois = rois.reshape(-1, 5)
    # preds = preds.reshape(-1,4)
    # cls = cls.reshape(-1, _num_classes + 1)
    labels = tf.reshape(labels, (-1, 1))
    # bbox_targets = bbox_targets.reshape(-1, _num_classes * 4)
    # bbox_inside_weights = bbox_inside_weights.reshape(-1, _num_classes * 4)
    # bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)

    bbox_targets = bbox_targets.reshape(-1, 4)
    bbox_inside_weights = bbox_inside_weights.reshape(-1, 4)
    bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)

    return np.float32(rois), labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, keep_inds, fg_nums


def _get_bbox_regression_labels(bbox_target_data, fgnums):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)
    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).
    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    clss = bbox_target_data[:, 0]
    clss[fgnums:] = 0.
    # bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_targets = np.zeros((clss.size, 4), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = int(4 * cls)
        end = start + 4
        # bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        # bbox_inside_weights[ind, start:end] = (1, 1, 1, 1)
        bbox_targets[ind, 0:4] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, 0:4] = (1, 1, 1, 1)
    return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_MEANS))
                   / np.array(cfg.TRAIN.BBOX_STDS))
    return np.hstack(
        (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


def _sample_rois(all_rois, gt_boxes, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, 1:5], dtype=np.float))  # (n,k)overlaps
    gt_assignment = overlaps.argmax(axis=1)  # get the gtbox with max overlaps(n,1)
    max_overlaps = overlaps.max(axis=1)  # get the max overlaps
    labels = np.where(max_overlaps[:] == 0, np.zeros(gt_assignment.shape, dtype='int32'), np.ones(gt_assignment.shape, 'int32'))
    # labels0 = tf.gather(gt_boxes, gt_assignment, axis=0)
    # labels1 = labels0[:, 4]
    # labels = tf.where(max_overlaps==0, labels1, tf.zeros(labels1.shape, tf.int32))

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = fg_inds.size
    # Sample foreground regions without replacement
    '''
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)
    '''

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    '''bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image'''
    bg_rois_per_this_image = bg_inds.size
    # Sample background regions without replacement
    '''
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)
    '''

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    fg_num = len(fg_inds)
    # labels = labels[keep_inds]
    # labels = tf.gather(labels, keep_inds, axis=0)
    labels = labels[keep_inds]
    labels[fg_rois_per_this_image:] = 0

    # Clamp labels for the background RoIs to 0
    # labels_fg = tf.cast(labels[:fg_rois_per_this_image], 'int32')
    # labels_bg = tf.zeros((labels[fg_rois_per_this_image:].shape[0],), dtype='int32')
    # labels = tf.concat((labels_fg, labels_bg), axis=-1)

    rois = all_rois[keep_inds]

    # temp = gt_boxes[gt_assignment[keep_inds], :4]
    temp = tf.gather(gt_boxes, gt_assignment[keep_inds])
    temp1 = tf.cast(temp[:, :4], 'float32')

    bbox_target_data = _compute_targets(
        rois[:, 1:5], temp1, labels)   # (labels,targets)(n,5)

    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, fg_num)

    return labels, rois, bbox_targets, bbox_inside_weights, keep_inds, fg_num
