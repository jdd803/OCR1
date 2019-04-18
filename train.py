from __future__ import absolute_import, division, print_function, unicode_literals

import time
import numpy as np
import tensorflow as tf
import cv2

from tensorflow.python import keras
import os
from config import config as cfg
from Model import *
from data.ICDAR_mask import generate_mask,box_mask
from lib.loss.loss_function import *
from lib.ROI_proposal.proposal_target_layer import proposal_target_layer
from lib.RPN.anchor_target_layer import anchor_target_layer
from lib.loss.loss_function import rpn_cls_loss, rpn_bbox_loss
from load_data import next_batch


def loss_fun(rois, cls, offset, roi_mask, img_mask, gt_boxes, rpn_cls_score, rpn_bbox_pred):
    # get the rois which need to compute gradient
    rois1, labels1, bbox_targets1, bbox_inside_weights1, bbox_outside_weights1, keep_inds, fg_num = proposal_target_layer(
        rois, gt_boxes, cfg.dataset.NUM_CLASSES
    )
    cls_scores1 = cls[keep_inds]
    preds1 = offset[keep_inds]

    # get image's mask
    im_shape = tf.shape(input=input_images)
    im_dims = im_shape[1:3]
    mask = tf.compat.v1.py_func(generate_mask, [im_dims, img_mask], tf.int32)

    # compute every box's loss
    # compute rpn_cls_loss and rpn_bbox_loss
    rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
        anchor_target_layer(rpn_cls_score=rpn_cls_score, gt_boxes=gt_boxes, im_dims=im_dims,
                            feat_stride=cfg.network.RPN_FEAT_STRIDE, anchor_scales=cfg.network.ANCHOR_SCALES)
    loss1 = rpn_cls_loss(rpn_cls_score, rpn_labels)
    loss2 = rpn_bbox_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)

    # compute all boxes' cls_loss
    loss3 = rfcn_cls_loss(rfcn_cls_score=cls_scores1, labels=labels1)
    # compute positive boxes' bbox_loss
    loss4 = rfcn_bbox_loss(rfcn_bbox_pred=preds1, bbox_targets=bbox_targets1,
                           roi_inside_weights=bbox_inside_weights1, roi_outside_weights=bbox_outside_weights1)
    # compute positive boxes' mask_loss
    mask_j = roi_mask[keep_inds[0]]  # (h,w,2)
    mask_j = upsample_image(mask_j, rois[keep_inds[0]][2] - rois[keep_inds[0]][0],
                            rois[keep_inds[0]][3] - rois[keep_inds[0]][1])
    mask_label = box_mask(rois[keep_inds[0]], mask)
    loss5 = mask_loss(rfcn_mask_score=mask_j, labels=mask_label)
    for j in keep_inds[1:fg_num]:
        mask_j = roi_mask[j]  # (h,w,2)
        mask_j = upsample_image(mask_j, rois[j][2] - rois[j][0], rois[j][3] - rois[j][1])
        mask_label = box_mask(rois[j], mask)
        loss5 += mask_loss(rfcn_mask_score=mask_j, labels=mask_label)

    total_loss = loss1 + loss2 + loss3 + loss4 + loss5
    return total_loss


def preprocess(img, gtbox, gtmask):
    img_shape1 = img.shape
    right = img_shape1[1] - 640
    bottom = img_shape1[0] - 640
    left = np.random.randint(0, right)
    top = np.random.randint(0, bottom)
    img1 = img[top:top + 640, left:left + 640]
    inds = gtbox[:, 0::2] >= top and gtbox[:, 1::3] < top + 640
    temp1 = gtbox[inds, 0] - left
    temp2 = gtbox[inds, 1] - top
    temp3 = gtbox[inds, 2] - left
    temp4 = gtbox[inds, 3] - top
    gtbox1 = np.stack((temp1, temp2, temp3, temp4), axis=-1)
    mask1 = gtmask[inds, 0] - left
    mask2 = gtmask[inds, 1] - top
    mask3 = gtmask[inds, 2] - left
    mask4 = gtmask[inds, 3] - top
    mask5 = gtmask[inds, 4] - left
    mask6 = gtmask[inds, 5] - top
    mask7 = gtmask[inds, 6] - left
    mask8 = gtmask[inds, 7] - top
    gtmask1 = np.stack((mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8), axis=-1)
    return img1, gtbox1, gtmask1


def main():
    model = MyModel()

    opt = keras.optimizers.Adam(lr=0.001)
    checkpoint_dir = 'path/to/model_dir'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for i in range(100):
        img, gtbox, gtmask = next_batch(1, i % 10)
        img = img['1']
        img1, gtbox1, gtmask1 = preprocess(img, gtbox, gtmask)
        images = image_mean_subtraction(img1)
        with tf.GradientTape() as t:
            rpn_cls_score, rpn_bbox_pred, result_keep = model(images)
            rois, cls, offset, roi_mask = result_keep[0], result_keep[1], result_keep[2], result_keep[3]
            loss = loss_fun(rois, cls, offset, roi_mask, gtmask1, gtbox1, rpn_cls_score, rpn_bbox_pred)
        grads = t.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        if i % 10 == 0:
            print('Training loss (for one batch) at step %s: %s' % (i, float(loss)))
            print('Seen so far: %s samples' % ((i + 1) * 64))

    # dog = cv2.imread('dog.jpeg')
    # dog_mean = np.mean(dog, axis=0)
    # x1 = tf.convert_to_tensor(value=dog, dtype=tf.float32)
    # x2 = tf.convert_to_tensor([[0, 0, 18, 16, 1]])
    roi, ps_score, bbox_shift, rpn_cls_score, rpn_bbox_pred = model_part1(images=input_images, is_training=True)
    # cls, cls_result, cls_score, mask_result, rois, bbox = model_part2(imdims=(224, 224), rois=roi,
    #                                                                   ps_score_map=ps_score, bbox_shift=bbox_shift)
    result = model_part2(imdims=(224, 224), rois=roi, ps_score_map=ps_score, bbox_shift=bbox_shift)
    result_keep = model_part3(results=result)

    rois, cls, offset, roi_mask = result_keep[0], result_keep[1], result_keep[2], result_keep[3]
    # get the rois which need to compute gradient
    rois1, labels1, bbox_targets1, bbox_inside_weights1, bbox_outside_weights1, keep_inds, fg_num = proposal_target_layer(
        rois, gt_boxes, cfg.dataset.NUM_CLASSES
    )
    cls_scores1 = cls[keep_inds]
    preds1 = offset[keep_inds]

    # get image's mask
    im_shape = tf.shape(input=input_images)
    im_dims = im_shape[1:3]
    mask = tf.compat.v1.py_func(generate_mask, [im_dims, img_mask], tf.int32)

    # compute every box's loss
    # compute rpn_cls_loss and rpn_bbox_loss
    rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
        anchor_target_layer(rpn_cls_score=rpn_cls_score, gt_boxes=gt_boxes, im_dims=im_dims,
                            feat_stride=cfg.network.RPN_FEAT_STRIDE, anchor_scales=cfg.network.ANCHOR_SCALES)
    loss1 = rpn_cls_loss(rpn_cls_score, rpn_labels)
    loss2 = rpn_bbox_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)

    # compute all boxes' cls_loss
    loss3 = rfcn_cls_loss(rfcn_cls_score=cls_scores1, labels=labels1)
    # compute positive boxes' bbox_loss
    loss4 = rfcn_bbox_loss(rfcn_bbox_pred=preds1, bbox_targets=bbox_targets1,
                           roi_inside_weights=bbox_inside_weights1, roi_outside_weights=bbox_outside_weights1)
    # compute positive boxes' mask_loss
    mask_j = roi_mask[keep_inds[0]]  # (h,w,2)
    mask_j = upsample_image(mask_j, rois[keep_inds[0]][2] - rois[keep_inds[0]][0],
                            rois[keep_inds[0]][3] - rois[keep_inds[0]][1])
    mask_label = box_mask(rois[keep_inds[0]], mask)
    loss5 = mask_loss(rfcn_mask_score=mask_j, labels=mask_label)
    for j in keep_inds[1:fg_num]:
        mask_j = roi_mask[j]   # (h,w,2)
        mask_j = upsample_image(mask_j, rois[j][2]-rois[j][0], rois[j][3]-rois[j][1])
        mask_label = box_mask(rois[j], mask)
        loss5 += mask_loss(rfcn_mask_score=mask_j, labels=mask_label)

    total_loss = loss1 + loss2 + loss3 + loss4 + loss5


if __name__ == "__main__":
    main()
