from __future__ import absolute_import, division, print_function, unicode_literals

import time
import numpy as np
import tensorflow as tf
import cv2

from tensorflow.python import keras
import os
from config import config as cfg
from Model import *
from data.ICDAR_mask import generate_mask, get_roi_mask
from lib.loss.loss_function import *
from lib.ROI_proposal.proposal_target_layer import proposal_target_layer
from lib.RPN.anchor_target_layer import anchor_target_layer
from lib.loss.loss_function import rpn_cls_loss, rpn_bbox_loss
from load_data import next_batch


def get_keep(inputs, keep):
    return inputs[keep]


def preprocess(img, gtbox, gtmask):
    img_shape1 = img.shape
    right = img_shape1[1] - 320
    bottom = img_shape1[0] - 320
    inds = []
    while len(inds) == 0:
        left = np.random.randint(0, right)
        top = np.random.randint(0, bottom)
        img1 = img[top:top + 320, left:left + 320]
        gtbox0 = np.array(gtbox)
        gtmask0 = np.array(gtmask)

        for i in range(len(gtbox0)):
            if gtbox0[i, 0] >= left and gtbox0[i, 2] < left+320 and gtbox0[i, 1] >= top and gtbox0[i, 3] < top+320:
                inds.append(i)

    # inds = gtbox0[gtbox0[:, 0::2] < +640, 0::2] >= top
    temp1 = gtbox0[inds, 0] - left
    temp2 = gtbox0[inds, 1] - top
    temp3 = gtbox0[inds, 2] - left
    temp4 = gtbox0[inds, 3] - top
    temp5 = gtbox0[inds, 4]
    gtbox1 = np.stack((temp1, temp2, temp3, temp4, temp5), axis=-1)
    mask1 = gtmask0[inds, 0] - left
    mask2 = gtmask0[inds, 1] - top
    mask3 = gtmask0[inds, 2] - left
    mask4 = gtmask0[inds, 3] - top
    mask5 = gtmask0[inds, 4] - left
    mask6 = gtmask0[inds, 5] - top
    mask7 = gtmask0[inds, 6] - left
    mask8 = gtmask0[inds, 7] - top
    gtmask1 = np.stack((mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8), axis=-1)
    return img1, gtbox1, gtmask1


def loss_mask(mask, roi, gtmask):
    roi_gtmask = get_roi_mask(roi, gtmask)
    loss = mask_loss(rfcn_mask_score=mask, labels=roi_gtmask)
    return loss


def loss_rpn(rpn_cls_score, rpn_bbox_pred, gt_boxes, img_dims):
    rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
        anchor_target_layer(rpn_cls_score, gt_boxes, img_dims,
                            8, (2, 4, 8, 16))
    loss1 = rpn_cls_loss(rpn_cls_score, rpn_labels)
    loss2 = rpn_bbox_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)
    return loss1 + loss2


def loss_cls_bbox(cls, offset, keep_inds, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights):
    # # get the rois which need to compute gradient
    # rois1, labels1, bbox_targets1, bbox_inside_weights1, \
    # bbox_outside_weights1, keep_inds, fg_num = proposal_target_layer(
    #     rois, gt_boxes, cfg.dataset.NUM_CLASSES
    # )
    # cls_scores1 = cls[keep_inds]
    # preds1 = offset[keep_inds]

    cls_scores1 = tf.gather(cls, keep_inds, axis=0)
    preds1 = tf.gather(offset, keep_inds, axis=0)

    # compute all boxes' cls_loss
    loss3 = rfcn_cls_loss(rfcn_cls_score=cls_scores1, labels=labels)
    # compute positive boxes' bbox_loss
    loss4 = rfcn_bbox_loss(rfcn_bbox_pred=preds1, bbox_targets=bbox_targets,
                           roi_inside_weights=bbox_inside_weights, roi_outside_weights=bbox_outside_weights)

    loss = loss3 + loss4
    return loss


def main():
    model = MyModel()
    # model.summary()
    opt = keras.optimizers.Adam(lr=0.001)
    checkpoint_dir = 'path/to/model_dir'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for i in range(100):
        img, gtbox, gtmask = next_batch(1, i % 10)
        img0 = img['1']
        gtbox0 = gtbox['1']
        gtmask0 = gtmask['1']
        img1, gtbox1, gtmask1 = preprocess(img0, gtbox0, gtmask0)
        img_dims = img1.shape[0:2]
        img_mask = generate_mask(img_dims, gtmask1)
        images = image_mean_subtraction(img1)
        with tf.GradientTape() as t:
            result, rpn_cls_score, rpn_bbox_pred, keep = model(images)
            model.summary()
            roi_pred = result[4]
            mask_pred = result[3]
            cls_score = result[2]
            cls_pred = result[0]
            offset_pred = result[5]
            roi_pred_keep = tf.gather(roi_pred, keep, axis=0)
            cls_score_keep = tf.gather(cls_score, keep, axis=0)
            cls_pred_keep = tf.gather(cls_pred, keep, axis=0)
            offset_pred_keep = tf.gather(offset_pred, keep, axis=0)

            rpn_loss = loss_rpn(rpn_cls_score, rpn_bbox_pred, gtbox1, img_dims)
            # the keep rois in rois and gt_boxes
            rois1, labels1, bbox_targets1, bbox_inside_weights1, \
            bbox_outside_weights1, keep_inds, fg_num = proposal_target_layer(
                roi_pred_keep, gtbox1, 1
            )

            rfcn_cls_bbox_loss = loss_cls_bbox(cls=cls_pred_keep, offset=offset_pred_keep, keep_inds=keep_inds,
                                               labels=labels1, bbox_targets=bbox_targets1,
                                               bbox_inside_weights=bbox_inside_weights1,
                                               bbox_outside_weights=bbox_outside_weights1)
            total_loss = rpn_loss + rfcn_cls_bbox_loss

            for j in tf.range(fg_num):
                if keep_inds[j] < keep.shape[0]:
                    mask1 = model_part3_2(roi_pred, keep, keep_inds[j], mask_pred, cls_score)
                    rfcn_mask_loss = loss_mask(mask1, roi_pred[keep[j]], img_mask)
                    total_loss = total_loss + rfcn_mask_loss
                else:
                    continue

            # cls, cls_result, cls_score, mask_result, rois, bbox = result[0], result[1], result[2], result[3],\
            #                                                       result[4], result[5], result[6]
            # rois, cls, offset, roi_mask = result_keep[0], result_keep[1], result_keep[2], result_keep[3]
            # loss = loss_fun(rois, cls, offset, roi_mask, gtmask1, gtbox1, rpn_cls_score, rpn_bbox_pred)
        grads = t.gradient(total_loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        if i % 10 == 0:
            print('Training loss (for one batch) at step %s: %s' % (i, float(total_loss)))
            print('Seen so far: %s samples' % ((i + 1) * 64))


if __name__ == "__main__":
    main()
