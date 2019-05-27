from __future__ import absolute_import, division, print_function, unicode_literals

import time
import numpy as np
import tensorflow as tf
import cv2

import os
from config import config as cfg
from Model1 import *
from data.ICDAR_mask import generate_mask, get_roi_mask
from lib.loss.loss_function import *
from lib.ROI_proposal.proposal_target_layer import proposal_target_layer
from lib.RPN.anchor_target_layer import anchor_target_layer
from lib.loss.loss_function import rpn_cls_loss, rpn_bbox_loss
from load_data import next_batch, next_batch1
from lib.bbox.bbox_transform import bbox_overlaps


def get_keep(inputs, keep):
    return inputs[keep]


def preprocess(img, gtbox, gtmask):
    img_size = 640
    img_shape1 = img.shape
    right = img_shape1[1] - img_size
    bottom = img_shape1[0] - img_size
    inds = []
    while len(inds) == 0:
        left = np.random.randint(0, right)
        top = np.random.randint(0, bottom)
        # left = 0
        # top = 0
        img1 = img[top:top + img_size, left:left + img_size]
        gtbox0 = np.array(gtbox)
        gtmask0 = np.array(gtmask)

        for i in range(len(gtbox0)):
            if gtbox0[i, 0] >= left and gtbox0[i, 2] < left+img_size and gtbox0[i, 1] >= top and gtbox0[i, 3] < top+img_size:
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


def preprocess1(img, gtbox, gtmask):
    img_shape1 = img.shape
    gtbox0 = np.array(gtbox)
    gtmask0 = np.array(gtmask)

    return img, gtbox0, gtmask0


def loss_mask(mask, roi0, gtmask):
    roi = tf.round(roi0,)
    roi = tf.cast(roi, tf.int32)
    roi_gtmask = get_roi_mask(roi, gtmask)
    loss = mask_loss(rfcn_mask_score=mask, labels=roi_gtmask)
    return loss


def loss_rpn(rpn_cls_score, rpn_bbox_pred, gt_boxes, img_dims):
    batch_size = rpn_cls_score.shape[0]
    gtbox_ind = np.where(gt_boxes[:, 0]==0)
    gtbox = gt_boxes[gtbox_ind, 1:]
    rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
        anchor_target_layer(tf.expand_dims(rpn_cls_score[0], 0), np.reshape(gtbox, (-1, 5)), img_dims,
                            8, (2, 4, 8, 16))

    for i in range(1, batch_size):
        gtbox_ind = np.where(gt_boxes[:, 0] == i)
        gtbox = gt_boxes[gtbox_ind, 1:]
        rpn_label, rpn_bbox_target, rpn_bbox_inside_weight, rpn_bbox_outside_weight = \
            anchor_target_layer(tf.expand_dims(rpn_cls_score[i], 0), np.reshape(gtbox, (-1, 5)), img_dims,
                                8, (2, 4, 8, 16))
        rpn_labels = tf.concat((rpn_labels, rpn_label), axis=0)
        rpn_bbox_targets = tf.concat((rpn_bbox_targets, rpn_bbox_target), axis=0)
        rpn_bbox_inside_weights = tf.concat((rpn_bbox_inside_weights,
                                             rpn_bbox_inside_weight), axis=0)
        rpn_bbox_outside_weights = tf.concat((rpn_bbox_outside_weights,
                                              rpn_bbox_outside_weight), axis=0)


    loss1 = rpn_cls_loss(rpn_cls_score, rpn_labels)
    loss2 = rpn_bbox_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)
    return loss1, loss2


def loss_cls_bbox(cls, offset, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights):
    # compute all boxes' cls_loss
    loss3 = rfcn_cls_loss(rfcn_cls_score=cls, labels=labels)
    # compute positive boxes' bbox_loss
    loss4 = rfcn_bbox_loss(rfcn_bbox_pred=offset, bbox_targets=bbox_targets,
                           roi_inside_weights=bbox_inside_weights, roi_outside_weights=bbox_outside_weights)

    # loss = loss3 + loss4
    return loss3, loss4


def compute_tp_fp_fn(cls, boxes, gt_boxes, threshold):
    gt_boxes_num = gt_boxes.shape[0]
    positive_inds = tf.where(cls > threshold)
    positive_inds = positive_inds.numpy()
    positive_num = positive_inds.shape[0]
    positive_boxes = boxes.numpy()[positive_inds, :]
    positive_boxes = np.reshape(positive_boxes, (-1, 4))

    overlaps = bbox_overlaps(
        np.ascontiguousarray(positive_boxes[:, :], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, 1:-1], dtype=np.float))  # (n,k)overlaps
    gt_assignment = overlaps.argmax(axis=1)  # get the gtbox with max overlaps(n,1)
    max_overlaps = overlaps.max(axis=1)  # get the max overlaps
    positive_overlaps = np.where(max_overlaps > 0.5)
    gt_inds = gt_assignment[positive_overlaps]
    gt_inds = np.unique(gt_inds)
    TP = gt_inds.size
    FP = positive_num - TP
    FN = gt_boxes_num - TP

    return TP, FP, FN


def main():
    batch_size = 3
    model = MyModel((432, 768), training=True)
    opt = tf.keras.optimizers.Adam(lr=0.0002)
    # model1 = tf.python.keras.Model()

    checkpoint_dir = 'path/to/model_dir'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=model)
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
        img_, gtbox_, gtmask_ = next_batch(1, 0)
        img0_ = img_['1']
        gtbox0_ = gtbox_['1']
        gtmask0_ = gtmask_['1']
        img1_, gtbox1_, gtmask1_ = preprocess1(img0_, gtbox0_, gtmask0_)
        images = image_mean_subtraction(img1_)
        pre_ini = model(images)
        model.model1.resnet50.load_weights("./path/resnet50_weights/resnet50_weights_tf.h5", by_name=True)

    pr_file = open("./pr.txt", mode='a', encoding='UTF-8-sig')

    epochs = 200
    ite = 10

    for epoch in range(epochs):
        TP = 0.
        FP = 0.
        FN = 0.

        for i in range(ite):

            ckpt.step.assign_add(1)
            img, gtbox, gtmask = next_batch1(batch_size, i % 400)
            img_dims = img.shape[1:3]
            img_mask = generate_mask(img_dims, gtmask, batch_size)
            img_mask = tf.cast(img_mask, tf.int32)
            images = image_mean_subtraction(img)
            with tf.GradientTape() as t:
                result, rpn_cls_score, rpn_bbox_pred = model(images)
                roi_pred = result[4]
                mask_pred = result[3]
                cls_score = result[2]
                cls_pred = result[0]
                offset_pred = result[5]

                # compute rpn loss
                rpn_cls_loss1, rpn_bbox_loss1 = loss_rpn(rpn_cls_score, rpn_bbox_pred, gtbox, img_dims)
                rpn_loss = rpn_cls_loss1 + rpn_bbox_loss1
                total_loss = rpn_loss

                # the first image
                batch_ind = np.reshape(np.where(np.reshape(roi_pred[:, 0], (-1)) == 0),(-1,))
                roi_pred1 = tf.gather(roi_pred, batch_ind)
                mask_pred1 = tf.gather(mask_pred, batch_ind)
                cls_score1 = tf.gather(cls_score, batch_ind)
                cls_pred1 = tf.gather(cls_pred, batch_ind)
                offset_pred1 = tf.gather(offset_pred, batch_ind)
                keep = model_part3_1(roi_pred1[:, 1:], cls_score1)

                roi_pred_keep = tf.gather(roi_pred1, keep, axis=0)
                cls_score_keep = tf.gather(cls_score1, keep, axis=0)
                cls_pred_keep = tf.gather(cls_pred1, keep, axis=0)
                offset_pred_keep = tf.gather(offset_pred1, keep, axis=0)

                rois1, labels1, bbox_targets1, bbox_inside_weights1, \
                bbox_outside_weights1, keep_inds, fg_num = proposal_target_layer(
                    roi_pred_keep, gtbox, 1
                )
                print("positive_nums:" + str(fg_num))
                cls_scores2 = tf.gather(cls_pred_keep, keep_inds, axis=0)
                preds2 = tf.gather(offset_pred_keep, keep_inds, axis=0)
                for j in range(fg_num):
                    mask1 = model_part3_2(roi_pred1, keep, keep_inds[j], mask_pred1, cls_score1)
                    rfcn_mask_loss = loss_mask(mask1, roi_pred_keep[keep_inds[j]], img_mask[0])
                    total_loss = total_loss + 2*rfcn_mask_loss

                # record the TP FP FN to compute precision and recall
                gtbox_ind = np.where(gtbox[:, 0]==0)
                TP1, FP1, FN1 = compute_tp_fp_fn(cls_pred_keep[:, 1], roi_pred_keep[:, 1:], gtbox[gtbox_ind], 0.5)
                TP += TP1
                FP += FP1
                FN += FN1

                for i1 in range(1, batch_size):
                    batch_ind = np.reshape(np.where(np.reshape(roi_pred[:, 0], (-1))==i1), (-1,))
                    roi_pred1 = tf.gather(roi_pred, batch_ind)
                    mask_pred1 = tf.gather(mask_pred, batch_ind)
                    cls_score1 = tf.gather(cls_score, batch_ind)
                    cls_pred1 = tf.gather(cls_pred, batch_ind)
                    offset_pred1 = tf.gather(offset_pred, batch_ind)
                    keep = model_part3_1(roi_pred1[:, 1:], cls_score1)

                    roi_pred_keep = tf.gather(roi_pred1, keep, axis=0)
                    cls_score_keep = tf.gather(cls_score1, keep, axis=0)
                    cls_pred_keep = tf.gather(cls_pred1, keep, axis=0)
                    offset_pred_keep = tf.gather(offset_pred1, keep, axis=0)

                    # the keep rois in rois and gt_boxes
                    rois1_1, labels1_1, bbox_targets1_1, bbox_inside_weights1_1, \
                    bbox_outside_weights1_1, keep_inds, fg_num = proposal_target_layer(
                        roi_pred_keep, gtbox, 1
                    )
                    print("positive_nums:"+str(fg_num))
                    cls_scores1_1 = tf.gather(cls_pred_keep, keep_inds, axis=0)
                    preds1_1 = tf.gather(offset_pred_keep, keep_inds, axis=0)
                    cls_scores2 = tf.concat((cls_scores2, cls_scores1_1), axis=0)
                    preds2 = tf.concat((preds2, preds1_1), axis=0)
                    labels1 = tf.concat((labels1, labels1_1), 0)
                    bbox_targets1 = tf.concat((bbox_targets1, bbox_targets1_1), axis=0)
                    bbox_inside_weights1 = tf.concat((bbox_inside_weights1, bbox_inside_weights1_1), 0)
                    bbox_outside_weights1 = tf.concat((bbox_outside_weights1, bbox_outside_weights1_1), 0)

                    for j in range(fg_num):
                        mask1 = model_part3_2(roi_pred1, keep, keep_inds[j], mask_pred1, cls_score1)
                        rfcn_mask_loss = loss_mask(mask1, roi_pred_keep[keep_inds[j]], img_mask[i1])
                        total_loss = total_loss + 2*rfcn_mask_loss

                    gtbox_ind = np.where(gtbox[:, 0] == i1)
                    TP1, FP1, FN1 = compute_tp_fp_fn(cls_pred_keep[:, 1], roi_pred_keep[:, 1:], gtbox[gtbox_ind], 0.5)
                    TP += TP1
                    FP += FP1
                    FN += FN1


                rfcn_cls_loss1, rfcn_bbox_loss1 = loss_cls_bbox(cls=cls_scores2, offset=preds2,
                                                                labels=labels1, bbox_targets=bbox_targets1,
                                                                bbox_inside_weights=bbox_inside_weights1,
                                                                bbox_outside_weights=bbox_outside_weights1)

                total_loss = total_loss + rfcn_cls_loss1 + rfcn_bbox_loss1

            if np.isnan(total_loss):
                exit()

            grads = t.gradient(total_loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))

            print('Training loss (for one batch) at epoch %s step %s: %s' % (epoch, i, float(total_loss)))
            print('rpn_cls_loss:{:.5f}---rpn_bbox_loss:{:.5f}---rfcn_cls_loss:{:.5f}---'
                  'rfcn_bbox_loss:{}'.format(rpn_cls_loss1, rpn_bbox_loss1,
                                             rfcn_cls_loss1, rfcn_bbox_loss1))
            print(' ' * 20)

            if int(ckpt.step) % 5 == 4:
                save_path = manager.save()
                print("save model")

        precision = TP / np.maximum((TP + FP), np.finfo(np.float64).eps)
        recall = TP / float(TP + FN)
        print('Precision: %s ------Recall: %s' % (precision, recall))
        print(' '*20)
        pr_file.write("P:"+str(precision) + '    '+"R"+str(recall))
        pr_file.write('\n')



if __name__ == "__main__":
    main()
