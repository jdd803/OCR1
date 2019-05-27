from __future__ import absolute_import, division, print_function, unicode_literals

import os
from Model1 import *
from lib.loss.loss_function import *
from load_data import next_batch, next_batch1
from lib.bbox.bbox_transform import bbox_overlaps


def get_keep(inputs, keep):
    return inputs[keep]


def preprocess1(img, gtbox, gtmask):
    gtbox0 = np.array(gtbox)
    gtmask0 = np.array(gtmask)

    return img, gtbox0, gtmask0


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
    opt = tf.keras.optimizers.Adam(lr=0.001)
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
        print("There's no model.")

    epochs = 300
    ite = 10

    for epoch in range(epochs):
        TP = 0.
        FP = 0.
        FN = 0.

        for i in range(ite):
            ckpt.step.assign_add(1)
            img, gtbox, gtmask = next_batch1(batch_size, i % 400)
            images = image_mean_subtraction(img)
            result, rpn_cls_score, rpn_bbox_pred = model(images)
            roi_pred = result[4]
            mask_pred = result[3]
            cls_score = result[2]
            cls_pred = result[0]
            offset_pred = result[5]

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

                gtbox_ind = np.where(gtbox[:, 0] == i1)
                TP1, FP1, FN1 = compute_tp_fp_fn(cls_pred_keep[:, 1], roi_pred_keep[:, 1:], gtbox[gtbox_ind], 0.5)
                TP += TP1
                FP += FP1
                FN += FN1

        precision = TP / np.maximum((TP + FP), np.finfo(np.float64).eps)
        recall = TP / float(TP + FN)
        print('Precision: %s ------Recall: %s' % (precision, recall))



if __name__ == "__main__":
    main()
