import time
import numpy as np
import tensorflow as tf
from Model import *
from data.ICDAR_mask import generate_mask,box_mask
from lib.loss.loss_function import *
from lib.ROI_proposal.proposal_target_layer import proposal_target_layer
from load_data import next_batch

def main():
    input_images = tf.compat.v1.placeholder(tf.float32, shape=[1,224,224,3], name='input_image')
    img_mask = tf.compat.v1.placeholder(tf.int32, shape=[None,8], name='mask')
    gt_boxes = tf.compat.v1.placeholder(tf.int32, shape=[None,5], name='groundtruth_boxes')

    global_step = tf.compat.v1.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=0.0001, global_step=global_step, decay_steps=10000,
                                               decay_rate=0.94, staircase=True)
    tf.compat.v1.summary.scalar('learning_rate',learning_rate)
    opt = tf.compat.v1.train.AdamOptimizer(learning_rate)

    init_op = tf.compat.v1.global_variables_initializer()


    with tf.compat.v1.variable_scope('model'):
        x1 = np.zeros((1,224,224,3))
        x2 = [[100,100,160,160,1]]
        roi, ps_score, bbox_shift, cls_loss_rpn, bbox_loss_rpn = model_part1(images=input_images,is_training=True,gt_boxes=gt_boxes)
        result = model_part2(imdims=(224,225), rois=roi, ps_score_map=ps_score, bbox_shift=bbox_shift)

    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        roi, ps_score, bbox_shift, cls_loss_rpn, \
        bbox_loss_rpn = sess.run((roi,ps_score,bbox_shift,cls_loss_rpn,bbox_loss_rpn),
                                 feed_dict={input_images:x1,gt_boxes:x2})
        result = sess.run(result)

    result_keep = model_part3(results=result)

    # compute the postive boxes and positive boxes
    # compute the boxes' inside_weights and out_side weights
    roi_item = result[str(result_keep[0])]['roi']   #(4,)
    offset_item = result[str(result_keep[0])]['shift']  #(4,)
    cls_item = result[str(result_keep[0])]['cls']
    rois = tf.reshape(roi_item, (1,-1))
    offset = tf.reshape(offset_item,(1,-1))
    cls = tf.reshape(cls_item,(1,-1))
    roi_mask = {}
    roi_mask['0'] = result[str(result_keep[0])]['mask']
    for i in range(1, len(result_keep)):
        roi_item = result[str(result_keep[i])]['roi']
        roi_item = tf.reshape(roi_item, (1,-1))
        rois = tf.concat((rois,roi_item), axis=0)   #(n,4)
        offset_item = result[str(result_keep[i])]['shift']
        offset = tf.reshape(offset_item, (1, -1))
        offset = tf.concat((offset, offset_item), axis=0)   #(n,4)
        cls_item = result[str(result_keep[i])]['cls']
        cls = tf.reshape(cls_item, (1, -1))
        cls = tf.concat((cls, cls_item), axis=0)    #(n,k+1)
        roi_mask[str(i)] = result[str(result_keep[i])]['mask']

    # get the rois which need to compute gradient
    rois1, labels1, bbox_targets1, bbox_inside_weights1, bbox_outside_weights1, keep_inds, fg_num = proposal_target_layer(
        rois, gt_boxes, cfg.dataset.NUM_CLASSES
    )
    cls_scores1 = cls[keep_inds]
    preds1 = offset[keep_inds]

    # get image's mask
    im_shape = tf.shape(input=input_images)
    im_dims = im_shape[1:3]
    mask = tf.compat.v1.py_func(generate_mask,[im_dims,img_mask],tf.int32)

    # compute every box's loss
    # compute rpn_cls_loss and rpn_bbox_loss
    loss1 = cls_loss_rpn
    loss2 = bbox_loss_rpn

    # compute all boxes' cls_loss
    loss3 = rfcn_cls_loss(rfcn_cls_score=cls_scores1,labels=labels1)
    # compute positive boxes' bbox_loss
    loss4 = rfcn_bbox_loss(rfcn_bbox_pred=preds1, bbox_targets=bbox_targets1,
                           roi_inside_weights=bbox_inside_weights1, roi_outside_weights=bbox_outside_weights1)
    # compute positive boxes' mask_loss
    mask_j = roi_mask[str(keep_inds[0])]  # (h,w,2)
    mask_j = upsample_image(mask_j, rois[keep_inds[0]][2] - rois[keep_inds[0]][0], rois[keep_inds[0]][3] - rois[keep_inds[0]][1])
    mask_label = box_mask(rois[keep_inds[0]], mask)
    loss5 = mask_loss(rfcn_mask_score=mask_j, labels=mask_label)
    for j in keep_inds[1:fg_num]:
        mask_j = roi_mask[str(j)]   #(h,w,2)
        mask_j = upsample_image(mask_j,rois[j][2]-rois[j][0],rois[j][3]-rois[j][1])
        mask_label = box_mask(rois[j],mask)
        loss5 += mask_loss(rfcn_mask_score=mask_j,labels=mask_label)

    total_loss = loss1 + loss2 + loss3 + loss4 + loss5
    training_step = opt.minimize(total_loss,global_step=global_step)


    with sess.as_default():
        start = time.time()
        for step in range(1000):
            x,y_gtbox,y_mask = next_batch(batch_size=1,pos=step)
            sess.run(training_step,feed_dict={input_images:x[str(step + 1)],
                                              img_mask:y_mask[str(step + 1)],
                                              gt_boxes:y_gtbox[str(step + 1)]
                                              })
            if step % 50 == 0:
                print('step {},rpn_cls_loss{},rpn_bbox_loss{},cls_loss{},'
                      'bbox_loss{},mask_loss{},total_loss{}'.format(step,loss1,loss2,
                                                                    loss3,loss4,loss5,total_loss))


        pass




if __name__ == "__main__":
    main()
