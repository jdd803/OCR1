import tensorflow as tf
import config as cfg


def rpn_cls_loss(rpn_cls_score, rpn_labels):
    '''
    Calculate the Region Proposal Network classifier loss. Measures how well
    the RPN is able to propose regions by the performance of its "objectness"
    classifier.
    Standard cross-entropy loss on logits
    '''
    # input shape dimensions
    shape = tf.shape(input=rpn_cls_score)

    # Stack all classification scores into 2D matrix
    rpn_cls_score = tf.transpose(a=rpn_cls_score, perm=[0, 3, 1, 2])
    rpn_cls_score = tf.reshape(rpn_cls_score, [shape[0], 2, shape[3] // 2 * shape[1], shape[2]])
    rpn_cls_score = tf.transpose(a=rpn_cls_score, perm=[0, 2, 3, 1])
    rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])

    # Stack labels
    rpn_labels = tf.reshape(rpn_labels, [-1])

    # Ignore label=-1 (Neither object nor background: IoU between 0.3 and 0.7)
    rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, tf.where(tf.not_equal(rpn_labels, -1))), [-1, 2])
    rpn_labels = tf.reshape(tf.gather(rpn_labels, tf.where(tf.not_equal(rpn_labels, -1))), [-1])

    # Cross entropy error
    rpn_cross_entropy = tf.reduce_mean(
        input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_labels))

    return rpn_cross_entropy


def rpn_bbox_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_inside_weights, rpn_outside_weights):
    '''
    Calculate the Region Proposal Network bounding box loss. Measures how well
    the RPN is able to propose regions by the performance of its localization.
    lam/N_reg * sum_i(p_i^* * L_reg(t_i,t_i^*))
    lam: classification vs bbox loss balance parameter
    N_reg: Number of anchor locations (~2500)
    p_i^*: ground truth label for anchor (loss only for positive anchors)
    L_reg: smoothL1 loss
    t_i: Parameterized prediction of bounding box
    t_i^*: Parameterized ground truth of closest bounding box
    '''
    # Transposing
    rpn_bbox_targets = tf.transpose(a=rpn_bbox_targets, perm=[0, 2, 3, 1])
    rpn_inside_weights = tf.transpose(a=rpn_inside_weights, perm=[0, 2, 3, 1])
    rpn_outside_weights = tf.transpose(a=rpn_outside_weights, perm=[0, 2, 3, 1])

    # How far off was the prediction?
    diff = tf.multiply(rpn_inside_weights, rpn_bbox_pred - rpn_bbox_targets)
    diff_sL1 = smoothL1(diff, 3.0)

    # Only count loss for positive anchors. Make sure it's a sum.
    rpn_bbox_reg = tf.reduce_sum(input_tensor=tf.multiply(rpn_outside_weights, diff_sL1))
    # rpn_bbox_reg = tf.reduce_sum(input_tensor=diff)

    # Constant for weighting bounding box loss with classification loss
    rpn_bbox_reg = 10 * rpn_bbox_reg

    return rpn_bbox_reg


def smoothL1(x, sigma):
    '''
    Tensorflow implementation of smooth L1 loss defined in Fast RCNN:
        (https://arxiv.org/pdf/1504.08083v2.pdf)
                    0.5 * (sigma * x)^2         if |x| < 1/sigma^2
    smoothL1(x) = {
                    |x| - 0.5/sigma^2           otherwise
    '''
    conditional = tf.less(tf.abs(x), 1 / sigma ** 2)

    close = 0.5 * (sigma * x) ** 2
    far = tf.abs(x) - 0.5 / sigma ** 2

    return tf.where(conditional, close, far)


def rfcn_cls_loss(rfcn_cls_score, labels):
    rfcn_cross_entropy = tf.reduce_mean(
        input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.squeeze(rfcn_cls_score),
                                                                    labels=tf.reshape(labels, (-1,))))

    return rfcn_cross_entropy


def mask_loss(rfcn_mask_score, labels):
    '''
    caculte the box's mask loss
    :param rfcn_mask_score:
    :param labels:
    :return:
    '''
    rfcn_mask_score = tf.reshape(rfcn_mask_score, (-1, 2), name='mask_score')
    labels = tf.reshape(labels, (-1,), name='mask_labels')
    # Cross entropy error
    mask_cross_entropy = tf.reduce_mean(
        input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rfcn_mask_score,
                                                                    labels=labels)
    )

    return mask_cross_entropy


def rfcn_bbox_loss(rfcn_bbox_pred, bbox_targets, roi_inside_weights, roi_outside_weights):
    # How far off was the prediction?
    diff = tf.multiply(roi_inside_weights, rfcn_bbox_pred - bbox_targets)
    diff_sL1 = smoothL1(diff, 1.0)

    # Only count loss for positive anchors
    roi_bbox_reg = tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=tf.multiply(roi_outside_weights, diff_sL1),
                                                             axis=[1]))

    # Constant for weighting bounding box loss with classification loss
    roi_bbox_reg = 1 * roi_bbox_reg

    return roi_bbox_reg
