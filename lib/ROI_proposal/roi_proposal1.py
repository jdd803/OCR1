import tensorflow as tf
from config import config as cfg
from lib.ROI_proposal.RPN_softmax import rpn_softmax
from lib.ROI_proposal.proposal_layer import proposal_layer
from lib.ROI_proposal.proposal_target_layer import proposal_target_layer


class RoiProposal(tf.keras.layers.Layer):
    '''
    Propose highest scoring boxes to the RCNN classifier
    In evaluation mode (eval_mode==True), gt_boxes should be None.
    Only support a single image
    '''

    def __init__(self, feat_stride, im_dims, eval_mode=False):
        super(RoiProposal, self).__init__()
        self.feat_stride = tf.convert_to_tensor(feat_stride)
        self.im_dims = tf.convert_to_tensor(im_dims)
        self.anchor_scales = tf.convert_to_tensor(cfg.network.ANCHOR_SCALES)
        self.eval_mode = eval_mode

    def call(self, inputs):
        rpn_cls_score, rpn_bbox_pred = inputs
        # Convert scores to probabilities
        rpn_cls_prob = rpn_softmax(rpn_cls_score)

        # Determine best proposals
        key = 0 if self.eval_mode is False else 1
        key = tf.convert_to_tensor(key)

        blobs = proposal_layer(rpn_bbox_cls_prob=tf.expand_dims(rpn_cls_prob[0], 0),
                               rpn_bbox_pred=tf.expand_dims(rpn_bbox_pred[0], 0),
                               im_dims=self.im_dims, cfg_key=key, _feat_stride=self.feat_stride,
                               anchor_scales=self.anchor_scales, batch_ind=0)
        blobs = tf.reshape(blobs, (-1, 5))

        for i in range(1, rpn_cls_score.shape[0]):
            blob = proposal_layer(rpn_bbox_cls_prob=tf.expand_dims(rpn_cls_prob[i], 0),
                                  rpn_bbox_pred=tf.expand_dims(rpn_bbox_pred[i], 0),
                                  im_dims=self.im_dims, cfg_key=key, _feat_stride=self.feat_stride,
                                  anchor_scales=self.anchor_scales, batch_ind=i)
            blob = tf.reshape(blob, (-1, 5))
            blobs = tf.concat((blobs, blob), axis=0)

        return blobs
