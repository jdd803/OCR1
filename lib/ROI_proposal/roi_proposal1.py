import tensorflow as tf
from config import config as cfg
from tensorflow.python import keras
from lib.ROI_proposal.RPN_softmax import rpn_softmax
from lib.ROI_proposal.proposal_layer import proposal_layer
from lib.ROI_proposal.proposal_target_layer import proposal_target_layer


class RoiProposal(keras.layers.Layer):
    '''
    Propose highest scoring boxes to the RCNN classifier
    In evaluation mode (eval_mode==True), gt_boxes should be None.
    Only support a single image
    '''

    def __init__(self, feat_stride, im_dims, eval_mode=False):
        super(RoiProposal, self).__init__()
        self.feat_stride = feat_stride
        self.im_dims = im_dims
        self.anchor_scales = cfg.network.ANCHOR_SCALES
        self.eval_mode = eval_mode

    def call(self, inputs):
        rpn_cls_score, rpn_bbox_pred = inputs
        # Convert scores to probabilities
        rpn_cls_prob = rpn_softmax(rpn_cls_score)

        # Determine best proposals
        key = 0 if self.eval_mode is False else 1
        blobs = proposal_layer(rpn_bbox_cls_prob=rpn_cls_prob, rpn_bbox_pred=rpn_bbox_pred,
                               im_dims=self.im_dims, cfg_key=key, _feat_stride=self.feat_stride,
                               anchor_scales=self.anchor_scales)

        return blobs

        # if self.eval_mode is False:
        #     # Calculate targets for proposals
        #     self.rois, self.labels, self.bbox_targets, self.bbox_inside_weights, self.bbox_outside_weights = \
        #         proposal_target_layer(rpn_rois=self.blobs, gt_boxes=self.gt_boxes,
        #                               _num_classes=self.num_classes)

    # def get_rois(self):
    #     return self.rois if self.eval_mode is False else self.blobs
    #
    # def get_labels(self):
    #     assert self.eval_mode is False, 'No labels without ground truth boxes'
    #     return self.labels
    #
    # def get_bbox_targets(self):
    #     assert self.eval_mode is False, 'No bounding box targets without ground truth boxes'
    #     return self.bbox_targets
    #
    # def get_bbox_inside_weights(self):
    #     assert self.eval_mode is False, 'No RPN inside weights without ground truth boxes'
    #     return self.bbox_inside_weights
    #
    # def get_bbox_outside_weights(self):
    #     assert self.eval_mode is False, 'No RPN outside weights without ground truth boxes'
    #     return self.bbox_outside_weights
