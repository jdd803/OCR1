import tensorflow as tf
from config import config as cfg
from lib.ROI_proposal.RPN_softmax import rpn_softmax
from lib.ROI_proposal.proposal_layer import proposal_layer


class roi_proposal:
    '''
    Propose highest scoring boxes to the RCNN classifier
    In evaluation mode (eval_mode==True), gt_boxes should be None.
    Only support a single image
    '''

    def __init__(self, rpn_net, im_dims, eval_mode):
        self.rpn_net = rpn_net
        self.rpn_cls_score = rpn_net.get_rpn_cls_score()
        self.rpn_bbox_pred = rpn_net.get_rpn_bbox_pred()
        self.im_dims = im_dims
        self.num_classes = cfg.dataset.NUM_CLASSES
        self.anchor_scales = cfg.network.ANCHOR_SCALES
        self.eval_mode = eval_mode

        self._network()

    def _network(self):

        with tf.compat.v1.variable_scope('roi_proposal'):
            # Convert scores to probabilities
            self.rpn_cls_prob = rpn_softmax(self.rpn_cls_score)

            # Determine best proposals
            key = 0 if self.eval_mode is False else 1
            self.blobs = proposal_layer(rpn_bbox_cls_prob=self.rpn_cls_prob, rpn_bbox_pred=self.rpn_bbox_pred,
                                        im_dims=self.im_dims, cfg_key=key, _feat_stride=self.rpn_net.feat_stride,
                                        anchor_scales=self.anchor_scales)

    def get_rois(self):
        # return self.rois if self.eval_mode is False else self.blobs
        return self.blobs

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
