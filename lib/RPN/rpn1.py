import tensorflow as tf
from config import config as cfg
import tensorflow.contrib.slim as slim
#from keras.layers import Conv2D
from lib.RPN.anchor_target_layer import anchor_target_layer
from lib.loss.loss_function import rpn_cls_loss,rpn_bbox_loss

class rpn:
    '''
    Region Proposal Network (RPN): From the convolutional feature maps
    (TensorBase Layers object) of the last layer, generate bounding boxes
    relative to anchor boxes and give an "objectness" score to each
    In evaluation mode (eval_mode==True), gt_boxes should be None.
    '''

    def __init__(self, featureMaps, gt_boxes, im_dims, _feat_stride, eval_mode):
        self.featureMaps = featureMaps
        self.gt_boxes = gt_boxes
        self.im_dims = im_dims
        self._feat_stride = _feat_stride
        self.anchor_scales = cfg.network.ANCHOR_SCALES
        self.eval_mode = eval_mode

        self._network()

    def _network(self):
        # There shouldn't be any gt_boxes if in evaluation mode
        if self.eval_mode is True:
            assert self.gt_boxes is None, \
                'Evaluation mode should not have ground truth boxes (or else what are you detecting for?)'

        _num_anchors = len(self.anchor_scales) * 4

        rpn_layers = self.featureMaps

        with tf.compat.v1.variable_scope('rpn'):
            # Spatial windowing
            # for i in range(len([512])):
            #     rpn_layers.conv2d(filter_size=3, output_channels=512)
            #     features = slim.conv2d(rpn_layers,256,(3,3))

            # features = slim.conv2d(rpn_layers,256,(3,3))
            features = tf.keras.layers.Conv2D(256,(3,3),padding='same')(rpn_layers)

            with tf.compat.v1.variable_scope('cls'):
                # Box-classification layer (objectness)
                # self.rpn_bbox_cls_layers = Layers(features)
                # self.rpn_bbox_cls_layers.conv2d(filter_size=1, output_channels=_num_anchors * 2, activation_fn=None)

                #self.rpn_bbox_cls_layers = slim.conv2d(features,_num_anchors*2,(1,1))
                self.rpn_bbox_cls_layers = tf.keras.layers.Conv2D(_num_anchors * 2, (1, 1))(features)

            with tf.compat.v1.variable_scope('target'):
                # Only calculate targets in train mode. No ground truth boxes in evaluation mode
                if self.eval_mode is False:
                    # Anchor Target Layer (anchors and deltas)
                    rpn_cls_score = self.rpn_bbox_cls_layers
                    self.rpn_labels, self.rpn_bbox_targets, self.rpn_bbox_inside_weights, self.rpn_bbox_outside_weights = \
                        anchor_target_layer(rpn_cls_score=rpn_cls_score, gt_boxes=self.gt_boxes, im_dims=self.im_dims,
                                            _feat_stride=self._feat_stride, anchor_scales=self.anchor_scales)

            with tf.compat.v1.variable_scope('bbox'):
                # Bounding-Box regression layer (bounding box predictions)
                # self.rpn_bbox_pred_layers = Layers(features)
                # self.rpn_bbox_pred_layers.conv2d(filter_size=1, output_channels=_num_anchors * 4, activation_fn=None)

                #self.rpn_bbox_pred_layers = slim.conv2d(features,_num_anchors*4,(1,1))
                self.rpn_bbox_pred_layers = tf.keras.layers.Conv2D(_num_anchors * 4, (1, 1))(features)

    # Get functions
    def get_rpn_cls_score(self):
        return self.rpn_bbox_cls_layers

    def get_rpn_labels(self):
        assert self.eval_mode is False, 'No RPN labels without ground truth boxes'
        return self.rpn_labels

    def get_rpn_bbox_pred(self):
        return self.rpn_bbox_pred_layers

    def get_rpn_bbox_targets(self):
        assert self.eval_mode is False, 'No RPN bounding box targets without ground truth boxes'
        return self.rpn_bbox_targets

    def get_rpn_bbox_inside_weights(self):
        assert self.eval_mode is False, 'No RPN inside weights without ground truth boxes'
        return self.rpn_bbox_inside_weights

    def get_rpn_bbox_outside_weights(self):
        assert self.eval_mode is False, 'No RPN outside weights without ground truth boxes'
        return self.rpn_bbox_outside_weights

    # Loss functions
    def get_rpn_cls_loss(self):
        assert self.eval_mode is False, 'No RPN cls loss without ground truth boxes'
        rpn_cls_score = self.get_rpn_cls_score()
        rpn_labels = self.get_rpn_labels()
        return rpn_cls_loss(rpn_cls_score, rpn_labels)

    def get_rpn_bbox_loss(self):
        assert self.eval_mode is False, 'No RPN bbox loss without ground truth boxes'
        rpn_bbox_pred = self.get_rpn_bbox_pred()
        rpn_bbox_targets = self.get_rpn_bbox_targets()
        rpn_bbox_inside_weights = self.get_rpn_bbox_inside_weights()
        rpn_bbox_outside_weights = self.get_rpn_bbox_outside_weights()
        return rpn_bbox_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)