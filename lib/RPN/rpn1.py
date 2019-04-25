import tensorflow as tf
from config import config as cfg
from tensorflow.python import keras
# from lib.RPN.anchor_target_layer import anchor_target_layer
# from lib.loss.loss_function import rpn_cls_loss, rpn_bbox_loss
from lib.RPN.anchor_target_layer import anchor_target_layer
from lib.loss.loss_function import rpn_cls_loss, rpn_bbox_loss


class RPN(keras.layers.Layer):
    '''
    Region Proposal Network (RPN): From the convolutional feature maps
    (TensorBase Layers object) of the last layer, generate bounding boxes
    relative to anchor boxes and give an "objectness" score to each
    In evaluation mode (eval_mode==True), gt_boxes should be None.
    '''

    def __init__(self, im_dims, feat_stride, eval_mode):
        super(RPN, self).__init__()
        self.im_dims = im_dims
        self.feat_stride = feat_stride
        self.anchor_scales = cfg.network.ANCHOR_SCALES
        self.eval_mode = eval_mode
        self.conv1 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='TruncatedNormal')
        self.conv2 = tf.keras.layers.Conv2D(self._num_anchors * 2, (1, 1))
        self.conv3 = tf.keras.layers.Conv2D(self._num_anchors * 4, (1, 1), kernel_initializer='TruncatedNormal')

    def call(self, inputs):
        _num_anchors = len(self.anchor_scales) * 4
        rpn_layers = inputs
        features = self.conv1(rpn_layers)
        rpn_bbox_cls_layers = self.conv2(features)

        # Bounding-Box regression layer (bounding box predictions)
        rpn_bbox_pred_layers = self.conv3(features)
        return rpn_bbox_cls_layers, rpn_bbox_pred_layers
