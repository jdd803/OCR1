from __future__ import absolute_import, division


import tensorflow as tf
from config import config as cfg
from lib.deform_psroi_pooling.ps_roi1 import ps_roi, ps_roi1
from lib.deform_conv_layer.deform_layer import ConvOffset2D


class PsRoiOffset(tf.keras.layers.Layer):
    def __init__(self, pool_size, pool, feat_stride, init_normal_stddev=0.01, **kwargs):
        super(PsRoiOffset, self).__init__()
        self.filters = 4*cfg.network.PSROI_BINS*cfg.network.PSROI_BINS
        self.pool_size = pool_size   # control nums of relative positions
        self.pool = 0  # control whether ave_pool the ps_score_map
        self.feat_stride = feat_stride
        self.lamda = 0.1
        self.conv1_deform = ConvOffset2D(self.filters)

    def call(self, inputs):
        features, rois = inputs
        self.filters = features.get_shape()[-1]
        # inputs_shape = features.get_shape()
        # roi_shape = rois.get_shape()
        # roi_width = rois[:, 3] - rois[:, 1] + 1    # (n, 1)
        # roi_height = rois[:, 4] - rois[:, 2] + 1    # (n, 1)

        offset_map = self.conv1_deform(features)

        pooled_response = ps_roi(features=offset_map, boxes=rois,
                                 k=self.pool_size, feat_stride=self.feat_stride)   # (n,depth,k,k)

        return pooled_response

