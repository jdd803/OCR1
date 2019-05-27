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

        #
        # # normalized offset (n*k*k,(c+1)*2)
        # # (2*(c+1)*2,k,k)
        # offset = ps_roi(offset_map, rois, k=cfg.network.PSROI_BINS)  # (n,c*2,k,k)
        # offset_shape = tf.shape(offset)
        # offset = tf.reshape(offset, (offset_shape[0], -1, 2, offset_shape[2], offset_shape[3]))  # (n,c,2,k,k)
        # offset = tf.transpose(a=offset, perm=(2, 3, 4, 1, 0))   # (2,k,k,c,n)
        #
        # offset = tf.cast(offset, tf.float32)
        #
        # # transform the normalized offsets to offsets by
        # # element-wise product with the roi's width and height
        # temp1 = offset[0, ...] * roi_width * tf.convert_to_tensor(value=self.lamda)
        # temp2 = offset[1, ...] * roi_height * tf.convert_to_tensor(value=self.lamda)
        # offset = tf.stack((temp1, temp2), axis=0)  # (2,k,k,c,n)
        # offset = tf.transpose(offset, (4, 1, 2, 3, 0))  # (n,k,k,c,2)
        # offset = tf.reshape(offset, (offset_shape[0], offset_shape[2]*offset_shape[3], -1, 2))  # (n,k*k,c,2)

        pooled_response = ps_roi(features=offset_map, boxes=rois,
                                 k=self.pool_size, feat_stride=self.feat_stride)   # (n,depth,k,k)

        return pooled_response

