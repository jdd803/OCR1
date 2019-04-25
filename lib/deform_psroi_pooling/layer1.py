from __future__ import absolute_import, division


import tensorflow as tf
from tensorflow.python import keras
from config import config as cfg
from lib.deform_psroi_pooling.ps_roi import ps_roi, tf_repeat, tf_flatten


class PsRoiOffset(keras.layers.Layer):
    def __init__(self, filters, pool_size, pool, feat_stride, init_normal_stddev=0.01, **kwargs):
        super(PsRoiOffset, self).__init__()
        self.filters = filters
        self.pool_size = pool_size   # control nums of relative positions
        self.pool = pool  # control whether ave_pool the ps_score_map
        self.feat_stride = feat_stride
        self.lamda = 0.1
        self.conv1 = tf.keras.layers.Conv2D(self.filters*2, (3, 3), padding='same',
                                            use_bias=False, kernel_initializer='TruncatedNormal')

    def call(self, inputs):
        features, rois = inputs
        inputs_shape = inputs.get_shape()
        roi_shape = rois.get_shape()
        roi_width = rois[3] - rois[1] + 1    # (1)
        roi_height = rois[4] - rois[2] + 1    # (1)
        offset_map = self.conv1(inputs)

        # normalized offset (n*k*k,(c+1)*2)
        # (2*(c+1)*2,k,k)
        offset = ps_roi(offset_map, self.rois, k=cfg.network.PSROI_BINS)
        offset = tf.transpose(a=offset, perm=(1, 2, 0))   # (k,k,c*2)
        offset = tf.reshape(offset, (-1, 2))  # normalized offset (k*k*(c+1),2)
        # repeats = tf.cast(offset.get_shape()[0],'int32')/tf.cast(roi_shape[0],'int32')
        # repeats = tf.cast(offset.get_shape()[0],'int32')

        # compute the roi's width and height
        roi_width = tf.cast(roi_width, 'float32')
        roi_height = tf.cast(roi_height, 'float32')

        # transform the normalized offsets to offsets by
        # element-wise product with the roi's width and height
        temp1 = offset[..., 0] * roi_width * tf.convert_to_tensor(value=self.lamda)
        temp2 = offset[..., 1] * roi_height * tf.convert_to_tensor(value=self.lamda)
        offset = tf.stack((temp1, temp2), axis=-1)  # (k*k*(c+1),2)
        pooled_response = ps_roi(features=features, boxes=rois, pool=self.pool, offsets=offset,
                                 k=self.pool_size, feat_stride=self.feat_stride)   # (depth,h,w)

        return pooled_response

