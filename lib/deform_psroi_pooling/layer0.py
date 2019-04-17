from __future__ import absolute_import, division


import tensorflow as tf
from config import config as cfg
from lib.deform_psroi_pooling.ps_roi import ps_roi,tf_repeat,tf_flatten

class PS_roi_offset():

    def __init__(self,features,rois,pool_size,pool,feat_stride,init_normal_stddev = 0.01,**kwargs):
        self.features = features
        self.filters = 4*cfg.network.PSROI_BINS*cfg.network.PSROI_BINS
        self.rois = rois
        self.pool_size = pool_size   #control nums of relative positions
        self.pool = pool  #control whether ave_pool the ps_score_map
        self.feat_stride = feat_stride
        self.lamda = 0.1


    def call(self, inputs):
        inputs_shape = inputs.get_shape()
        roi_shape = self.rois.get_shape()
        roi_width = self.rois[3] - self.rois[1]    #(1)
        roi_height = self.rois[4] - self.rois[2]    #(1)
        offset_map = tf.keras.layers.Conv2D(self.filters*2,(3,3),padding='same',
                                            use_bias=False)(inputs)
        roi = tf_flatten(self.rois)
        #roi = tf.reshape(self.rois, (-1))
        offset = ps_roi(offset_map,roi,k=cfg.network.PSROI_BINS)  # normalized offset (n*k*k,(c+1)*2)(2*(c+1)*2,k,k)
        offset = tf.transpose(offset,(1,2,0))   #(k,k,c*2)
        offset = tf.reshape(offset,(-1,2)) # normalized offset (k*k*(c+1),2)
        # repeats = tf.cast(offset.get_shape()[0],'int32')/tf.cast(roi_shape[0],'int32')
        # repeats = tf.cast(offset.get_shape()[0],'int32')


        # compute the roi's width and height
        roi_width = tf.cast(roi_width,'float32')
        roi_height = tf.cast(roi_height,'float32')

        # transform the normalized offsets to offsets by
        # element-wise product with the roi's width and height
        temp1 = offset[...,0] * roi_width * tf.convert_to_tensor(self.lamda)
        temp2 = offset[...,1] * roi_height * tf.convert_to_tensor(self.lamda)
        offset = tf.stack((temp1,temp2),axis=-1)  #(k*k*(c+1),2)
        pooled_response = ps_roi(features=self.features,
                                 boxes=roi,pool=self.pool,offsets=offset,
                                 k=self.pool_size,feat_stride=self.feat_stride)   #(depth,h,w)
        return pooled_response