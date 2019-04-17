from __future__ import absolute_import, division


import tensorflow as tf
#from keras.layers import Conv2D
#from keras.initializers import RandomNormal
from lib.deform_conv_layer.deform_conv import tf_batch_map_offsets

class ConvOffset2D(tf.keras.layers.Conv2D):
    '''ConvOffset2D'''

    def __init__(self, filters, init_normal_stddev=0.01, **kwargs):
        self.filters = filters
        super(ConvOffset2D, self).__init__(
            self.filters * 2, (3, 3), padding='same', use_bias=False,
            kernel_initializer='zeros',
            # kernel_initializer=tf.keras.initializers.RandomNormal(0, init_normal_stddev),
            **kwargs
        )

    def call(self, x):
        inputs_shape = tf.shape(input=x)
        # super(ConvOffset2D, self).build(inputs_shape)
        offset = super(ConvOffset2D, self).call(x)
        # offset = slim.conv2d(x,self.filters*2,(3,3),padding='SAME')
        offset = self._to_bc_h_w_2(offset, inputs_shape)
        x = self._to_bc_h_w(x, inputs_shape)
        x_offset = tf_batch_map_offsets(x, offset)
        x_offset = self._to_b_h_w_c(x_offset, inputs_shape)
        return x_offset

    def compute_output_shape(self, input_shape):
        return input_shape

    @staticmethod
    def _to_bc_h_w_2(inputs, inputs_shape):
        '''(b, h, w, 2c) ---> (b*c, h, w, 2)'''
        x = tf.transpose(a=inputs, perm=[0, 3, 1, 2])
        x = tf.reshape(x, (-1, inputs_shape[1], inputs_shape[2], 2))
        return x

    @staticmethod
    def _to_bc_h_w(inputs, inputs_shape):
        '''(b, h, w, c) ---> (b*c, h, w)'''
        x = tf.transpose(a=inputs, perm=[0, 3, 1, 2])
        x = tf.reshape(x, (-1, inputs_shape[1], inputs_shape[2]))
        return x

    @staticmethod
    def _to_b_h_w_c(inputs, inputs_shape):
        '''(b*c, h, w) ---> (b, h, w, c)'''
        x = tf.reshape(
            inputs, (-1, inputs_shape[3], inputs_shape[1], inputs_shape[2])
        )
        x = tf.transpose(a=x, perm=[0, 2, 3, 1])
        return x