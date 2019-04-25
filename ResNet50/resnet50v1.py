from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings
import tensorflow as tf
from tensorflow.python import keras

backend = None

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.2/'
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.2/'
                       'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')


class IdentityBlock(keras.layers.Layer):
    def __init__(self, kernel_size, filters, stage, block):
        super(IdentityBlock, self).__init__()
        self.kernel_size = kernel_size
        self.filters1, self.filters2, self.filters3 = filters
        self.stage = stage
        self.block = block
        bn_axis = 3
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        self.conv1 = tf.keras.layers.Conv2D(self.filters1, (1, 1),
                                            kernel_initializer='TruncatedNormal',
                                            name=conv_name_base + '2a')
        self.bn1 = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')
        self.ac1 = tf.keras.layers.Activation('relu')

        self.conv2 = tf.keras.layers.Conv2D(self.filters2, kernel_size,
                                            padding='same',
                                            kernel_initializer='TruncatedNormal',
                                            name=conv_name_base + '2b')
        self.bn2 = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')
        self.ac2 = tf.keras.layers.Activation('relu')

        self.conv3 = tf.keras.layers.Conv2D(self.filters3, (1, 1),
                                            kernel_initializer='TruncatedNormal',
                                            name=conv_name_base + '2c')
        self.bn3 = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')

        self.add1 = tf.keras.layers.Add()
        self.ac3 = tf.keras.layers.Activation('relu')

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.ac1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.ac2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.add1(x, inputs)
        x = self.ac3(x)
        return x


class ConvBlock(keras.layers.Layer):
    def __init__(self, kernel_size, filters, stage, block, strides=(2, 2)):
        super(ConvBlock, self).__init__()
        self.kernel.size = kernel_size
        self.filters1, self.filters2, self.filters3 = filters
        self.stage = stage
        self.block = block
        self.strides = strides
        bn_axis = 3
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        self.conv1 = tf.keras.layers.Conv2D(self.filters1, (1, 1), strides=strides,
                                            kernel_initializer='TruncatedNormal',
                                            name=conv_name_base + '2a')
        self.bn1 = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')
        self.ac1 = tf.keras.layers.Activation('relu')

        self.conv2 = tf.keras.layers.Conv2D(self.filters2, kernel_size, padding='same',
                                            kernel_initializer='TruncatedNormal',
                                            name=conv_name_base + '2b')
        self.bn2 = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')
        self.ac2 = tf.keras.layers.Activation('relu')

        self.conv3 = tf.keras.layers.Conv2D(self.filters3, (1, 1),
                                            kernel_initializer='TruncatedNormal',
                                            name=conv_name_base + '2c')
        self.bn3 = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')

        self.conv4 = tf.keras.layers.Conv2D(self.filters3, (1, 1), strides=strides,
                                            kernel_initializer='TruncatedNormal',
                                            name=conv_name_base + '1')
        self.bn4 = tf.keras.layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '1')

        self.add1 = tf.keras.layers.Add()
        self.ac3 = tf.keras.layers.Activation('relu')

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.ac1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.ac2(x)
        x = self.conv3(x)
        x = self.bn3(x)

        shortcut = self.conv4(inputs)
        shortcut = self.bn4(shortcut)

        x = self.add1([x, shortcut])
        x = self.ac3(x)
        return x


class IdentityBlock1(keras.layers.Layer):
    def __init__(self, kernel_size, filters, stage, block):
        super(IdentityBlock1, self).__init__()
        self.kernel_size = kernel_size
        self.filters1, self.filters2, self.filters3 = filters
        self.stage = stage
        self.block = block
        bn_axis = 3
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        self.conv1 = tf.keras.layers.Conv2D(self.filters1, (1, 1), padding='same', dilation_rate=2,
                                            kernel_initializer='TruncatedNormal',
                                            name=conv_name_base + '2a')
        self.bn1 = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')
        self.ac1 = tf.keras.layers.Activation('relu')

        self.conv2 = tf.keras.layers.Conv2D(self.filters2, kernel_size,
                                            padding='same', dilation_rate=2,
                                            kernel_initializer='TruncatedNormal',
                                            name=conv_name_base + '2b')
        self.bn2 = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')
        self.ac2 = tf.keras.layers.Activation('relu')

        self.conv3 = tf.keras.layers.Conv2D(self.filters3, (1, 1), padding='same', dilation_rate=2,
                                            kernel_initializer='TruncatedNormal',
                                            name=conv_name_base + '2c')
        self.bn3 = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')

        self.add1 = tf.keras.layers.Add()
        self.ac3 = tf.keras.layers.Activation('relu')

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.ac1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.ac2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.add1(x, inputs)
        x = self.ac3(x)
        return x


class ConvBlock1(keras.layers.Layer):
    def __init__(self, kernel_size, filters, stage, block, strides=(2, 2)):
        super(ConvBlock1, self).__init__()
        self.kernel.size = kernel_size
        self.filters1, self.filters2, self.filters3 = filters
        self.stage = stage
        self.block = block
        self.strides = strides
        bn_axis = 3
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        self.conv1 = tf.keras.layers.Conv2D(self.filters1, (1, 1), strides=self.strides, padding='same',
                                            kernel_initializer='TruncatedNormal', dilation_rate=2,
                                            name=conv_name_base + '2a')
        self.bn1 = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')
        self.ac1 = tf.keras.layers.Activation('relu')

        self.conv2 = tf.keras.layers.Conv2D(self.filters2, self.kernel_size, padding='same',
                                            kernel_initializer='TruncatedNormal', dilation_rate=2,
                                            name=conv_name_base + '2b')
        self.bn2 = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')
        self.ac2 = tf.keras.layers.Activation('relu')

        self.conv3 = tf.keras.layers.Conv2D(self.filters3, (1, 1),
                                            kernel_initializer='TruncatedNormal', dilation_rate=2,
                                            name=conv_name_base + '2c')
        self.bn3 = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')

        self.conv4 = tf.keras.layers.Conv2D(self.filters3, (1, 1), strides=self.strides, padding='same',
                                            kernel_initializer='TruncatedNormal', dilation_rate=2,
                                            name=conv_name_base + '1')
        self.bn4 = tf.keras.layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '1')

        self.add1 = tf.keras.layers.Add()
        self.ac3 = tf.keras.layers.Activation('relu')

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.ac1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.ac2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        shortcut = self.conv4(inputs)
        shortcut = self.bn4(shortcut)

        x = self.add1([x, shortcut])
        x = self.ac3(x)
        return x


class ResNet50(keras.Model):
    def __init__(self, weights=None, input_shape=None, pooling=None):
        super(ResNet50, self).__init__()
        self.weights = weights
        self.input_shape = input_shape
        self.pooling = pooling
        bn_axis = 3
        self.pad1 = tf.keras.layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')
        self.conv1 = tf.keras.layers.Conv2D(64, (7, 7),
                                            strides=(2, 2),
                                            padding='valid',
                                            kernel_initializer='TruncatedNormal',
                                            name='conv1')
        self.bn1 = tf.keras.layers.BatchNormalization(axis=bn_axis, name='bn_conv1')
        self.ac1 = tf.keras.layers.Activation('relu')
        self.pad2 = tf.keras.layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')
        self.max_pool1 = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))

        self.cblock1 = ConvBlock(3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        self.iblock1_1 = IdentityBlock(3, [64, 64, 256], stage=2, block='b')
        self.iblock1_2 = IdentityBlock(3, [64, 64, 256], stage=2, block='c')

        self.cblock2 = ConvBlock(3, [128, 128, 512], stage=3, block='a')
        self.iblock2_1 = IdentityBlock(3, [128, 128, 512], stage=3, block='b')
        self.iblock2_2 = IdentityBlock(3, [128, 128, 512], stage=3, block='c')
        self.iblock2_3 = IdentityBlock(3, [128, 128, 512], stage=3, block='d')

        self.cblock3 = ConvBlock(3, [256, 256, 1024], stage=4, block='a')
        self.iblock3_1 = IdentityBlock(3, [256, 256, 1024], stage=4, block='b')
        self.iblock3_2 = IdentityBlock(3, [256, 256, 1024], stage=4, block='c')
        self.iblock3_3 = IdentityBlock(3, [256, 256, 1024], stage=4, block='d')
        self.iblock3_4 = IdentityBlock(3, [256, 256, 1024], stage=4, block='e')
        self.iblock3_5 = IdentityBlock(3, [256, 256, 1024], stage=4, block='f')

        self.cblock4 = ConvBlock1(3, [512, 512, 2048], stage=5, block='a')
        self.iblock4_1 = IdentityBlock1(3, [512, 512, 2048], stage=5, block='b')
        self.iblock4_2 = IdentityBlock1(3, [512, 512, 2048], stage=5, block='c')

    def call(self, inputs, training=None, mask=None):
        img_input = tf.expand_dims(input=inputs, axis=0)
        img_input = tf.cast(img_input, tf.float32)
        x = self.pad1(img_input)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.ac1(x)
        x = self.pad2(x)
        x = self.max_pool1(x)

        x = self.cblock1(x)
        x = self.iblock1_1(x)
        x = self.iblock1_2(x)

        x = self.cblock2(x)
        x = self.iblock2_1(x)
        x = self.iblock2_2(x)
        x = self.iblock2_3(x)
        x1 = x

        x = self.cblock3(x)
        x = self.iblock3_1(x)
        x = self.iblock3_2(x)
        x = self.iblock3_3(x)
        x = self.iblock3_4(x)
        x = self.iblock3_5(x)
        x2 = x

        x = self.cblock4(x)
        x = self.iblock4_1(x)
        x = self.iblock4_2(x)
        x3 = x
        return x1, x2, x3
