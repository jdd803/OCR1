import tensorflow as tf
from tensorflow.python import keras
from lib.deform_conv_layer.deform_layer import ConvOffset2D


def lrelu(x, leak=0.3, name="lrelu"):
    with tf.compat.v1.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


class InceptionTextLayer(keras.layers.Layer):
    def __init__(self):
        super(InceptionTextLayer, self).__init__()
        self.conv11_size = 256
        self.conv11_11_size = 256
        self.conv33_11_size = 256
        self.conv33_22_size = 256
        self.conv55_11_size = 256
        self.conv55_22_size = 256
        self.conv_shortcut_size = 256

        self.conv1_1 = tf.keras.layers.Conv2D(self.conv11_size, 1, kernel_initializer='TruncatedNormal')
        self.conv1_2 = tf.keras.layers.Conv2D(self.conv11_11_size, 1, kernel_initializer='TruncatedNormal')
        self.conv1_deform = ConvOffset2D(256, name='conv11_offset')
        self.conv1_3 = tf.keras.layers.Conv2D(self.conv11_11_size, 3, padding='same', activation=lrelu,
                                              kernel_initializer='TruncatedNormal')

        self.conv2_1 = tf.keras.layers.Conv2D(self.conv11_size, 1, kernel_initializer='TruncatedNormal')
        self.conv2_2 = tf.keras.layers.Conv2D(self.conv33_11_size, (1, 3), padding='same',
                                              kernel_initializer='TruncatedNormal')
        self.conv2_3 = tf.keras.layers.Conv2D(self.conv33_22_size, (3, 1), padding='same',
                                              kernel_initializer='TruncatedNormal')
        self.conv2_defom = ConvOffset2D(256, name='conv33_offset')
        self.conv2_4 = tf.keras.layers.Conv2D(self.conv33_22_size, 3, padding='same',
                                              kernel_initializer='TruncatedNormal', activation=lrelu)

        self.conv3_1 = tf.keras.layers.Conv2D(self.conv11_size, 1, kernel_initializer='TruncatedNormal')
        self.conv3_2 = tf.keras.layers.Conv2D(self.conv55_11_size, (1, 5), padding='same',
                                              kernel_initializer='TruncatedNormal')
        self.conv3_3 = tf.keras.layers.Conv2D(self.conv55_22_size, (5, 1), padding='same',
                                              kernel_initializer='TruncatedNormal')
        self.conv3_deform = ConvOffset2D(256, name="conv55_offser")
        self.conv3_4 = tf.keras.layers.Conv2D(self.conv55_22_size, 3, padding='same',
                                              kernel_initializer='TruncatedNormal', activation=lrelu)

        self.conv4 = tf.keras.layers.Conv2D(self.conv_shortcut_size, 1, kernel_initializer='TruncatedNormal')
        self.conv5 = tf.keras.layers.Conv2D(256, 1, kernel_initializer='TruncatedNormal')
        self.add1 = tf.keras.layers.Add()
        self.ac1 = tf.keras.layers.Activation(activation='relu')

    def call(self, inputs):
        conv11_redu = self.conv1_1(inputs)
        conv11_11 = self.conv1_2(conv11_redu)
        conv11_11_offset = self.conv1_deform(conv11_11)
        conv11_out = self.conv1_3(conv11_11_offset)

        conv33_redu = self.conv2_1(inputs)
        conv33_11 = self.conv2_2(conv33_redu)
        conv33_22 = self.conv2_3(conv33_11)
        conv33_offset = self.conv2_defom(conv33_22)
        conv33_out = self.conv2_4(conv33_offset)

        conv55_redu = self.conv3_1(inputs)
        conv55_11 = self.conv3_2(conv55_redu)
        conv55_22 = self.conv3_3(conv55_11)
        conv55_offset = self.conv3_deform(conv55_22)
        conv55_out = self.conv3_4(conv55_offset)

        conv_shortcut = self.conv4(inputs)
        print("Inception_conv11_out.shape:" + str(conv11_out.shape))
        print("Inception_conv33_out.shape:" + str(conv33_out.shape))
        print("Inception_conv55_out.shape:" + str(conv55_out.shape))

        conv_concat = tf.concat([conv11_out, conv33_out, conv55_out], 3)
        print("Inception_conv_concat:" + str(conv_concat.shape))
        conv_concat_conv = self.conv5(conv_concat)  # (batch,16,16,256)
        temp = self.add1([conv_concat_conv, conv_shortcut])
        last_result = self.ac1(temp)
        print("Inception_last_result:" + str(last_result.shape))

        return last_result
