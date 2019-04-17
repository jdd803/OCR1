import tensorflow as tf
#from keras.layers import Conv2D
from lib.deform_conv_layer.deform_layer import ConvOffset2D

def lrelu(x, leak=0.3, name="lrelu"):
    with tf.compat.v1.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def inception_text_layer(inputs, conv11_size=256,conv11_11_size=256, conv33_11_size=256, conv33_22_size=256,
                        conv55_11_size=256, conv55_22_size=256, conv_shortcut_size=256):
    '''
    Implement of inception-text module
    :param inputs: shape(batch,h,w,c)
    :param conv11_size: 1*1 conv num used to reduce channels
    :param conv11_11_size: 1*1 branch conv nums
    :param conv33_11_size: 1*3 conv nums
    :param conv33_22_size: 3*1 conv nums
    :param conv55_11_size: 1*5 conv nums
    :param conv55_22_size: 5*1 conv nums
    :param conv_shortcut_size: 1*1 shorcut conv nums
    :return:
    '''
    pass

    with tf.compat.v1.variable_scope("conv_1x1"):
        # conv11_redu = slim.conv2d(inputs,conv11_size,(1,1),padding='same')
        # conv11_11 = slim.conv2d(conv11_redu,conv11_11_size,(1,1),padding='same')
        conv11_redu = tf.keras.layers.Conv2D(conv11_size, 1, kernel_initializer='TruncatedNormal')(inputs)
        # conv11_redu = tf.keras.layers.BatchNormalization(axis=3)(conv11_redu)
        conv11_11 = tf.keras.layers.Conv2D(conv11_11_size, 1, kernel_initializer='TruncatedNormal')(conv11_redu)
        # conv11_11 = tf.keras.layers.BatchNormalization(axis=3)(conv11_11)
        conv11_11_offset = ConvOffset2D(256, name='conv11_offset')(conv11_11)
        conv11_out = tf.keras.layers.Conv2D(conv11_11_size, 3, padding='same', activation=lrelu,
                                            kernel_initializer='TruncatedNormal')(conv11_11_offset)
    with tf.compat.v1.variable_scope("conv_3x3"):
        # conv33_redu = slim.conv2d(inputs,conv11_size,(1,1),padding='same')
        # conv33_11 = slim.conv2d(conv33_redu,conv33_11_size,(1,3),padding='same')
        # conv33_22 = slim.conv2d(conv33_11,conv33_22_size,(3,1),padding='same')
        conv33_redu = tf.keras.layers.Conv2D(conv11_size, 1, kernel_initializer='TruncatedNormal')(inputs)
        # conv33_redu = tf.keras.layers.BatchNormalization(axis=3)(conv33_redu)
        conv33_11 = tf.keras.layers.Conv2D(conv33_11_size, (1, 3), padding='same',
                                           kernel_initializer='TruncatedNormal')(conv33_redu)
        conv33_22 = tf.keras.layers.Conv2D(conv33_22_size, (3, 1), padding='same',
                                           kernel_initializer='TruncatedNormal')(conv33_11)
        # conv33_22 = tf.keras.layers.BatchNormalization(axis=3)(conv33_22)
        conv33_offset = ConvOffset2D(256, name='conv33_offset')(conv33_22)
        conv33_out = tf.keras.layers.Conv2D(conv33_22_size, 3, padding='same',
                                            kernel_initializer='TruncatedNormal', activation=lrelu)(conv33_offset)
    with tf.compat.v1.variable_scope("conv5x5"):
        # conv55_redu = slim.conv2d(inputs,conv11_size,(1,1),padding='same')
        # conv55_11 = slim.conv2d(conv55_redu,conv55_11_size,(1,5),padding='same')
        # conv55_22 = slim.conv2d(conv55_11,conv55_22_size,(5,1),padding='same')
        conv55_redu = tf.keras.layers.Conv2D(conv11_size, 1, kernel_initializer='TruncatedNormal')(inputs)
        # conv55_redu = tf.keras.layers.BatchNormalization(axis=3)(conv55_redu)
        conv55_11 = tf.keras.layers.Conv2D(conv55_11_size, (1, 5), padding='same',
                                           kernel_initializer='TruncatedNormal')(conv55_redu)
        conv55_22 = tf.keras.layers.Conv2D(conv55_22_size, (5, 1), padding='same',
                                           kernel_initializer='TruncatedNormal')(conv55_11)
        # conv55_22 = tf.keras.layers.BatchNormalization(axis=3)(conv55_22)
        conv55_offset = ConvOffset2D(256, name="conv55_offser")(conv55_22)
        conv55_out = tf.keras.layers.Conv2D(conv55_22_size, 3, padding='same',
                                            kernel_initializer='TruncatedNormal', activation=lrelu)(conv55_offset)
    conv_shortcut = tf.keras.layers.Conv2D(conv_shortcut_size, 1, kernel_initializer='TruncatedNormal')(inputs)
    # conv_shortcut = tf.keras.layers.BatchNormalization(axis=3)(conv_shortcut)
    print("Inception_conv11_out.shape:" + str(conv11_out.shape))
    print("Inception_conv33_out.shape:" + str(conv33_out.shape))
    print("Inception_conv55_out.shape:" + str(conv55_out.shape))

    conv_concat = tf.concat([conv11_out, conv33_out, conv55_out], 3)
    print("Inception_conv_concat:" + str(conv_concat.shape))
    conv_concat_conv = tf.keras.layers.Conv2D(256, 1,
                                              kernel_initializer='TruncatedNormal')(conv_concat)  # (batch,16,16,256)
    last_result = tf.nn.relu(tf.add(conv_concat_conv, conv_shortcut))
    print("Inception_last_result:" + str(last_result.shape))

    return last_result
