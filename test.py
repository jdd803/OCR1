import tensorflow as tf
import numpy as np


def sum_even(items):
    # items = items.numpy()
    if items[0] % 2 > 0.:
        s = 3 * items[0]
    else:
        s = 2 * items[0]
    s = tf.reshape(s, (-1,))
    for i in range(1, items.shape[0]):
        if items[i] % 2 > 0.:
            s1 = 3*items[i]
        else:
            s1 = 2*items[i]
        s1 = tf.reshape(s1, (-1,))
        s = tf.concat((s, s1), axis=0)
    return s


a = tf.Variable([1.0, 2.0, 3.0])


with tf.GradientTape(persistent=True) as tape:
    # a = tf.constant([10, 12, 15, 20])
    s = tf.py_function(sum_even, [a], tf.float32)
    s1 = a*a
grad = tape.gradient(s1, a)
grad1 = tape.gradient(s, a)

print(s)
