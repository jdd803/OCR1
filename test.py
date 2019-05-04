import tensorflow as tf
import numpy as np

a = tf.Variable([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]])
print(a.shape)
b = tf.ones((2, 2, 3))
b2 = tf.cast(b, tf.float32)


def fun(a, b):
    a1 = a + 2.
    b1 = b + 1.
    c = a1 - b1
    d = tf.multiply(c, c)
    return d


with tf.GradientTape() as tape:
    d = fun(a, b)

grad = tape.gradient(d, a)
print(grad)
