import tensorflow as tf
import numpy as np

a = tf.range(16)
b = a + 2
print(b)
b[0] = 0
print(b)
keep = tf.convert_to_tensor([0, 2, 3])
d = tf.gather(b, keep, axis=0)
print(d)
