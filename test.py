import tensorflow as tf
import numpy as np

a = tf.convert_to_tensor([1, 2, 3])
b = a + 1
c = b.numpy()