import tensorflow as tf
from tensorflow.python import keras
import numpy as np


class Dog(keras.layers.Layer):
    def __init__(self):
        super(Dog, self).__init__()

    def call(self, inputs, **kwargs):
        a, b = inputs
        self.result = a + b

    def get_result(self):
        return self.result

dog = Dog()
a = tf.reshape(tf.range(4), (2, 2))
a = tf.cast(a, 'float32')
b = tf.ones((2, 2))
dog([a, b])
result = dog.get_result()
print(result)
