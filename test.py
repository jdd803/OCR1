import tensorflow as tf
from tensorflow.python import keras
import numpy as np

temp = np.array([40., 40., 40.])
temp[1:] = np.ceil(temp[1:]) - 1
box1 = np.ceil(temp) - 1
print(box1)

