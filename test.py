import tensorflow as tf
import numpy as np

a = [1, 2, 3, 4, 5, 6]
a = np.array(a)
b = np.where((a >= 2) & (a <= 4))
print(b)
print(a[b])
