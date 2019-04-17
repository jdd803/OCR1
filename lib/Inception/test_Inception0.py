import lib.Inception.Inception_text as inc
import numpy as np
import tensorflow.contrib.keras.api.keras.backend as K
def main():
    input = np.random.random((4, 32, 32, 1024))

    result = inc.inception_text_layer(K.variable(input))
    #print(result.shape())


if __name__=='__main__':
    main()
