import tensorflow as tf

image_input = tf.keras.Input(shape=(32, 32, 3), name='img_input')
timeseries_input = tf.keras.Input(shape=(None, 10), name='ts_input')

x1 = tf.keras.layers.Conv2D(3, 3)(image_input)
x1 = tf.keras.layers.GlobalMaxPooling2D()(x1)

x2 = tf.keras.layers.Conv1D(3, 3)(timeseries_input)
x2 = tf.keras.layers.GlobalMaxPooling1D()(x2)

x = tf.keras.layers.concatenate([x1, x2])

score_output = tf.keras.layers.Dense(1, name='score_output')(x)
class_output = tf.keras.layers.Dense(5, activation='softmax', name='class_output')(x)

model = tf.keras.Model(inputs=[image_input, timeseries_input],
                       outputs=[score_output, class_output])

tf.keras.utils.plot_model(model, 'multi_input_and_output_model.png', show_shapes=True)
