import tensorflow as tf


class SineModel(tf.keras.Model):
    def __init__(self):
        super(SineModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', input_shape=(1,))
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.denseout = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.denseout(x)
        return out
