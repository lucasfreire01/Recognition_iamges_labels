import tensorflow as tf
import numpy as np


class transformer_pipeline:

    def __init__(self, X, y, cat=10):

        assert isinstance(cat, int), "cat must be of type int"

        self.X = X
        self.y = y
        self.cat = cat
        self.modified_x_vector = None
        self.modified_y_vector = None

    def encoding(self):
        self.modified_y_vector = tf.keras.utils.to_categorical(self.y, self.cat)

    def normalization(self):
        self.modified_x_vector = self.X / 255.0

    def grayscale(self):

        self.modified_x_vector = np.sum(self.X / 3, axis=3, keepdims=True)

    def fit_transform(self):
        self.normalization()
        self.grayscale()
        self.encoding()
        return self.modified_x_vector, self.modified_y_vector
