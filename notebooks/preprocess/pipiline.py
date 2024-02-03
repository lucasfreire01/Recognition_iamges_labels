import tensorflow as tf
import numpy as np


class transformer_pipeline:

    def __init__(self, X, y, cat=10):

        assert isinstance(cat, int), "cat must be of type int"

        self.X = X
        self.y = y
        self.cat = cat

    def encoding(self):
        return tf.keras.utils.to_categorical(self.y, self.cat)

    def normalization(self):
        _X = self.X / 255.0
        return _X

    def grayscale(self, _X):

        return np.sum(_X / 3, axis=3, keepdims=True)

    def fit_transform(self):
        _NEW_X = self.normalization()
        _RETURNING_X = self.grayscale(_X=_NEW_X)
        _RETURNING_y = self.encoding()
        return _RETURNING_X, _RETURNING_y
