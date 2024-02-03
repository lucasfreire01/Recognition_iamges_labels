import tensorflow


def load_dataset(return_test=True) -> tuple(tuple):
    if return_test:
        return tensorflow.keras.datasets.cifar10.load_data()
    return tensorflow.keras.datasets.cifar10.load_data()[0]
