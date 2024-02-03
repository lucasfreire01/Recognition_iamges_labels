import tensorflow


def is_gpu_active() -> bool:
    return len(tensorflow.config.list_physical_devices("GPU")) > 0
