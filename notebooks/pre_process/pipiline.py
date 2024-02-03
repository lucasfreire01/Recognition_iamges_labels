class pipiline: 
    def __init__(self):
        pass
    def encoding(x, cat):
        x = tf.keras.utils.to_categorical(x, cat)
            return x
    def normalization(var):
        var_norm = var / 255.0
        return var_norm
    def grayscale(x):
        x= np.sum(x / 3, axis=3, keepdims=True)
        return x
    def random_noise(x):
        return x + 0.1 * np.random.normal(size=x.shape)
    def gaussian_filter_smoothad(x, sigma):
        x = gaussian_filter(x, sigma=sigma)
        return x
    
    