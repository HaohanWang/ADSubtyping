class MinMaxNormalization(object):
    """Normalizes a tensor between 0 and 1"""

    def __call__(self, image):
        return (image - image.min()) / (image.max() - image.min())

def get_transform(config, parameters):
    f = globals().get(config.transform.name)
    if config.transform.params is None:
        return f()
    else:
        return f(**config.transform.params)
