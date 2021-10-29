import numpy as np
import tensorflow as tf


"""
Class to compute and visualize activation maximizations for different CNN filter layers 
"""


class ActivationMaximizer(object):
    def __init__(self, model, iters=30, lr=10.0):
        self.model = model
        self.iters = iters
        self.learning_rate = lr

    def compute_loss(self, input_image, filter_index):
        activation = self.model(input_image, training=False)
        # We avoid border artifacts by only involving non-border pixels in the loss.
        # TODO is this necessary
        filter_activation = activation[:, 2:-2, 2:-2, 2:-2, filter_index]
        return tf.reduce_mean(filter_activation)

    def initialize_image(self):
        # TODO verify dimensions of 3D images and the expected range of input values
        img = tf.random.uniform(shape=(169, 208, 179))
        return img

    @tf.function
    def gradient_ascent_step(self, img, filter_index, learning_rate):
        with tf.GradientTape() as tape:
            tape.watch(img)
            loss = self.compute_loss(img, filter_index)
        # Compute gradients.
        grads = tape.gradient(loss, img)
        # Normalize gradients.
        grads = tf.math.l2_normalize(grads)
        img += learning_rate * grads
        return loss, img

    def visualize_activation(self, filter_index):
        # We run gradient ascent for 30 steps
        img = self.initialize_image()
        loss = None

        for iteration in range(self.iters):
            loss, img = self.gradient_ascent_step(img, filter_index, self.learning_rate)

        # Decode the resulting input image
        img = self.deprocess_image(img[0].numpy())
        return loss, img

    def deprocess_image(self, img):
        # Normalize array: center on 0., ensure variance is 0.15
        img -= img.mean()
        img /= img.std() + 1e-5
        img *= 0.15

        # Center crop
        img = img[25:-25, 25:-25, 25:-25, :]

        # Clip to [0, 1]
        img += 0.5
        img = np.clip(img, 0, 1)

        # Convert to RGB array
        # img *= 255
        # img = np.clip(img, 0, 255).astype("uint8")
        return img