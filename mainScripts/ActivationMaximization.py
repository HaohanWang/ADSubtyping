import numpy as np
import tensorflow as tf
from tensorflow import keras


class ActivationMaximation():
    def __int__(self, model):
        self.feature_extractor = model
        # TODO add parameters for gradient ascent

    def compute_loss(self, input_image, filter_index):
        activation = self.feature_extractor(input_image, training=False)
        # We avoid border artifacts by only involving non-border pixels in the loss.
        filter_activation = activation[:, 2:-2, 2:-2, 2:-2, filter_index]
        return tf.reduce_mean(filter_activation)

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

    def visualize_activation(self, img, filter_index):
        # We run gradient ascent for 30 steps
        iterations = 30
        learning_rate = 10.0  # TODO tune parameters
        for iteration in range(iterations):
            loss, img = self.gradient_ascent_step(img, filter_index, learning_rate)

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