import numpy as np
import tensorflow as tf


DEFAULT_SAVE_PATH = "./activation_visualization/"
"""
Class to compute and visualize activation maximizations for different CNN filter layers 
"""

class ActivationMaximizer(object):
    def __init__(self, model, intermediate_ly_idx, featureIdx, iters=15, lr=10):
        self.model = model
        self.layer_idx = intermediate_ly_idx
        self.featureIdx = featureIdx
        self.iters = iters
        self.learning_rate = lr

    def compute_loss(self, images):
        activation = self.model.extract_embedding(images, self.layer_idx)
        # cropping is done scientifically during MRI data postprocessing
        # return tf.reduce_mean(activation)
        return activation[:, self.featureIdx]

    @tf.function
    def gradient_ascent_step(self, images, learning_rate):
        images = tf.convert_to_tensor(images)
        with tf.GradientTape() as tape:
            tape.watch(images)
            loss = self.compute_loss(images)
        # Compute gradients.
        grads = tape.gradient(loss, images).numpy()
        # Normalize gradients.
        grads = tf.math.l2_normalize(grads)
        images += learning_rate * grads
        return loss, images

    def visualize_activation(self, dataset, total_batch_steps, save_dir=DEFAULT_SAVE_PATH):
        for i in range(total_batch_steps):
            images, labels, subject_ids, session_ids = dataset[i]
            for iteration in range(self.iters):
                loss, computed_images = self.gradient_ascent_step(images, self.learning_rate)

            for img, subject_id, session_id in zip(computed_images, subject_ids, session_ids):
                np.save("{}/{}_{}_activation_layer_{}.npy".format(save_dir, subject_id, session_id, self.layer_idx), img)