import numpy as np
import tensorflow as tf


DEFAULT_SAVE_PATH = "./activation_visualization/"
"""
Class to compute and visualize activation maximizations for different CNN filter layers 
"""

class ActivationMaximizer(object):
    def __init__(self, model, featureIdx, iters=15, lr=10):
        self.model = model
        self.featureIdx = featureIdx
        self.iters = iters
        self.learning_rate = lr

    def compute_loss(self, images):
        logits = self.model.call(images)
        softmax = tf.keras.activations.softmax(logits)
        return softmax[:,self.featureIdx]

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
                if self.compute_loss(images) > 0.99:
                    break
                loss, computed_images = self.gradient_ascent_step(images, self.learning_rate)

            for img, subject_id, session_id in zip(computed_images, subject_ids, session_ids):
                np.save("{}/{}_{}.npy".format(save_dir, subject_id, session_id), img)

class ActivationAttack(object):
    def __init__(self, model, featureIdx, iters=5, lr=0.0002):
        self.model = model
        self.featureIdx = featureIdx
        self.iters = iters
        self.learning_rate = lr

    def compute_loss(self, images): # todo: this is not actually the attack loss
        logits = self.model.call(images)
        softmax = tf.keras.activations.softmax(logits)
        return softmax[:,self.featureIdx]

    @tf.function
    def gradient_ascent_step(self, images, learning_rate):
        images = tf.convert_to_tensor(images)
        with tf.GradientTape() as tape:
            tape.watch(images)
            loss = self.compute_loss(images)
        # Compute gradients.
        grads = tape.gradient(loss, images).numpy()
        images += learning_rate * tf.sign(grads)

        return loss, images

    def visualize_activation(self, dataset, total_batch_steps, save_dir=DEFAULT_SAVE_PATH):
        for i in range(total_batch_steps):
            images, labels, subject_ids, session_ids = dataset[i]
            computed_images = np.copy(images)
            for iteration in range(self.iters):
                # if self.compute_loss(images) > 0.99:
                #     break
                _, computed_images = self.gradient_ascent_step(computed_images, self.learning_rate)
            img_diff = computed_images - images

            for img, subject_id, session_id in zip(computed_images, subject_ids, session_ids):
                np.save("{}/{}_{}.npy".format(save_dir, subject_id, session_id), img_diff)