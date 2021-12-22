import tensorflow as tf
from tensorflow.keras import losses

import numpy as np


class SaliencyMapGenerator(object):
    def __init__(self, model):
        self.model = model

# generic class to generate saliency map for all subjects in the (batched) dataset
    def generate(self, dataset, total_batch_steps, save_dir, subject_ids_included=False):
        subject_ids = None
        for i in range(total_batch_steps):
            if subject_ids_included:
                images, labels, subject_ids, session_ids = dataset[i]
            else:
                images, labels = dataset[i]
            images = tf.convert_to_tensor(images)
            with tf.GradientTape() as tape:
                tape.watch(images)

                prediction = self.model(images, training=False)

                loss = losses.CategoricalCrossentropy(from_logits=True)(labels, prediction)

            gradients = tape.gradient(loss, images)
            gradients = gradients.numpy()

            if subject_ids is None:
                subject_ids = [range(len(gradients))]
                session_ids = [range(len(gradients))]

            for gradient, subject_id, session_id in zip(gradients, subject_ids, session_ids):
                # normalize between 0 and 1
                min_val, max_val = np.min(gradient), np.max(gradient)
                smap = (gradient - min_val) / (max_val - min_val + tf.keras.backend.epsilon())

                np.save("{}/vanilla/{}_{}_smap.npy".format(save_dir, subject_id, session_id), smap)
