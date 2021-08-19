__author__ = 'Haohan Wang'

import os
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import losses, optimizers, metrics, layers

import numpy as np
import time
import math
import json
import argparse

from DataGenerator import MRIDataGenerator


class minMaxPool(tf.keras.layers.Layer):
    def __init__(self, pool_size, strides=1):
        super(minMaxPool, self).__init__()
        self.pool_size = pool_size
        self.strides = strides
        self.maxPool = layers.MaxPool3D(self.pool_size)
        # self.normalization = keras.layers.BatchNormalization()

    def call(self, inputs):
        return self.maxPool(inputs) + self.maxPool(-inputs)


class MRIImaging3DConvModel(tf.keras.Model):
    def __init__(self, nClass, args):
        super(MRIImaging3DConvModel, self).__init__()

        self.weights_folder = '../pretrainModels/best_model/fold_' + str(args.idx_fold) + '/npy_weights/'

        self.conv1 = layers.Conv3D(filters=8, kernel_size=3,
                                   weights=self.setConvWeights(0))
        self.bn1 = layers.BatchNormalization(weights=self.setBatchNormWeights(1))
        self.pool1 = layers.MaxPool3D(pool_size=2)

        self.conv2 = layers.Conv3D(filters=16, kernel_size=3,
                                   weights=self.setConvWeights(4))
        self.bn2 = layers.BatchNormalization(weights=self.setBatchNormWeights(5))
        self.pool2 = layers.MaxPool3D(pool_size=2)

        self.conv3 = layers.Conv3D(filters=32, kernel_size=3,
                                   weights=self.setConvWeights(8))
        self.bn3 = layers.BatchNormalization(weights=self.setBatchNormWeights(9))
        self.pool3 = layers.MaxPool3D(pool_size=2)

        self.conv4 = layers.Conv3D(filters=64, kernel_size=3,
                                   weights=self.setConvWeights(12))
        self.bn4 = layers.BatchNormalization(weights=self.setBatchNormWeights(13))
        self.pool4 = layers.MaxPool3D(pool_size=2)

        self.conv5 = layers.Conv3D(filters=128, kernel_size=3,
                                   weights=self.setConvWeights(16))
        self.bn5 = layers.BatchNormalization(weights=self.setBatchNormWeights(17))
        self.pool5 = layers.MaxPool3D(pool_size=2)

        self.gap = layers.GlobalAveragePooling3D()
        self.dp = layers.Dropout(0.5)
        self.dense1 = layers.Dense(units=1200, activation="relu")
        self.dense2 = layers.Dense(units=50, activation="relu")
        self.classifier = layers.Dense(units=nClass, activation="relu")

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.nn.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = tf.nn.relu(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = tf.nn.relu(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = tf.nn.relu(x)
        x = self.pool5(x)
        x = self.gap(x)
        x = self.dp(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.classifier(x)
        return x

    def setConvWeights(self, c):
        w = np.load(self.weights_folder + 'features.' + str(c) + '.weight.npy').astype(np.float32)
        b = np.load(self.weights_folder + 'features.' + str(c) + '.bias.npy').astype(np.float32)
        return [w.T, b]

    def setBatchNormWeights(self, c):
        w = np.load(self.weights_folder + 'features.' + str(c) + '.weight.npy').astype(np.float32)
        b = np.load(self.weights_folder + 'features.' + str(c) + '.bias.npy').astype(np.float32)
        m = np.load(self.weights_folder + 'features.' + str(c) + '.running_mean.npy').astype(np.float32)
        v = np.load(self.weights_folder + 'features.' + str(c) + '.running_var.npy').astype(np.float32)

        return [w, b, m, v]


def train(args):
    num_classes = 2

    transform = None

    trainData = MRIDataGenerator('/media/haohanwang/Info/ADNI_CAPS',
                                 split='train',
                                 batchSize=args.batch_size,
                                 MCI_included=True,
                                 MCI_included_as_soft_label=True,
                                 idx_fold=args.idx_fold,
                                 transform=transform)

    validationData = MRIDataGenerator('/media/haohanwang/Info/ADNI_CAPS',
                                      batchSize=args.batch_size,
                                      idx_fold=args.idx_fold,
                                      split='val')

    testData = MRIDataGenerator('/media/haohanwang/Info/ADNI_CAPS',
                                batchSize=args.batch_size,
                                idx_fold=args.idx_fold,
                                split='test')

    model = MRIImaging3DConvModel(nClass=num_classes, args=args)

    opt = optimizers.Adam(learning_rate=5e-5)
    loss_fn = losses.CategoricalCrossentropy(from_logits=True)
    train_acc_metric = metrics.CategoricalAccuracy()
    val_acc_metric = metrics.CategoricalAccuracy()

    model.build(input_shape=(1, 169, 208, 179, 1))
    print(model.summary())

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            # embedding = model_encoder(x, training=True)
            # logits = model_decoder(embedding, training=True)
            logits = model(x, training=True)
            loss_value = loss_fn(y, logits)

        grads = tape.gradient(loss_value, model.trainable_weights)
        opt.apply_gradients(zip(grads, model.trainable_weights))

        train_acc_metric.update_state(y, logits)

        return loss_value

    @tf.function
    def test_step(x, y):
        # embedding = model_encoder(x, training=False)
        # val_logits = model_decoder(embedding, training=False)
        val_logits = model(x, training=False)
        val_acc_metric.update_state(y, val_logits)

    total_step_train = math.ceil(len(trainData) / args.batch_size)
    total_step_val = math.ceil(len(validationData) / args.batch_size)
    total_step_test = math.ceil(len(testData) / args.batch_size)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        for i in range(total_step_train):
            images, labels = trainData[i]
            loss_value = train_step(images, labels)
            train_acc = train_acc_metric.result()

            print("Training loss %.4f at step %d/%d at Epoch %d with current accuracy %.4f" % (
                float(loss_value), int(i + 1), total_step_train, epoch, train_acc), end='\r')

        train_acc = train_acc_metric.result()
        print(
            "\n\tEpoch %d, Training loss %.4f and acc over epoch: %.4f" % (epoch, float(loss_value), float(train_acc)),
            end='\t')

        train_acc_metric.reset_states()
        print("with: %.2f seconds" % (time.time() - start_time), end='\t')

        for i in range(total_step_val):
            images, labels = validationData[i]
            test_step(images, labels)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))

        for i in range(total_step_test):
            images, labels = testData[i]
            test_step(images, labels)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("\t\tTest acc: %.4f" % (float(val_acc),))

        sys.stdout.flush()

    model.save_weights('weights/weights')


def main(args):
    train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=20, help='How many epochs to run in total?')
    parser.add_argument('-b', '--batch_size', type=int, default=8,
                        help='Batch size during training per GPU')
    parser.add_argument('-a', '--action', type=int, default=0, help='action to take')
    parser.add_argument('-s', '--seed', type=int, default=1, help='random seed')
    parser.add_argument('-i', '--idx_fold', type=int, default=0, help='which partition of data to use')

    args = parser.parse_args()

    tf.random.set_seed(100)
    np.random.seed(1)

    # pretty print args
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))
    main(args)
