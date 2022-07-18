__author__ = 'Haohan Wang'

import os
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import tensorflow as tf
from tensorflow.keras import losses, optimizers, metrics, layers
from tensorflow_addons.metrics import F1Score

import numpy as np
import pandas as pd
import time
import math
import json
import argparse
import psutil
from os import makedirs
from os.path import join
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent

from DataGenerator import MRIDataGenerator, MRIDataGenerator_Simple
from dataAugmentation import MRIDataAugmentation

from ActivationMaximization import ActivationMaximizer
from DropBlock3DTF import DropBlock3D, DropBlockFlatten


if psutil.Process().username() == 'haohanwang':
    READ_DIR = '/media/haohanwang/Storage/AlzheimerImagingData/'
    WEIGHTS_DIR = 'weights/'
else:
    READ_DIR = '/home/ec2-user/mnt/home/ec2-user/alzstudy/AlzheimerData/'
    WEIGHTS_DIR = '/home/ec2-user/mnt/home/ec2-user/alzstudy/weights/'


class minMaxPool(tf.keras.layers.Layer):
    def __init__(self, pool_size, strides=1):
        super(minMaxPool, self).__init__()
        self.pool_size = pool_size
        self.strides = strides
        self.maxPool = layers.MaxPool3D(self.pool_size)
        # self.normalization = keras.layers.BatchNormalization()

    def call(self, inputs):
        return self.maxPool(inputs) + self.maxPool(-inputs)

class MinMaxNormalization(object):
    def __call__(self, images):
        return (images - images.min()) / (images.max() - images.min())

class MRIImaging3DConvModel(tf.keras.Model):
    def __init__(self, nClass, args):
        super(MRIImaging3DConvModel, self).__init__()

        # if args.continueEpoch == 0 and args.dropBlock == 0 and args.gradientGuidedDropBlock == 0 and args.dropBlock3D == 0:
        #     self.weights_folder = '../pretrainModels/best_model/fold_' + str(args.idx_fold) + '/npy_weights/'
        #     self.conv1 = layers.Conv3D(filters=8, kernel_size=3,
        #                                weights=self.setConvWeights(0))
        #     self.bn1 = layers.BatchNormalization(weights=self.setBatchNormWeights(1))
        #     if args.minmax:
        #         self.pool1 = minMaxPool(pool_size=2)
        #     else:
        #         self.pool1 = layers.MaxPool3D(pool_size=2)
        #
        #     self.conv2 = layers.Conv3D(filters=16, kernel_size=3,
        #                                weights=self.setConvWeights(4))
        #     self.bn2 = layers.BatchNormalization(weights=self.setBatchNormWeights(5))
        #     if args.minmax:
        #         self.pool2 = minMaxPool(pool_size=2)
        #     else:
        #         self.pool2 = layers.MaxPool3D(pool_size=2)
        #
        #     self.conv3 = layers.Conv3D(filters=32, kernel_size=3,
        #                                weights=self.setConvWeights(8))
        #     self.bn3 = layers.BatchNormalization(weights=self.setBatchNormWeights(9))
        #     if args.minmax:
        #         self.pool3 = minMaxPool(pool_size=2)
        #     else:
        #         self.pool3 = layers.MaxPool3D(pool_size=2)
        #
        #     self.conv4 = layers.Conv3D(filters=64, kernel_size=3,
        #                                weights=self.setConvWeights(12))
        #     self.bn4 = layers.BatchNormalization(weights=self.setBatchNormWeights(13))
        #     if args.minmax:
        #         self.pool4 = minMaxPool(pool_size=2)
        #     else:
        #         self.pool4 = layers.MaxPool3D(pool_size=2)
        #
        #     self.conv5 = layers.Conv3D(filters=128, kernel_size=3,
        #                                weights=self.setConvWeights(16))
        #     self.bn5 = layers.BatchNormalization(weights=self.setBatchNormWeights(17))
        #     if args.minmax:
        #         self.pool5 = minMaxPool(pool_size=2)
        #     else:
        #         self.pool5 = layers.MaxPool3D(pool_size=2)
        #
        #     # Dropblock 3D for feature maps
        #     self.dropblock = DropBlock3D(keep_prob=0.5, block_size=3)
        #     self.dropblock_flatten = DropBlockFlatten(keep_prob=0.5, block_size=8*8*8)
        #
        #     self.flatten = layers.Flatten()
        #
        #     self.dp = layers.Dropout(0.3)
        #     self.dense1 = layers.Dense(units=1024, activation="relu")
        #     self.dense2 = layers.Dense(units=128, activation="relu")
        #     self.classifier = layers.Dense(units=nClass, activation="relu")

        initializer = tf.keras.initializers.HeNormal()

        self.conv1 = layers.Conv3D(filters=8, kernel_size=3, input_shape=(169, 208, 179), kernel_initializer=initializer)
        self.bn1 = layers.BatchNormalization()
        if args.minmax:
            self.pool1 = minMaxPool(pool_size=2)
        else:
            self.pool1 = layers.MaxPool3D(pool_size=2)

        self.conv2 = layers.Conv3D(filters=16, kernel_size=3, kernel_initializer=initializer)
        self.bn2 = layers.BatchNormalization()
        if args.minmax:
            self.pool2 = minMaxPool(pool_size=2)
        else:
            self.pool2 = layers.MaxPool3D(pool_size=2)

        self.conv3 = layers.Conv3D(filters=32, kernel_size=3, kernel_initializer=initializer)
        self.bn3 = layers.BatchNormalization()
        if args.minmax:
            self.pool3 = minMaxPool(pool_size=2)
        else:
            self.pool3 = layers.MaxPool3D(pool_size=2)

        self.conv4 = layers.Conv3D(filters=64, kernel_size=3, kernel_initializer=initializer)
        self.bn4 = layers.BatchNormalization()
        if args.minmax:
            self.pool4 = minMaxPool(pool_size=2)
        else:
            self.pool4 = layers.MaxPool3D(pool_size=2)

        self.conv5 = layers.Conv3D(filters=128, kernel_size=3, kernel_initializer=initializer)
        self.bn5 = layers.BatchNormalization()
        if args.minmax:
            self.pool5 = minMaxPool(pool_size=2)
        else:
            self.pool5 = layers.MaxPool3D(pool_size=2)

        # Dropblock for feature maps, before / after the flatten layer
        self.dropblock = DropBlock3D(keep_prob=0.5, block_size=3)
        self.dropblock_flatten = DropBlockFlatten(keep_prob=0.5, block_size=8 * 8 * 8)

        self.flatten = layers.Flatten()
        self.dp = layers.Dropout(0.5)
        self.dense1 = layers.Dense(units=1024, activation="elu")
        self.dense2 = layers.Dense(units=128, activation="elu")
        self.classifier = layers.Dense(units=nClass, activation="elu")

        self.data_aug = MRIDataAugmentation((169, 208, 179), 0.5)

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

        x = self.flatten(x)
        if training and args.dropBlock3D:
            x = tf.reshape(x, [x.shape[0], -1, 1])
            x = self.dropblock_flatten(x)

            x = tf.reshape(x, [x.shape[0], x.shape[1]])

        x = self.dp(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.classifier(x)
        return x

    def extract_embedding(self, inputs):
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
        x = self.flatten(x)
        x = self.dp(x)
        x = self.dense1(x)  # the representation with 1024 values
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

    def calculateGradients(self, x, y):
        x = tf.convert_to_tensor(x)
        with tf.GradientTape() as tape:
            tape.watch(x)
            prediction = self(x, training=False)
            loss = losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(y, prediction)
            # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, x)

        return gradient


def getSaveName(args):
    saveName = ''
    if args.augmented:
        saveName = saveName + '_aug'
        if args.augmented_fancy:
            saveName = saveName + '_fancy'
    if args.mci:
        saveName = saveName + '_mci'
        if args.mci_balanced:
            saveName = saveName + '_balanced'
    if args.pgd != 0:
        saveName = saveName + '_pgd_' + str(args.pgd)
    if args.minmax:
        saveName = saveName + '_mm'
    if args.dropBlock:
        saveName = saveName + '_db'
    if args.worst_sample:
        saveName = saveName + '_ws_' + str(args.worst_sample)
    if args.consistency:
        saveName = saveName + '_con_' + str(args.consistency)

    saveName = saveName + '_fold_' + str(args.idx_fold) + '_seed_' + str(args.seed)
    return saveName


def train(args):
    num_classes = 2
    ## todo: augment some data that are simple to classify at the beginning

    ## todo: let's reorder the samples with age information

    if args.consistency != 0:
        args.batch_size = args.batch_size // 2

    if args.worst_sample == 0:
        trainData = MRIDataGenerator(READ_DIR + 'ADNI_CAPS',
                                     split='train',
                                     batchSize=args.batch_size,
                                     MCI_included=args.mci,
                                     MCI_included_as_soft_label=args.mci_balanced,
                                     idx_fold=args.idx_fold,
                                     augmented=args.augmented,
                                     augmented_fancy=args.augmented_fancy,
                                     dropBlock=args.dropBlock,
                                     dropBlockIterationStart=int(args.continueEpoch*1700/args.batch_size),
                                     gradientGuidedDropBlock=args.gradientGuidedDropBlock,
                                     mci_finetune=args.mci_finetune)
    else:
        trainData = MRIDataGenerator(READ_DIR + 'ADNI_CAPS',
                                     split='train',
                                     batchSize=args.batch_size * args.worst_sample,
                                     MCI_included=args.mci,
                                     MCI_included_as_soft_label=args.mci_balanced,
                                     idx_fold=args.idx_fold,
                                     augmented=args.augmented,
                                     augmented_fancy=args.augmented_fancy,
                                     dropBlock=args.dropBlock,
                                     dropBlockIterationStart=int(args.continueEpoch*1700/args.batch_size),
                                     gradientGuidedDropBlock=args.gradientGuidedDropBlock,
                                     mci_finetune=args.mci_finetune)


    validationData = MRIDataGenerator(READ_DIR + 'ADNI_CAPS',
                                      batchSize=args.batch_size,
                                      idx_fold=args.idx_fold,
                                      split='val',
                                      mci_finetune=args.mci_finetune)

    testData = MRIDataGenerator(READ_DIR + 'ADNI_CAPS',
                                batchSize=args.batch_size,
                                idx_fold=args.idx_fold,
                                split='test',
                                mci_finetune=args.mci_finetune)

    print(f"Size of training data: {len(trainData)}")
    print(f"Size of validation data: {len(validationData)}")
    print(f"Size of test data: {len(testData)}")

    if args.gpu:
        init_gpu(args.gpu)
    strategy = tf.distribute.MirroredStrategy()
    GLOBAL_BATCH_SIZE = args.batch_size * strategy.num_replicas_in_sync

    with strategy.scope():
        model = MRIImaging3DConvModel(nClass=num_classes, args=args)
        # opt = optimizers.Adam(learning_rate=5e-6)
        opt = optimizers.Adam(learning_rate=1e-5)
        loss_fn = losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

        def compute_loss(labels, predictions):
            per_example_loss = loss_fn(labels, predictions)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

        def calculate_loss(x, y):
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                per_example_loss = loss_fn(y, logits)
                return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

        train_acc_metric = metrics.CategoricalAccuracy()
        val_acc_metric = metrics.CategoricalAccuracy()

        def train_step(x, y):
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss_value = compute_loss(y, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            opt.apply_gradients(zip(grads, model.trainable_weights))

            train_acc_metric.update_state(y, logits)
            return loss_value

        def train_step_consistency(x, z, y):
            with tf.GradientTape() as tape:
                logits_1 = model(x, training=True)
                logits_2 = model(z, training=True)
                loss_value_1 = compute_loss(y, logits_1)
                loss_value_2 = compute_loss(y, logits_2)
                loss_value = loss_value_1 + loss_value_2 + args.consistency*tf.norm(logits_1 - logits_2, ord=2)

            grads = tape.gradient(loss_value, model.trainable_weights)
            opt.apply_gradients(zip(grads, model.trainable_weights))

            train_acc_metric.update_state(y, logits_1)
            train_acc_metric.update_state(y, logits_2)

            return loss_value

        def test_step(x, y):
            val_logits = model(x, training=False)
            val_acc_metric.update_state(y, val_logits)

        @tf.function
        def distributed_train_step(dataset_inputs, data_labels):
            per_replica_losses = strategy.run(train_step, args=(dataset_inputs, data_labels))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        @tf.function
        def distributed_train_step_consistency(dataset_inputs_1, dataset_inputs_2, data_labels):
            per_replica_losses = strategy.run(train_step_consistency, args=(dataset_inputs_1, dataset_inputs_2, data_labels))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        @tf.function
        def distributed_test_step(dataset_inputs, data_labels):
            return strategy.run(test_step, args=(dataset_inputs, data_labels))


        total_step_train = math.ceil(len(trainData) / args.batch_size)
        total_step_val = math.ceil(len(validationData) / args.batch_size)
        total_step_test = math.ceil(len(testData) / args.batch_size)

        if args.continueEpoch != 0:
            model.load_weights(WEIGHTS_DIR + args.weights_folder + '/weights' + getSaveName(args) + '_epoch_' + str(args.continueEpoch))
            # model.load_weights(WEIGHTS_DIR + 'weights_regular_training/weights_aug_fold_0_seed_1_epoch_1')
        elif args.dropBlock or args.worst_sample or args.gradientGuidedDropBlock or args.dropBlock3D:
            # dropblock training is too hard, so let's load the previous one to continue as epoch 1
            model.load_weights(WEIGHTS_DIR + 'weights_regular_training/weights_aug_fold_0_seed_1_epoch_50')

        for epoch in range(1, args.epochs + 1):
            if epoch <= args.continueEpoch:
                continue

            start_time = time.time()
            # for i in range(total_step_train):
            #     images, labels = trainData[i]
            #
            #     if args.worst_sample != 0:
            #         losses_total = np.zeros(args.worst_sample*args.batch_size)
            #         for k in range(args.worst_sample):
            #             xtmp = images[k*args.batch_size:(k+1)*args.batch_size]
            #             ytmp = labels[k*args.batch_size:(k+1)*args.batch_size]
            #
            #             loss_batch = calculate_loss(xtmp, ytmp)
            #             losses_total[k*args.batch_size:(k+1)*args.batch_size] = loss_batch
            #
            #         idc1 = np.argsort(-losses_total)[:int(args.batch_size/2)]
            #         idc2 = np.random.choice(range(args.batch_size*args.worst_sample), int(args.batch_size/2), replace=False)
            #         idx = np.append(idc1, idc2)
            #         images = images[idx]
            #         labels = labels[idx]
            #
            #     if args.pgd != 0:
            #         # todo: what's the visual difference between an AD and a normal (what are the differences we need)
            #
            #         if args.consistency == 0:
            #             images += (np.random.random(size=images.shape) * 2 - 1) * args.pgd
            #             for pgd_index in range(5):
            #                 grad = model.calculateGradients(images, labels)
            #                 images += (args.pgd / 5) * np.sign(grad)
            #
            #                 images = np.clip(images,
            #                                  images - args.pgd,
            #                                  images + args.pgd)
            #                 images = np.clip(images, 0, 1)  # ensure valid pixel range
            #         else:
            #             images2 = images + (np.random.random(size=images.shape) * 2 - 1) * args.pgd
            #             for pgd_index in range(5):
            #                 grad = model.calculateGradients(images2, labels)
            #                 images2 += (args.pgd / 5) * np.sign(grad)
            #
            #                 images2 = np.clip(images2,
            #                                  images2 - args.pgd,
            #                                  images2 + args.pgd)
            #                 images2 = np.clip(images2, 0, 1)  # ensure valid pixel range
            #
            #     if args.gradientGuidedDropBlock:
            #         grads = model.calculateGradients(images, labels)
            #         # perform dropblock per sample based on gradients - mutating the training images
            #         images = model.data_aug.augmentData_batch_erasing_grad_guided(images, trainData.dropBlock_iterationCount, grads)
            #         trainData.dropBlock_iterationCount += 1
            #
            #     if args.consistency == 0:
            #         loss_value = distributed_train_step(images, labels)
            #     else:
            #         loss_value = distributed_train_step_consistency(images, images2, labels)
            #         # todo: what will the corresponding one on consistency loss looks like
            #
            #     train_acc = train_acc_metric.result()
            #     print("Training loss %.4f at step %d/%d at Epoch %d with current accuracy %.4f" % (
            #         float(loss_value), int(i + 1), total_step_train, epoch, train_acc), end='\r')

            # train_acc = train_acc_metric.result()
            # print(
            #     "\n\tEpoch %d, Training loss %.4f and acc over epoch: %.4f" % (epoch, float(loss_value), float(train_acc)),
                # end='\t')

            # train_acc_metric.reset_states()
            # print("with: %.2f seconds" % (time.time() - start_time), end='\t')

            for i in range(total_step_val):
                images, labels = validationData[i]
                distributed_test_step(images, labels)

            val_acc = val_acc_metric.result()
            val_acc_metric.reset_states()
            print("Validation acc: %.4f" % (float(val_acc),))

            for i in range(total_step_test):
                images, labels = testData[i]
                distributed_test_step(images, labels)

            val_acc = val_acc_metric.result()
            val_acc_metric.reset_states()
            print("\t\tTest acc: %.4f" % (float(val_acc),))

            sys.stdout.flush()

            model.save_weights(WEIGHTS_DIR + args.weights_folder + '/weights' + getSaveName(args) + '_epoch_' + str(epoch))


def evaluate_crossDataSet(args):
    num_classes = 2

    ADNI_testData = MRIDataGenerator(READ_DIR + 'ADNI_CAPS',
                                     batchSize=args.batch_size,
                                     idx_fold=args.idx_fold,
                                     split='test')

    AIBL_testData = MRIDataGenerator_Simple(READ_DIR + 'AIBL_CAPS',
                                            'aibl_info.csv', batchSize=args.batch_size)

    MIRIAD_testData = MRIDataGenerator_Simple(READ_DIR + 'MIRIAD_CAPS',
                                              'miriad_test_info.csv', batchSize=args.batch_size)

    OASIS3_testData = MRIDataGenerator_Simple(READ_DIR + 'OASIS3_CAPS',
                                              'oasis3_test_info_2.csv', batchSize=args.batch_size)

    model = MRIImaging3DConvModel(nClass=num_classes, args=args)

    val_acc_metric = metrics.CategoricalAccuracy()
    val_f1_metric = F1Score(num_classes=2)

    @tf.function
    def test_step(x, y):
        val_logits = model(x, training=False)
        val_acc_metric.update_state(y, val_logits)
        val_f1_metric.update_state(y, val_logits)

    total_step_test_ADNI = math.ceil(len(ADNI_testData) / args.batch_size)
    total_step_test_AIBL = math.ceil(len(AIBL_testData) / args.batch_size)
    total_step_test_MIRIAD = math.ceil(len(MIRIAD_testData) / args.batch_size)
    total_step_test_OASIS3 = math.ceil(len(OASIS3_testData) / args.batch_size)

    model.load_weights(
        WEIGHTS_DIR + args.weights_folder + '/weights' + getSaveName(args) + '_epoch_' + str(args.continueEpoch))

    print('Testing Start ...')

    for i in range(total_step_test_ADNI):
        images, labels = ADNI_testData[i]
        test_step(images, labels)
    val_acc = val_acc_metric.result()
    val_f1 = val_f1_metric.result()[1]
    val_acc_metric.reset_states()
    val_f1_metric.reset_states()
    print("\tADNI Test acc: %.4f" % (float(val_acc),))
    print("\tADNI Test F1: %.4f" % (float(val_f1),))

    for i in range(total_step_test_AIBL):
        images, labels = AIBL_testData[i]
        test_step(images, labels)
    val_acc = val_acc_metric.result()
    val_f1 = val_f1_metric.result()[1]
    val_acc_metric.reset_states()
    val_f1_metric.reset_states()
    print("\tAIBL Test acc: %.4f" % (float(val_acc),))
    print("\tAIBL Test F1: %.4f" % (float(val_f1),))

    for i in range(total_step_test_MIRIAD):
        images, labels = MIRIAD_testData[i]
        test_step(images, labels)
    val_acc = val_acc_metric.result()
    val_f1 = val_f1_metric.result()[1]
    val_acc_metric.reset_states()
    val_f1_metric.reset_states()
    print("\tMIRIAD Test acc: %.4f" % (float(val_acc),))
    print("\tMIRIAD Test F1: %.4f" % (float(val_f1),))

    for i in range(total_step_test_OASIS3):
        images, labels = OASIS3_testData[i]
        test_step(images, labels)
    val_acc = val_acc_metric.result()
    val_f1 = val_f1_metric.result()[1]
    val_acc_metric.reset_states()
    val_f1_metric.reset_states()
    print("\tOASIS3 Test acc: %.4f" % (float(val_acc),))
    print("\tOASIS3 Test F1: %.4f" % (float(val_f1),))

    sys.stdout.flush()


def evaluate_crossDataSet_at_individual(args):
    def writeOutResults(dataset, prediction, subjectIDs, sessionIDs):
        info = {}
        if dataset == 'ADNI':
            tmp = [line.strip() for line in
                   open(READ_DIR + 'ADNI_CAPS/split.pretrained.0.csv')]
            for line in tmp:
                if line.find('test') != -1:
                    items = line.split(',')
                    info[items[0] + '#' + items[1]] = line

        elif dataset == 'AIBL':
            text = [line.strip() for line in
                    open(READ_DIR + 'AIBL_CAPS/aibl_info.csv')]
            for line in text:
                items = line.split(',')
                info[items[0] + '#' + items[1]] = line

        elif dataset == 'MIRIAD':
            text = [line.strip() for line in
                    open(READ_DIR + 'MIRIAD_CAPS/miriad_test_info.csv')]
            for line in text:
                items = line.split(',')
                info[items[0] + '#' + items[1]] = line

        elif dataset == 'OASIS3':
            text = [line.strip() for line in
                    open(READ_DIR + 'OASIS3_CAPS/oasis3_test_info_2.csv')]
            for line in text:
                items = line.split(',')
                info[items[0] + '#' + items[1]] = line

        f = open('predictionResults/result' + getSaveName(args) + '_epoch_' + str(
            args.continueEpoch) + '_' + dataset + '.csv', 'w')
        pl = prediction.tolist()
        for i in range(len(pl)):
            line = info[subjectIDs[i] + '#' + sessionIDs[i]]
            f.writelines(line + ',')
            if pl[i] == 0:
                f.writelines('CN\n')
            else:
                f.writelines('AD\n')
        f.close()

    tf.config.run_functions_eagerly(True)

    num_classes = 2

    ADNI_testData = MRIDataGenerator(READ_DIR + 'ADNI_CAPS',
                                     batchSize=args.batch_size,
                                     idx_fold=args.idx_fold,
                                     split='test',
                                     returnSubjectID=True)

    AIBL_testData = MRIDataGenerator_Simple(READ_DIR + 'AIBL_CAPS',
                                            'aibl_info.csv', batchSize=args.batch_size, returnSubjectID=True)

    MIRIAD_testData = MRIDataGenerator_Simple(READ_DIR + 'MIRIAD_CAPS',
                                              'miriad_test_info.csv', batchSize=args.batch_size, returnSubjectID=True)

    OASIS3_testData = MRIDataGenerator_Simple(READ_DIR + 'OASIS3_CAPS',
                                              'oasis3_test_info_2.csv', batchSize=args.batch_size, returnSubjectID=True)

    model = MRIImaging3DConvModel(nClass=num_classes, args=args)

    val_acc_metric = metrics.CategoricalAccuracy()
    val_f1_metric = F1Score(num_classes=2)

    @tf.function
    def test_step(x, y):
        val_logits = model(x, training=False)
        val_acc_metric.update_state(y, val_logits)
        val_f1_metric.update_state(y, val_logits)
        return tf.math.argmax(val_logits, 1).numpy()

    total_step_test_ADNI = math.ceil(len(ADNI_testData) / args.batch_size)
    total_step_test_AIBL = math.ceil(len(AIBL_testData) / args.batch_size)
    total_step_test_MIRIAD = math.ceil(len(MIRIAD_testData) / args.batch_size)
    total_step_test_OASIS3 = math.ceil(len(OASIS3_testData) / args.batch_size)

    model.load_weights(
        WEIGHTS_DIR + args.weights_folder + '/weights' + getSaveName(args) + '_epoch_' + str(args.continueEpoch))

    print('Testing Start ...')

    prediction = None
    subjectIDs = []
    sessionIDs = []

    for i in range(total_step_test_ADNI):
        images, labels, subjects, sessions = ADNI_testData[i]
        prediction_tmp = test_step(images, labels)
        if prediction is None:
            prediction = prediction_tmp
        else:
            prediction = np.append(prediction, prediction_tmp)
        subjectIDs.extend(subjects)
        sessionIDs.extend(sessions)

    val_acc = val_acc_metric.result()
    val_f1 = val_f1_metric.result()[1]
    val_acc_metric.reset_states()
    val_f1_metric.reset_states()
    print("\tADNI Test acc: %.4f" % (float(val_acc),))
    print("\tADNI Test F1: %.4f" % (float(val_f1),))

    writeOutResults('ADNI', prediction, subjectIDs, sessionIDs)

    prediction = None
    subjectIDs = []
    sessionIDs = []

    for i in range(total_step_test_AIBL):
        images, labels, subjects, sessions = AIBL_testData[i]
        prediction_tmp = test_step(images, labels)
        if prediction is None:
            prediction = prediction_tmp
        else:
            prediction = np.append(prediction, prediction_tmp)
        subjectIDs.extend(subjects)
        sessionIDs.extend(sessions)

    val_acc = val_acc_metric.result()
    val_f1 = val_f1_metric.result()[1]
    val_acc_metric.reset_states()
    val_f1_metric.reset_states()
    print("\tAIBL Test acc: %.4f" % (float(val_acc),))
    print("\tAIBL Test F1: %.4f" % (float(val_f1),))

    writeOutResults('AIBL', prediction, subjectIDs, sessionIDs)

    prediction = None
    subjectIDs = []
    sessionIDs = []

    for i in range(total_step_test_MIRIAD):
        images, labels, subjects, sessions = MIRIAD_testData[i]
        prediction_tmp = test_step(images, labels)
        if prediction is None:
            prediction = prediction_tmp
        else:
            prediction = np.append(prediction, prediction_tmp)
        subjectIDs.extend(subjects)
        sessionIDs.extend(sessions)

    val_acc = val_acc_metric.result()
    val_f1 = val_f1_metric.result()[1]
    val_acc_metric.reset_states()
    val_f1_metric.reset_states()
    print("\tMIRIAD Test acc: %.4f" % (float(val_acc),))
    print("\tMIRIAD Test F1: %.4f" % (float(val_f1),))

    writeOutResults('MIRIAD', prediction, subjectIDs, sessionIDs)

    prediction = None
    subjectIDs = []
    sessionIDs = []

    for i in range(total_step_test_OASIS3):
        images, labels, subjects, sessions = OASIS3_testData[i]
        prediction_tmp = test_step(images, labels)
        if prediction is None:
            prediction = prediction_tmp
        else:
            prediction = np.append(prediction, prediction_tmp)
        subjectIDs.extend(subjects)
        sessionIDs.extend(sessions)

    val_acc = val_acc_metric.result()
    val_f1 = val_f1_metric.result()[1]
    val_acc_metric.reset_states()
    val_f1_metric.reset_states()
    print("\tOASIS3 Test acc: %.4f" % (float(val_acc),))
    print("\tOASIS3 Test F1: %.4f" % (float(val_f1),))

    writeOutResults('OASIS3', prediction, subjectIDs, sessionIDs)
    sys.stdout.flush()


def embedding_extractor(args):
    # todo: so the embedings have some clear clustering structure,
    #  and the clustering structure will not disappear for different models
    #  but the clustering structure is only there when visualized by TSNE

    def saveEmebddings(dataset, embedding, subjectIDs, sessionIDs, split='test'):
        info = {}
        if dataset == 'ADNI':
            tmp = [line.strip() for line in
                   open(READ_DIR + 'ADNI_CAPS/split.pretrained.0.csv')]
            for line in tmp:
                if line.find(split) != -1:
                    items = line.split(',')
                    info[items[0] + '#' + items[1]] = line

        elif dataset == 'AIBL':
            text = [line.strip() for line in
                    open(READ_DIR + 'AIBL_CAPS/aibl_info.csv')]
            for line in text:
                items = line.split(',')
                info[items[0] + '#' + items[1]] = line

        elif dataset == 'MIRIAD':
            text = [line.strip() for line in
                    open(READ_DIR + 'MIRIAD_CAPS/miriad_test_info.csv')]
            for line in text:
                items = line.split(',')
                info[items[0] + '#' + items[1]] = line

        elif dataset == 'OASIS3':
            text = [line.strip() for line in
                    open(READ_DIR + 'OASIS3_CAPS/oasis3_test_info_2.csv')]
            for line in text:
                items = line.split(',')
                info[items[0] + '#' + items[1]] = line

        if split == 'train' or split == 'val':
            np.save(READ_DIR + 'embeddingResult/result' + getSaveName(args) + '_epoch_' + str(
                args.continueEpoch) + '_' + dataset + '_' + split + '.npy', embedding)
            f = open(READ_DIR + 'embeddingResult/result' + getSaveName(args) + '_epoch_' + str(
                args.continueEpoch) + '_' + dataset + '_' + split + '.csv', 'w')
        else:
            np.save(READ_DIR + 'embeddingResult/result' + getSaveName(args) + '_epoch_' + str(
                args.continueEpoch) + '_' + dataset + '.npy', embedding)
            f = open(READ_DIR + 'embeddingResult/result' + getSaveName(args) + '_epoch_' + str(
                args.continueEpoch) + '_' + dataset + '.csv', 'w')

        for i in range(len(subjectIDs)):
            f.writelines(subjectIDs[i] + ',' + sessionIDs[i] + '\n')
        f.close()

    tf.config.run_functions_eagerly(True)

    num_classes = 2

    ADNI_trainData = MRIDataGenerator(READ_DIR + 'ADNI_CAPS',
                                      batchSize=args.batch_size,
                                      idx_fold=args.idx_fold,
                                      split='train',
                                      returnSubjectID=True,
                                      mci_finetune=args.mci_finetune)

    ADNI_valData = MRIDataGenerator(READ_DIR + 'ADNI_CAPS',
                                    batchSize=args.batch_size,
                                    idx_fold=args.idx_fold,
                                    split='val',
                                    returnSubjectID=True,
                                    mci_finetune=args.mci_finetune)

    ADNI_testData = MRIDataGenerator(READ_DIR + 'ADNI_CAPS',
                                     batchSize=args.batch_size,
                                     idx_fold=args.idx_fold,
                                     split='test',
                                     returnSubjectID=True,
                                     mci_finetune=args.mci_finetune)

    AIBL_testData = MRIDataGenerator_Simple(READ_DIR + 'AIBL_CAPS',
                                            'aibl_info.csv', batchSize=args.batch_size, returnSubjectID=True)

    MIRIAD_testData = MRIDataGenerator_Simple(READ_DIR + 'MIRIAD_CAPS',
                                              'miriad_test_info.csv', batchSize=args.batch_size, returnSubjectID=True)

    OASIS3_testData = MRIDataGenerator_Simple(READ_DIR + 'OASIS3_CAPS',
                                              'oasis3_test_info_2.csv', batchSize=args.batch_size, returnSubjectID=True)

    model = MRIImaging3DConvModel(nClass=num_classes, args=args)

    @tf.function
    def extract_embedding(x):
        embedding = model.extract_embedding(x)
        return embedding.numpy()

    total_step_train_ADNI = math.ceil(len(ADNI_trainData) / args.batch_size)
    total_step_val_ADNI = math.ceil(len(ADNI_valData) / args.batch_size)
    total_step_test_ADNI = math.ceil(len(ADNI_testData) / args.batch_size)
    total_step_test_AIBL = math.ceil(len(AIBL_testData) / args.batch_size)
    total_step_test_MIRIAD = math.ceil(len(MIRIAD_testData) / args.batch_size)
    total_step_test_OASIS3 = math.ceil(len(OASIS3_testData) / args.batch_size)

    model.load_weights(
        WEIGHTS_DIR + args.weights_folder + '/weights' + getSaveName(args) + '_epoch_' + str(args.continueEpoch))

    print('Testing Start ...')

    embedding = None
    subjectIDs = []
    sessionIDs = []

    for i in range(total_step_train_ADNI):
        images, labels, subjects, sessions = ADNI_trainData[i]
        embedding_tmp = extract_embedding(images)
        if embedding is None:
            embedding = embedding_tmp
        else:
            embedding = np.append(embedding, embedding_tmp, 0)
        subjectIDs.extend(subjects)
        sessionIDs.extend(sessions)

    saveEmebddings('ADNI', embedding, subjectIDs, sessionIDs, 'train')

    embedding = None
    subjectIDs = []
    sessionIDs = []

    for i in range(total_step_val_ADNI):
        images, labels, subjects, sessions = ADNI_valData[i]
        embedding_tmp = extract_embedding(images)
        if embedding is None:
            embedding = embedding_tmp
        else:
            embedding = np.append(embedding, embedding_tmp, 0)
        subjectIDs.extend(subjects)
        sessionIDs.extend(sessions)

    saveEmebddings('ADNI', embedding, subjectIDs, sessionIDs, 'val')

    embedding = None
    subjectIDs = []
    sessionIDs = []

    for i in range(total_step_test_ADNI):
        images, labels, subjects, sessions = ADNI_testData[i]
        embedding_tmp = extract_embedding(images)
        if embedding is None:
            embedding = embedding_tmp
        else:
            embedding = np.append(embedding, embedding_tmp, 0)
        subjectIDs.extend(subjects)
        sessionIDs.extend(sessions)

    saveEmebddings('ADNI', embedding, subjectIDs, sessionIDs)

    embedding = None
    subjectIDs = []
    sessionIDs = []

    for i in range(total_step_test_AIBL):
        images, labels, subjects, sessions = AIBL_testData[i]
        embedding_tmp = extract_embedding(images)
        if embedding is None:
            embedding = embedding_tmp
        else:
            embedding = np.append(embedding, embedding_tmp, 0)
        subjectIDs.extend(subjects)
        sessionIDs.extend(sessions)

    saveEmebddings('AIBL', embedding, subjectIDs, sessionIDs)

    embedding = None
    subjectIDs = []
    sessionIDs = []

    for i in range(total_step_test_MIRIAD):
        images, labels, subjects, sessions = MIRIAD_testData[i]
        embedding_tmp = extract_embedding(images)
        if embedding is None:
            embedding = embedding_tmp
        else:
            embedding = np.append(embedding, embedding_tmp, 0)
        subjectIDs.extend(subjects)
        sessionIDs.extend(sessions)

    saveEmebddings('MIRIAD', embedding, subjectIDs, sessionIDs)

    embedding = None
    subjectIDs = []
    sessionIDs = []

    for i in range(total_step_test_OASIS3):
        images, labels, subjects, sessions = OASIS3_testData[i]
        embedding_tmp = extract_embedding(images)
        if embedding is None:
            embedding = embedding_tmp
        else:
            embedding = np.append(embedding, embedding_tmp, 0)
        subjectIDs.extend(subjects)
        sessionIDs.extend(sessions)

    saveEmebddings('OASIS3', embedding, subjectIDs, sessionIDs)

    sys.stdout.flush()


# a direct migration from visualization/saliency.py with TF implementation
def saliency(args):

    def generate_saliency_map(model, data_gen, total_batch_steps, save_dir, split, extract_features=False):
        label_list = []
        sub_list = []
        sess_list = []
        prob_list = []

        for i in range(total_batch_steps):
            images, labels, subject_ids, session_ids = data_gen[i]
            images = tf.convert_to_tensor(images)

            label_list.extend(labels)
            sub_list.extend(subject_ids)
            sess_list.extend(session_ids)

            with tf.GradientTape() as tape:
                tape.watch(images)

                if extract_features:
                    outputs = model.extract_embedding(images)
                else:
                    outputs = model(images, training=False)
                prob = tf.keras.activations.softmax(outputs)
                prob_list.append(prob.numpy())
                loss = losses.CategoricalCrossentropy(from_logits=True)(labels, outputs)

            gradients = tape.gradient(loss, images)
            gradients = gradients.numpy()

            saliency_maps = tf.reduce_max(tf.abs(gradients), axis=-1)

            for sal, subject_id, session_id in zip(saliency_maps, subject_ids, session_ids):
                makedirs(join(save_dir, split, subject_id), exist_ok=True)
                np.save(join(save_dir, split, subject_id, session_id + '.npy'), sal)

        df = pd.DataFrame(columns=['participant_id', 'session_id', 'diagnosis', 'prob_AD'])
        df['participant_id'] = sub_list
        df['session_id'] = sess_list
        df['diagnosis'] = list(map(lambda ts: 0 if ts[0] == 1 else 1, label_list))
        df['prob_AD'] = np.concatenate(prob_list, axis=0)[:, 1]
        df.to_csv(join(save_dir, split + '_saliency_info.csv'), index=None)


    num_classes = 2
    model = MRIImaging3DConvModel(nClass=num_classes, args=args)
    model.load_weights(WEIGHTS_DIR + args.weights_folder + '/weights' + getSaveName(args) + '_epoch_' + str(args.continueEpoch))

    ADNI_trainData = MRIDataGenerator(READ_DIR + 'ADNI_CAPS',
                                      transform=MinMaxNormalization(),
                                      batchSize=args.batch_size,
                                      idx_fold=args.idx_fold,
                                      split='train',
                                      returnSubjectID=True,
                                      mci_finetune=args.mci_finetune)
    ADNI_valData = MRIDataGenerator(READ_DIR + 'ADNI_CAPS',
                                    transform=MinMaxNormalization(),
                                    batchSize=args.batch_size,
                                    idx_fold=args.idx_fold,
                                    split='val',
                                    returnSubjectID=True,
                                    mci_finetune=args.mci_finetune)
    ADNI_testData = MRIDataGenerator(READ_DIR + 'ADNI_CAPS',
                                     transform=MinMaxNormalization(),
                                     batchSize=args.batch_size,
                                     idx_fold=args.idx_fold,
                                     split='test',
                                     returnSubjectID=True,
                                     mci_finetune=args.mci_finetune)

    total_step_train = math.ceil(len(ADNI_trainData) / args.batch_size)
    total_step_val = math.ceil(len(ADNI_valData) / args.batch_size)
    total_step_test = math.ceil(len(ADNI_testData) / args.batch_size)
    makedirs(args.saliency_save_dir, exist_ok=True)

    generate_saliency_map(model, ADNI_trainData, total_step_train, args.saliency_save_dir, "train", extract_features=args.extract_features)
    generate_saliency_map(model, ADNI_valData, total_step_val, args.saliency_save_dir, "val", extract_features=args.extract_features)
    generate_saliency_map(model, ADNI_testData, total_step_test, args.saliency_save_dir, "test", extract_features=args.extract_features)


# a direct migration from visualization/visualize.py with TF implementation
def attack_visualization(args):

    def visualize_single_attack_image(model, data_gen, total_batch_steps, save_dir, split, extract_features=False):
        label_list = []
        sub_list = []
        sess_list = []
        prob_list = []
        prob_attack_list = []

        for i in range(total_batch_steps):
            images, labels, subject_ids, session_ids = data_gen[i]

            images = tf.convert_to_tensor(images)

            images_attack = projected_gradient_descent(model, images, eps=0.05, eps_iter=0.00125, nb_iter=10, norm=np.inf, rand_init=True)

            label_list.extend(labels)
            sub_list.extend(subject_ids)
            sess_list.extend(session_ids)

            with tf.GradientTape() as tape:
                tape.watch(images)

                if extract_features:
                    outputs = model.extract_embedding(images)
                    outputs_attack = model.extract_embedding(images_attack)
                else:
                    outputs = model(images, training=True)
                    outputs_attack = model(images_attack, training=True)

                prob = tf.keras.activations.softmax(outputs)
                prob_attack = tf.keras.activations.softmax(outputs_attack)
                prob_list.append(prob.numpy())
                prob_attack_list.append(prob_attack.numpy())

            images_diff_abs = tf.abs(tf.squeeze(images_attack) - tf.squeeze(images)).numpy()
            for diff, subject_id, session_id in zip(images_diff_abs, subject_ids, session_ids):
                makedirs(join(save_dir, split, subject_id), exist_ok=True)
                np.save(join(save_dir, split, subject_id, session_id + '.npy'), diff)

        prob_list = np.vstack(prob_list)
        prob_attack_list = np.vstack(prob_attack_list)

        df = pd.DataFrame(columns=['participant_id', 'session_id', 'diagnosis', 'prob_AD', 'prob_AD_attack'])
        df['participant_id'] = sub_list
        df['session_id'] = sess_list
        df['diagnosis'] = list(map(lambda ts: 0 if ts[0] == 1 else 1, label_list))
        df['prob_AD'] = prob_list[:, 1]
        df['prob_AD_attack'] = prob_attack_list[:, 1]

        df.to_csv(join(save_dir, split + '_attack_info.csv'), index=None)


    num_classes = 2
    model = MRIImaging3DConvModel(nClass=num_classes, args=args)
    model.load_weights(
        WEIGHTS_DIR + args.weights_folder + '/weights' + getSaveName(args) + '_epoch_' + str(args.continueEpoch))

    batch_size = args.batch_size // 2

    ADNI_trainData = MRIDataGenerator(READ_DIR + 'ADNI_CAPS',
                                      transform=MinMaxNormalization(),
                                      batchSize=batch_size,
                                      idx_fold=args.idx_fold,
                                      split='train',
                                      returnSubjectID=True)

    ADNI_valData = MRIDataGenerator(READ_DIR + 'ADNI_CAPS',
                                    transform=MinMaxNormalization(),
                                    batchSize=batch_size,
                                    idx_fold=args.idx_fold,
                                    split='val',
                                    returnSubjectID=True)

    ADNI_testData = MRIDataGenerator(READ_DIR + 'ADNI_CAPS',
                                     transform=MinMaxNormalization(),
                                     batchSize=batch_size,
                                     idx_fold=args.idx_fold,
                                     split='test',
                                     returnSubjectID=True)

    total_step_train = math.ceil(len(ADNI_trainData) / batch_size)
    total_step_val = math.ceil(len(ADNI_valData) / batch_size)
    total_step_test = math.ceil(len(ADNI_testData) / batch_size)
    makedirs(args.attack_visualization_save_dir, exist_ok=True)

    visualize_single_attack_image(model, ADNI_trainData, total_step_train, args.attack_visualization_save_dir, 'train', extract_features=args.extract_features)
    visualize_single_attack_image(model, ADNI_valData, total_step_val, args.attack_visualization_save_dir, 'val', extract_features=args.extract_features)
    visualize_single_attack_image(model, ADNI_testData, total_step_test, args.attack_visualization_save_dir, 'test', extract_features=args.extract_features)


def activation_maximization_visualize(args):
    num_classes = 2

    tf.config.run_functions_eagerly(True)
    model = MRIImaging3DConvModel(nClass=num_classes, args=args)

    model.load_weights(
        WEIGHTS_DIR + args.weights_folder + '/weights' + getSaveName(args) + '_epoch_' + str(args.continueEpoch))

    ADNI_testData = MRIDataGenerator(READ_DIR + 'ADNI_CAPS',
                                     batchSize=1,
                                     idx_fold=args.idx_fold,
                                     split='test',
                                     returnSubjectID=True)

    AIBL_testData = MRIDataGenerator_Simple(READ_DIR + 'AIBL_CAPS',
                                            'aibl_info.csv', batchSize=1, returnSubjectID=True)

    MIRIAD_testData = MRIDataGenerator_Simple(READ_DIR + 'MIRIAD_CAPS',
                                              'miriad_test_info.csv', batchSize=1, returnSubjectID=True)

    OASIS3_testData = MRIDataGenerator_Simple(READ_DIR + 'OASIS3_CAPS',
                                              'oasis3_test_info_2.csv', batchSize=1, returnSubjectID=True)

    total_step_test_ADNI = int(math.ceil(len(ADNI_testData) / args.batch_size))
    total_step_test_AIBL = int(math.ceil(len(AIBL_testData) / args.batch_size))
    total_step_test_MIRIAD = int(math.ceil(len(MIRIAD_testData) / args.batch_size))
    total_step_test_OASIS3 = int(math.ceil(len(OASIS3_testData) / args.batch_size))
    print('Activation Maximization Starts ...')

    activation_maximizer = ActivationMaximizer(model, args.visualize_feature_idx)

    activation_maximizer.visualize_activation(ADNI_testData, total_step_test_ADNI)
    activation_maximizer.visualize_activation(AIBL_testData, total_step_test_AIBL)
    activation_maximizer.visualize_activation(MIRIAD_testData, total_step_test_MIRIAD)
    activation_maximizer.visualize_activation(OASIS3_testData, total_step_test_OASIS3)


def main(args):
    train(args)


def get_model_info(args):
    num_classes = 2
    model = MRIImaging3DConvModel(nClass=num_classes, args=args)
    model.load_weights(
        WEIGHTS_DIR + args.weights_folder + '/weights' + getSaveName(args) + '_epoch_' + str(args.continueEpoch))

    print(model.summary())

def init_gpu(gpu_index, force=False):
    if isinstance(gpu_index, list):
        gpu_num = ','.join([str(i) for i in gpu_index])
    else:
        gpu_num = str(gpu_index)
    if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] and not force:
        print('GPU already initiated')
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=50, help='How many epochs to run in total?')
    parser.add_argument('-b', '--batch_size', type=int, default=8,
                        help='Batch size during training per GPU')
    parser.add_argument('-a', '--action', type=int, default=0, help='action to take')
    parser.add_argument('-s', '--seed', type=int, default=1, help='random seed')
    parser.add_argument('-i', '--idx_fold', type=int, default=0, help='which partition of data to use')
    parser.add_argument('-u', '--augmented', type=int, default=0, help='whether use augmentation or not')
    parser.add_argument('-g', '--augmented_fancy', type=int, default=0,
                        help='whether use the fancy, Alzheimer specific augmentation or not')
    parser.add_argument('-m', '--mci', type=int, default=0, help='whether use MCI data or not')
    parser.add_argument('-l', '--mci_balanced', type=int, default=0,
                        help='when using MCI, whether including it as a balanced data')
    parser.add_argument('-c', '--continueEpoch', type=int, default=0, help='continue from current epoch')
    parser.add_argument('-p', '--pgd', type=float, default=0, help='whether we use pgd (actually fast fgsm)')
    parser.add_argument('-n', '--minmax', type=int, default=0, help='whether we use min max pooling')
    parser.add_argument('-f', '--weights_folder', type=str, default='.', help='the folder weights are saved')
    parser.add_argument('-v', '--saliency_save_dir', type=str, default=READ_DIR + 'saliency_maps',
                        help='the folder to save saliency maps')
    parser.add_argument('-j', '--extract_features', type=int, default=0,
                        help='whether the model should exclude the last FC layer')
    parser.add_argument('-k', '--attack_visualization_save_dir', type=str, default=READ_DIR + 'attack_visualization',
                        help='the folder to save adversarial attack visualization')
    parser.add_argument('-w', '--activation_maximization_dir', type=str, default=READ_DIR + 'activation_maximizations',
                        help='the folder to save visualized activation maximizations')
    parser.add_argument('-z', '--visualize_feature_idx', type=int, default=0,
                        help='feature visualizing activation maximization, only 0 or 1 for if we consider the model prediction')
    parser.add_argument('-d', '--dropBlock', type=int, default=0,
                        help='whether we drop half of the information of the images')
    parser.add_argument('-o', '--gradientGuidedDropBlock', type=int, default=0,
                        help='whether we perform gradient guided dropBlock')
    parser.add_argument('-q', '--dropBlock3D', type=int, default=0, help='whether we perform 3D dropblock on conv layers')
    parser.add_argument('-r', '--worst_sample', type=int, default=0, help='whether we use min max pooling')
    parser.add_argument('-y', '--consistency', type=float, default=0, help='whether we use min max pooling')
    parser.add_argument('-t', '--gpu', type=str, default=0,
                        help='specify maximum GPU ID we want to distribute the training to')

    parser.add_argument('-ft', '--mci_finetune', type=int, default=0, help='whether we finetune the trained model with only MCI subjects')
    args = parser.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # pretty print args
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))

    if args.action == 0:
        main(args)
    elif args.action == 1:
        evaluate_crossDataSet(args)
    elif args.action == 2:
        evaluate_crossDataSet_at_individual(args)
    elif args.action == 3:
        embedding_extractor(args)
    elif args.action == 4:
        saliency(args)
    elif args.action == 5:
        attack_visualization(args)
    elif args.action == 6:
        activation_maximization_visualize(args)
    elif args.action == 7:
        get_model_info(args)
