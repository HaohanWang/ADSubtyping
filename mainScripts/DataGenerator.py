__author__ = 'Haohan Wang'

import cv2
from tensorflow import keras
import tensorflow as tf

import numpy as np

import torch
from os.path import join

from dataAugmentation import MRIDataAugmentation
import mainScripts.MCIFinetuneDataCleaning

class MRIDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, img_dir,
                 split,
                 transform=None,
                 idx_fold=0,
                 num_fold=5,
                 batchSize=32,
                 dim=(169, 208, 179),
                 n_channels=1,
                 n_classes=2,
                 augmented=True,
                 augmented_fancy=False,
                 MCI_included=False,
                 MCI_included_as_soft_label=False,
                 returnSubjectID=False,
                 dropBlock = False,
                 dropBlockIterationStart = 0,
                 gradientGuidedDropBlock=False,
                 mci_finetune=False
                 ):
        # 'Initialization'

        self.img_dir = img_dir
        self.split = split
        self.transform = transform
        self.idx_fold = idx_fold
        self.num_fold = num_fold
        self.batch_size = batchSize
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.augmented = augmented
        self.augmented_fancy = augmented_fancy
        self.MCI_included = MCI_included
        self.MCI_included_as_soft_label = MCI_included_as_soft_label
        self.returnSubjectID = returnSubjectID
        self.dropBlock = dropBlock

        self.dropBlock_iterationCount = dropBlockIterationStart
        self.gradientGuidedDropBlock = gradientGuidedDropBlock
        self.mci_finetune = mci_finetune

        self.parse_csv_file()
        self._get_batch_split()
        self.on_epoch_end()

        self.dataAugmentation = MRIDataAugmentation(self.dim, 0.5)

    def __len__(self):
        self.on_epoch_end()
        return self.totalLength

    def __getitem__(self, idx):
        if self.split == 'train':
            if not self.returnSubjectID:  # training tricks such as balance the two labels and augmentation will not happen once subject ID is required
                images, labels = self._load_batch_image_train(idx)
                if self.augmented:
                    if self.augmented_fancy:
                        images = self.dataAugmentation.augmentData_batch_withLabel(images, labels)
                    if self.dropBlock:
                        images = self.dataAugmentation.augmentData_batch_erasing(images, self.dropBlock_iterationCount)
                        self.dropBlock_iterationCount += 1
                    images = self.dataAugmentation.augmentData_batch(images)

                if self.transform:
                    images = self.transform(images)
                return images, labels
            else:
                images, labels, subjectLists, sessionLists = self._load_batch_image_test(
                    idx)  # if subject ID is required, these can be saved as the test list
                if self.transform:
                    images = self.transform(images)
                return images, labels, subjectLists, sessionLists
        else:
            if self.returnSubjectID:
                images, labels, subjectLists, sessionLists = self._load_batch_image_test(idx)
                if self.transform:
                    images = self.transform(images)
                return images, labels, subjectLists, sessionLists
            else:
                images, labels = self._load_batch_image_test(idx)
                if self.transform:
                    images = self.transform(images)
                return images, labels

    def parse_csv_file(self):
        csv_path = join(self.img_dir, f'split.pretrained.{self.idx_fold}.csv')
        text = [line.strip() for line in open(csv_path)]
        self.filePaths_AD = []
        label_AD = []
        self.filePaths_CN = []
        label_CN = []
        self.filePaths_MCI = []
        label_MCI = []
        subject_AD = []
        subject_CN = []
        subject_MCI = []
        session_AD = []
        session_CN = []
        session_MCI = []

        self.mci_labels = []

        if self.mci_finetune:
            assert(not self.MCI_included_as_soft_label, "We should not use MCI as soft label when finetuning with MCI subjects")
            self.mci_subjects_to_new_label = MCIFinetuneDataCleaning.find_mci_subjects(self.img_dir, self.idx_fold)

        for line in text[1:]:
            items = line.split(',')
            if self.mci_finetune and items[0] not in self.mci_subjects_to_new_label:
                # when we finetune with MCI subjects, we exclude the subject if they do not show MCI in any of
                # their sessions
                continue

            if self.mci_finetune:
                self.mci_labels.append(self.mci_subjects_to_new_label[items[0]])


            if items[-1] == self.split:
                image_path = join(self.img_dir, 'subjects', items[0], items[1], 'deeplearning_prepare_data',
                                  'image_based',
                                  't1_linear',
                                  items[0] + '_' + items[
                                      1] + '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.pt')
                if items[-2] == 'AD':
                    self.filePaths_AD.append(image_path)
                    label_AD.append(1)
                    subject_AD.append(items[0])
                    session_AD.append(items[1])
                elif items[-2] == 'CN':
                    self.filePaths_CN.append(image_path)
                    label_CN.append(0)
                    subject_CN.append(items[0])
                    session_CN.append(items[1])
                elif items[-2] == 'MCI':
                    self.filePaths_MCI.append(image_path)
                    label_MCI.append(1)
                    subject_MCI.append(items[0])
                    session_MCI.append(items[1])

        if self.returnSubjectID or (not self.split == 'train'):
            if self.MCI_included:
                self.filePaths_test = self.filePaths_AD + self.filePaths_CN + self.filePaths_MCI
                self.labels_test = label_AD + label_CN + label_MCI
                self.subjects_test = subject_AD + subject_CN + subject_MCI
                self.sessions_test = session_AD + session_CN + session_MCI
            else:
                self.filePaths_test = self.filePaths_AD + self.filePaths_CN
                self.labels_test = label_AD + label_CN
                self.subjects_test = subject_AD + subject_CN
                self.sessions_test = session_AD + session_CN

        self.totalLength = len(self.filePaths_AD) + len(self.filePaths_CN) + len(self.filePaths_MCI) * self.MCI_included

        if self.mci_finetune:
            print("sanity check: verifying totalLength is the same as the size of mci labels")
            assert(self.totalLength == len(self.mci_labels))

    def on_epoch_end(self):
        if self.split == 'train':
            np.random.shuffle(self.filePaths_CN)
            np.random.shuffle(self.filePaths_AD)
            np.random.shuffle(self.filePaths_MCI)

    def _load_one_image(self, image_path):
        d = torch.load(image_path).cpu().numpy().astype(np.float32)
        d = (d - np.min(d)) / (np.max(d) - np.min(d))
        return d

    def _get_batch_split(self):
        if self.MCI_included:
            if self.MCI_included_as_soft_label:
                self.batch_size_CN = int(self.batch_size * 0.375)
                self.batch_size_AD = int(self.batch_size * 0.375)
                self.batch_size_MCI = int(self.batch_size * 0.25)
            else:
                self.batch_size_CN = int(self.batch_size * 0.5)
                self.batch_size_AD = int(self.batch_size * 0.25)
                self.batch_size_MCI = int(self.batch_size * 0.25)
        else:
            self.batch_size_CN = int(self.batch_size * 0.5)
            self.batch_size_AD = int(self.batch_size * 0.5)

    def _rotate_idx(self, l, m):
        for i in range(len(l)):
            while l[i] >= m:
                l[i] = l[i] - m
        return l

    def _load_batch_image_train(self, idx):
        idxlist_CN = [*range(idx * self.batch_size_CN, (idx + 1) * self.batch_size_CN)]
        idxlist_AD = [*range(idx * self.batch_size_AD, (idx + 1) * self.batch_size_AD)]

        idxlist_CN = self._rotate_idx(idxlist_CN, len(self.filePaths_CN))
        idxlist_AD = self._rotate_idx(idxlist_AD, len(self.filePaths_AD))

        images = np.zeros((self.batch_size, *self.dim, self.n_channels))
        labels = np.zeros((self.batch_size, self.n_classes))

        for i in range(self.batch_size_CN):
            images[i, :, :, :, 0] = self._load_one_image(self.filePaths_CN[idxlist_CN[i]])

            if self.mci_finetune:
                labels[i, self.mci_labels[i]] = 1
            else:
                labels[i, 0] = 1

        for i in range(self.batch_size_AD):
            images[i + self.batch_size_CN, :, :, :, 0] = self._load_one_image(self.filePaths_AD[idxlist_AD[i]])

            if self.mci_finetune:
                labels[i + self.batch_size_CN, self.mci_labels[i + self.batch_size_CN]] = 1
            else:
                labels[i + self.batch_size_CN, 1] = 1

        if self.MCI_included:
            idxlist_MCI = [*range(idx * self.batch_size_MCI, (idx + 1) * self.batch_size_MCI)]
            idxlist_MCI = self._rotate_idx(idxlist_MCI, len(self.filePaths_MCI))

            for i in range(self.batch_size_MCI):
                images[i + self.batch_size_CN + self.batch_size_AD, :, :, :, 0] = self._load_one_image(
                    self.filePaths_MCI[idxlist_MCI[i]])

                if self.MCI_included_as_soft_label:
                    t = np.random.random()
                    labels[i + self.batch_size_CN + self.batch_size_AD, 0] = t
                    labels[i + self.batch_size_CN + self.batch_size_AD, 1] = 1 - t
                else:

                    if self.mci_finetune:
                        labels[i + self.batch_size_CN + self.batch_size_AD, self.mci_labels[i + self.batch_size_CN + self.batch_size_AD]] = 1
                    else:
                        labels[i + self.batch_size_CN + self.batch_size_AD, 1] = 1

        return images, labels

    def _load_batch_image_test(self, idx):
        idxlist = [*range(idx * self.batch_size, (idx + 1) * self.batch_size)]
        idxlist = self._rotate_idx(idxlist, len(self.filePaths_test))

        images = np.zeros((self.batch_size, *self.dim, self.n_channels)).astype(np.float32)
        labels = np.zeros((self.batch_size))

        subjectList = []
        sessionList = []
        for i in range(self.batch_size):
            images[i, :, :, :, 0] = self._load_one_image(self.filePaths_test[idxlist[i]])

            if self.mci_finetune:
                sub = self.subjects_test[idxlist[i]]
                labels[i] = self.mci_subjects_to_new_label[sub]
            else:
                labels[i] = self.labels_test[idxlist[i]]


            subjectList.append(self.subjects_test[idxlist[i]])
            sessionList.append(self.sessions_test[idxlist[i]])

        if self.returnSubjectID:
            return images, tf.one_hot(labels, self.n_classes), subjectList, sessionList
        else:
            return images, tf.one_hot(labels, self.n_classes)


class MRIDataGenerator_Simple(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, img_dir,
                 csv_path,
                 batchSize=32,
                 dim=(169, 208, 179),
                 n_channels=1,
                 n_classes=2,
                 returnSubjectID=False
                 ):
        # 'Initialization'

        self.img_dir = img_dir
        self.csv_path = csv_path
        self.batch_size = batchSize
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.returnSubjectID = returnSubjectID

        self.diagIndex = -1
        if self.img_dir.find('OASIS') != -1:
            self.diagIndex = -2

        self.parse_csv_file()

        self.dataAugmentation = MRIDataAugmentation(self.dim, 0.5)

    def __len__(self):
        self.on_epoch_end()
        return self.totalLength

    def __getitem__(self, idx):
        if self.returnSubjectID:
            images, labels, subjects, sessions = self._load_batch_image_test(idx)
            return images, labels, subjects, sessions
        else:
            images, labels = self._load_batch_image_test(idx)
            return images, labels

    def parse_csv_file(self):
        csv_path = join(self.img_dir, self.csv_path)
        text = [line.strip() for line in open(csv_path)]
        self.filePaths_test = []
        self.labels_test = []
        self.subjects_test = []
        self.sessions_test = []

        for line in text[1:]:
            items = line.split(',')
            image_path = join(self.img_dir, 'subjects', items[0], items[1], 't1_linear',
                              items[0] + '_' + items[1] + '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.pt')

            if items[self.diagIndex] == 'AD':
                self.labels_test.append(1)
                self.filePaths_test.append(image_path)
                self.subjects_test.append(items[0])
                self.sessions_test.append(items[1])
            elif items[self.diagIndex] == 'CN':
                self.labels_test.append(0)
                self.filePaths_test.append(image_path)
                self.subjects_test.append(items[0])
                self.sessions_test.append(items[1])

        self.totalLength = len(self.filePaths_test)

    def on_epoch_end(self):
        pass

    def _rotate_idx(self, l, m):
        for i in range(len(l)):
            while l[i] >= m:
                l[i] = l[i] - m
        return l

    def _load_one_image(self, image_path):
        d = torch.load(image_path).cpu().numpy().astype(np.float32)
        d = (d - np.min(d)) / (np.max(d) - np.min(d))
        return d

    def _load_batch_image_test(self, idx):
        idxlist = [*range(idx * self.batch_size, (idx + 1) * self.batch_size)]
        idxlist = self._rotate_idx(idxlist, len(self.filePaths_test))

        images = np.zeros((self.batch_size, *self.dim, self.n_channels)).astype(np.float32)
        labels = np.zeros((self.batch_size))

        subjectList = []
        sessionList = []
        for i in range(self.batch_size):
            images[i, :, :, :, 0] = self._load_one_image(self.filePaths_test[idxlist[i]])
            labels[i] = self.labels_test[idxlist[i]]
            subjectList.append(self.subjects_test[idxlist[i]])
            sessionList.append(self.sessions_test[idxlist[i]])

        if self.returnSubjectID:
            return images, tf.one_hot(labels, self.n_classes), subjectList, sessionList
        else:
            return images, tf.one_hot(labels, self.n_classes)
