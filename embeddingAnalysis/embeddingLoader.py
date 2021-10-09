__author__ = 'Haohan Wang'

import sys

import numpy as np

def getFileName(folderName):
    if folderName == 'aug_fancy':
        fileName = '/result_aug_fancy_fold_0_seed_1_epoch_120_'
    elif folderName == 'PGD':
        fileName = '/result_aug_pgd_0.001_fold_0_seed_1_epoch_45_'
    elif folderName == 'vanilla':
        fileName = '/result_aug_fold_0_seed_1_epoch_50_'
    else:
        sys.exit()

    return fileName

def getLabels(dataBaseName):

    agIdx = 2
    gdIdx = 3

    if dataBaseName == 'ADNI':
        diagIndex = -2
        csv_path = '/media/haohanwang/Storage/AlzheimerImagingData/ADNI_CAPS/split.pretrained.0.csv'
    elif dataBaseName == 'AIBL':
        diagIndex = -1
        csv_path = '/media/haohanwang/Storage/AlzheimerImagingData/AIBL_CAPS/aibl_info.csv'
    elif dataBaseName == 'MIRIAD':
        diagIndex = -1
        csv_path = '/media/haohanwang/Storage/AlzheimerImagingData/MIRIAD_CAPS/miriad_test_info.csv'
    elif dataBaseName == 'OASIS3':
        diagIndex = -2
        csv_path = '/media/haohanwang/Storage/AlzheimerImagingData/OASIS3_CAPS/oasis3_test_info_2.csv'
    else:
        sys.exit()


    text = [line.strip() for line in open(csv_path)]

    labels = {}
    ages = {}
    genders = {}

    for line in text[1:]:
        items = line.split(',')

        did = items[0] + '#' + items[1]

        if items[diagIndex] == 'AD':
            labels[did] = 1
        elif items[diagIndex] == 'CN':
            labels[did] = 0

        if dataBaseName.find('OASIS3') == -1:

            ages[did] = float(agIdx)

            if items[gdIdx].startswith('F'):
                genders[did] = 0
            else:
                genders[did] = 1

    if dataBaseName.find('OASIS3') == -1:
        return labels, ages, genders
    else:
        return labels, None, None


def embeddingFileLoader(folderName, dataBaseName, split='test'):

    fileName = getFileName(folderName)

    if split == 'test':
        ebd = np.load('../mainScripts/embeddingResult/' + folderName + fileName + dataBaseName + '.npy')
        ids = ['#'.join(line.strip().split(',')) for line in
               open('../mainScripts/embeddingResult/' + folderName + fileName + dataBaseName + '.csv')]
    else:
        ebd = np.load(
            '../mainScripts/embeddingResult/' + folderName + fileName + dataBaseName + '_' + split + '.npy')
        ids = ['#'.join(line.strip().split(',')) for line in
               open('../mainScripts/embeddingResult/' + folderName + fileName + dataBaseName + '_' + split + '.csv')]

    labelDict, ageDict, genderDict = getLabels(dataBaseName)

    label = np.zeros(ebd.shape[0])
    age = np.zeros(ebd.shape[0])
    gender = np.zeros(ebd.shape[0])

    for i in range(len(ids)):
        label[i] = labelDict[ids[i]]

    if dataBaseName.find('OASIS3') == -1:
        for i in range(len(ids)):
            age[i] = ageDict[ids[i]]

        for i in range(len(ids)):
            gender[i] = genderDict[ids[i]]

    return ebd, label, age, gender

if __name__ == '__main__':
    embeddingFileLoader('aug_fancy', 'AIBL')
