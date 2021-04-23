__author__ = 'Haohan Wang'

from os import path
import numpy as np

# filePath = '/media/haohanwang/Elements/saliency_map/'


def loadNPYFile(subject, sess, filePath):
    if path.exists(filePath + 'train/' + subject + '/' + sess + '.npy'):
        data = np.load(filePath + 'train/' + subject + '/' + sess + '.npy')
    else:
        data = np.load(filePath + 'test/' + subject + '/' + sess + '.npy')

    return data


if __name__ == '__main__':
    tmp = loadNPYFile('sub-ADNI002S1261', 'ses-M00', '/media/haohanwang/Elements/saliency_map/')

    print (tmp.shape)