__author__ = 'Haohan Wang'

import torch
import numpy as np

from os.path import join

from matplotlib import pyplot as plt

from scipy.stats import ttest_ind
import time

img_dir = '/media/haohanwang/Info/ADNI_CAPS'

def loadOneSample_subject_session(subjectID, sessID):

    image_path = join(img_dir, 'subjects', subjectID, sessID, 'deeplearning_prepare_data',
                      'image_based',
                      't1_linear',
                      subjectID + '_' + sessID + '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.pt')

    data = torch.load(image_path).cpu().numpy()[0]
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def loadOneSample_path(image_path):
    data = torch.load(image_path).cpu().numpy()[0]
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def loadSubjectIDs(idx_fold = 0):
    img_dir = '/media/haohanwang/Info/ADNI_CAPS'

    csv_path = join(img_dir, f'split.pretrained.{idx_fold}.csv')
    text = [line.strip() for line in open(csv_path)]

    filePaths = []
    labels = []

    for line in text[1:]:
        items = line.split(',')
        if items[-1] == 'train':
            image_path = join(img_dir, 'subjects', items[0], items[1], 'deeplearning_prepare_data',
                              'image_based',
                              't1_linear',
                              items[0] + '_' + items[
                                  1] + '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.pt')
            if items[-2] == 'AD':
                filePaths.append(image_path)
                labels.append(1)
            elif items[-2] == 'CN':
                filePaths.append(image_path)
                labels.append(0)

    return filePaths, labels

def _test_pixel_statistics(pixels, labels):
    pixels = np.array(pixels)
    labels = np.array(labels)

    a = pixels[labels==0]
    b = pixels[labels==1]

    print (a)
    print (b)
    print (np.mean(a))
    print (np.mean(b))

    s, p = ttest_ind(a, b)
    return p

def testPixelStatistics(pixelIndex, idx_fold):
    filePaths, labels = loadSubjectIDs(idx_fold)

    print (len(filePaths))

    pixels = []
    for fp in filePaths:
        data = loadOneSample_path(fp)
        pixels.append(data[pixelIndex[0], pixelIndex[1], pixelIndex[2]])

    return _test_pixel_statistics(pixels, labels)

def testStatistics(idx_fold):
    # shape 169, 208, 179
    start = time.time()
    for i in range(169):
        for j in range(208):
            for k in range(179):
                p = testPixelStatistics((i, j, k), idx_fold)
                print (i, j, k, p)
                print ('with %d seconds passed' % (time.time()-start))


if __name__ == '__main__':

    testStatistics(0)

    # print (ttest_ind([1,2,3,4,5], [8, 9, 2, 2, 1])[1])

    # data = loadOneSample_subject_session('sub-ADNI021S0332', 'ses-M00')
    #
    # print (np.mean(data))
    # print (np.max(data))
    # print (np.min(data))
    # print (np.std(data))
    #
    # print (data.shape)
    #
    # slice = data[:,:,80].reshape((169, 208))
    #
    # plt.imshow(slice)
    # plt.show()