__author__ = 'Haohan Wang'


import torch
import numpy as np
from os.path import join

from matplotlib import pyplot as plt
from scipy.stats import ttest_ind
import time
import cv2

img_dir = '/media/haohanwang/Storage/AlzheimerImagingData/ADNI_CAPS'

def loadOneSample_subject_session(subjectID, sessID):

    image_path = join(img_dir, 'subjects', subjectID, sessID, 'deeplearning_prepare_data',
                      'image_based',
                      't1_linear',
                      subjectID + '_' + sessID + '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.pt')

    data = torch.load(image_path).cpu().numpy()[0]
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def imgwrite(path, img):
    img = (img - np.min(img))/(np.max(img) - np.min(img)) * 255
    cv2.imwrite(path, img)


if __name__ == '__main__':
    from scipy.ndimage.morphology import grey_erosion, grey_dilation


    data1 = loadOneSample_subject_session('sub-ADNI021S0332', 'ses-M00') # this is an AD case

    data1_1 = grey_dilation(data1, size=(4, 4, 4))
    data1_2 = grey_erosion(data1, size=(4, 4, 4))

    # print (np.mean(data1))
    # print (np.max(data1))
    # print (np.min(data1))
    # print (np.std(data1))

    # print (data1.shape)

    slice1 = data1[:,:,80].reshape((169, 208))
    #
    # print (np.mean(slice1))
    # print (np.max(slice1))
    # print (np.min(slice1))
    # print (np.std(slice1))
    #
    # print (slice1.shape)

    slice1_1 = data1_1[:,:,80].reshape((169, 208))
    slice1_2 = data1_2[:,:,80].reshape((169, 208))

    plt.imshow(slice1)
    plt.show()
    plt.imshow(slice1_1)
    plt.show()
    plt.imshow(slice1_2)
    plt.show()

    imgwrite('tmpImages/slice1.png', slice1)
    imgwrite('tmpImages/slice1_1.png', slice1_1)
    imgwrite('tmpImages/slice1_2.png', slice1_2)


    data2 = loadOneSample_subject_session('sub-ADNI029S0824', 'ses-M84')

    data2_1 = grey_dilation(data2, size=(5, 5, 5))
    data2_2 = grey_erosion(data2, size=(5, 5, 5))

    # print (np.mean(data2))
    # print (np.max(data2))
    # print (np.min(data2))
    # print (np.std(data2))

    # print (data2.shape)

    slice2 = data2[:,:,80].reshape((169, 208))

    slice2_1 = data2_1[:,:,80].reshape((169, 208))
    slice2_2 = data2_2[:,:,80].reshape((169, 208))

    # print (slice2.shape)
    #
    # print (np.mean(slice2))
    # print (np.max(slice2))
    # print (np.min(slice2))
    # print (np.std(slice2))

    plt.imshow(slice2)
    plt.show()
    plt.imshow(slice2_1)
    plt.show()
    plt.imshow(slice2_2)
    plt.show()

    imgwrite('tmpImages/slice2.png', slice2)
    imgwrite('tmpImages/slice2_1.png', slice2_1)
    imgwrite('tmpImages/slice2_2.png', slice2_2)
    # slice_diff = np.abs(slice1 - slice2)
    #
    # print (np.mean(slice_diff))
    # print (np.max(slice_diff))
    # print (np.min(slice_diff))
    # print (np.std(slice_diff))
    #
    # plt.imshow(slice_diff)
    # plt.show()
