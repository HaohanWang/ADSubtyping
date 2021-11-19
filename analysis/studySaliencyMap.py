__author__ = 'Haohan Wang'

import torch
import numpy as np
from os.path import join

from matplotlib import pyplot as plt
from scipy.stats import ttest_ind
import time
import cv2

from scipy.ndimage.morphology import grey_erosion, grey_dilation

def loadSmapData(folderPath, subjectID, sessionID):
    smap = np.load(folderPath + '/' + subjectID + '_' + sessionID + '_smap.npy')

    return smap.reshape([169, 208, 179])

img_dir = '/media/haohanwang/Storage/AlzheimerImagingData/ADNI_CAPS'

def loadOneSample_subject_session(subjectID, sessID):

    image_path = join(img_dir, 'subjects', subjectID, sessID, 'deeplearning_prepare_data',
                      'image_based',
                      't1_linear',
                      subjectID + '_' + sessID + '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.pt')

    data = torch.load(image_path).cpu().numpy()[0]
    return (data - np.min(data)) / (np.max(data) - np.min(data))

if __name__ == '__main__':

    for folderName in ['vanilla', 'pgd', 'drop_block']:

        folderPath = '../mainScripts/saliency_maps/' + folderName + '/adni_test'
        subjectID = 'sub-ADNI005S0814'
        sessionID = 'ses-M00'

        smap = loadSmapData(folderPath, subjectID, sessionID)

        smap = np.abs(smap)

        smap = grey_dilation(smap, size=(3, 3, 3))

        smap = (smap-np.min(smap))/(np.max(smap) - np.min(smap))

        smap_slice1_1 = smap[:,:,80].reshape((169, 208))

        image = loadOneSample_subject_session(subjectID, sessionID)

        image_slice1_1 = image[:,:,80].reshape((169, 208))

        combined = image_slice1_1 + smap_slice1_1

        pltImages = np.zeros([169, 208, 3])
        pltImages[:,:,0] = image_slice1_1
        pltImages[:,:,1] = image_slice1_1
        pltImages[:,:,2] = image_slice1_1

        pltImages[:,:,1] += smap_slice1_1


        # plt.imshow(image_slice1_1)
        # plt.show()
        # plt.imshow(smap_slice1_1)
        # plt.show()

        plt.imshow(pltImages)
        plt.show()