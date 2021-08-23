__author__ = 'Haohan Wang'

import torch
import numpy as np

from os.path import join

from matplotlib import pyplot as plt

img_dir = '/media/haohanwang/Info/ADNI_CAPS'

def loadOneSample(subjectID, sessID):

    image_path = join(img_dir, 'subjects', subjectID, sessID, 'deeplearning_prepare_data',
                      'image_based',
                      't1_linear',
                      subjectID + '_' + sessID + '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.pt')

    data = torch.load(image_path).cpu().numpy()
    return (data - np.min(data)) / (np.max(data) - np.min(data))

if __name__ == '__main__':
    data = loadOneSample('sub-ADNI021S0332', 'ses-M00')

    print (np.mean(data))
    print (np.max(data))
    print (np.min(data))
    print (np.std(data))

    slice = data[:,:,80].reshape((169, 179))

    plt.imshow(slice)
    plt.show()