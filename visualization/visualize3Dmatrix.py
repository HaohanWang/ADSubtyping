__author__ = 'Haohan Wang'

import utility.loadResults as utils

import ipyvolume as ipv

import numpy as np

from os.path import join

def visualizeData(subj, sess, filePath):
    data = utils.loadNPYFile(subj, sess, filePath)

    # mini = np.min(data)
    # maxi = np.max(data)
    #
    # data = (data - mini)/(maxi - mini)

    print (np.std(data))
    print (np.mean(data))
    print ('-------------')

    out = sess + '.html'
    outpath = join(filePath, out)


    ipv.pylab.clear()
    ipv.volshow(data, level=[0.2,0.2,0.2])
    ipv.pylab.save(outpath)



if __name__ == '__main__':
    subj = 'sub-ADNI002S0619'
    sess = 'ses-M00'
    filePath = '/home/ec2-user/alzstudy/saliency_maps/adni_train/vanilla/'
    visualizeData(subj, sess, filePath)

    filePath2 = '/media/haohanwang/Elements/saliency_map2/'
    visualizeData(subj, sess, filePath2)

