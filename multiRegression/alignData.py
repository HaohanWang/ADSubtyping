__author__ = 'Haohan Wang'

import numpy as np
from scipy.ndimage import zoom

from utility.loadResults import loadNPYFile

filePath = '/media/haohanwang/Elements/attack_visualization/'

def generateIDs():
    text1 = [line.strip() for line in open('../data/split.stratified.0.csv')]
    id1 = {}
    for line in text1:
        s = line.split(',')[0][8:]
        id1[s] = line
    text2 = [line.strip() for line in open('../data/samples.txt')]
    id2 = {}
    for line in text2:
        s = line.replace('_','')
        id2[s] = line

    print len(id1)
    print len(id2)

    commonIDs = {}
    for m in id2:
        if m in id1:
            commonIDs[m] = 1

    f1 = open('../data/multiRegression/mriID.txt', 'w')
    for k in id1:
        if k in commonIDs:
            f1.writelines(id1[k] + '\n')
    f1.close()

    f2 = open('../data/multiRegression/snpID.txt', 'w')
    for k in id2:
        if k in commonIDs:
            f2.writelines(id2[k] + '\n')
    f2.close()

def downSamplingMRIResult():
    text = [line.strip() for line in open('../data/multiRegression/mriID.txt')]

    Data = []

    for line in text:
        items = line.split(',')

        subj = items[0]
        sess = items[1]

        data = loadNPYFile(subj, sess)

        data = zoom(data, [0.1, 0.1, 0.1])
        data[data<1e-5] = 0
        print data.shape
        # assert data.shape == [17, 21, 18]
        data = data.reshape([17*21*18])
        Data.append(data)

    Data = np.array(Data)

    print Data.shape

    np.save('../data/multiRegression/mriData.npy', Data)

def organizeSNPdata():
    snp_data = np.load('../data/snps.npy')

    id1 = [line.strip() for line in open('../data/samples.txt')]
    id2 = [line.strip() for line in open('../data/multiRegression/snpID.txt')]

    idx = []

    for i in range(len(id1)):
        if id1[i] in id2:
            idx.append(i)

    idx = np.array(idx)

    data = snp_data[idx,:]
    print data.shape

    np.save('../data/multiRegression/snps.npy', data)


if __name__ == '__main__':
    pass
    # step 1
    # generateIDs()
    # step 2
    # downSamplingMRIResult()
    # step 3
    organizeSNPdata()