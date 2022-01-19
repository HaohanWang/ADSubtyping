__author__ = 'Haohan Wang'

from utility.loadResults import loadNPYFile

import ipyvolume as ipv

import numpy as np

from os.path import join

def visualizeData(subj, sess, filePath):
    data = loadNPYFile(subj, sess, filePath)

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


def visualizeResults(fileName):
    data = np.load('../analysis/visualizations/' + fileName + '.npy')

    print (np.std(data))
    print (np.mean(data))
    print ('-------------')

    outpath = '../analysis/visualizations/' + fileName + '.html'

    ipv.pylab.clear()
    ipv.volshow(data, level=[0.2,0.2,0.2])
    ipv.pylab.save(outpath)


if __name__ == '__main__':
    fileNames = ['ses-M00', 'ses-M00_', 'ses-M06', 'ses-M06_']

    for fn in fileNames:
        visualizeResults(fn)
