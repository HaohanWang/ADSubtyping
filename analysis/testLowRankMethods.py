__author__ = 'Haohan Wang'

from utility.loadResults import loadNPYFile

import ipyvolume as ipv

from os.path import join

def visualizeDataAndLowRankData(subj, sess, trainFlag):
    data = loadNPYFile(subj, sess, trainFlag)

    print (data.shape)

    out = sess + '.html'
    outpath = join('visualizedResults/', out)

    ipv.pylab.clear()
    ipv.volshow(data, level=[1,1,1])
    ipv.pylab.save(outpath)


if __name__ == '__main__':
    subj = 'sub-ADNI002S1261'
    sess = 'ses-M00'
    trainFlag = False
    visualizeDataAndLowRankData(subj, sess, trainFlag)

