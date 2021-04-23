__author__ = 'Haohan Wang'

from utility.loadDataStatistics import loadInformation
from utility.loadResults import loadNPYFile

from matplotlib import pyplot as plt
import numpy as np

def load_Plot():
    labels_train, predictions_train = loadInformation(trainFlag=True)

    print np.mean(labels_train)

    plt.hist(predictions_train)
    plt.show()

def checkDifferences():
    infoFile = [line.strip() for line in open('../outputs/sortedInfo.csv')]

    info = []
    for line in infoFile:
        info.append(line.split(','))

    for (subj, sess, _, _, _) in info:
        loadNPYFile(subj, sess, False)


if __name__ == '__main__':
    checkDifferences()