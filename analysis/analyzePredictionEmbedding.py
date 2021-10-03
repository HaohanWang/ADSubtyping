__author__ = 'Haohan Wang'

import sys
import numpy as np

from matplotlib import pyplot as plt


def getFileName(folderName, dataBase):
    if folderName == 'aug_fancy':
        fileName = folderName + '/result_aug_fancy_fold_0_seed_1_epoch_120_' + dataBase
    elif folderName == 'PGD':
        fileName = folderName + '/result_aug_pgd_0.001_fold_0_seed_1_epoch_45_' + dataBase
    elif folderName == 'vanilla':
        fileName = folderName + '/result_aug_fold_0_seed_1_epoch_50_' + dataBase
    else:
        sys.exit()

    return fileName


def getFileEmbeddings(fileName):
    idx = ['#'.join(line.strip().split(',')) for line in open('../mainScripts/embeddingResult/' + fileName + '.csv')]
    embeddings = np.load('../mainScripts/embeddingResult/' + fileName + '.npy')

    return idx, embeddings


def getPredictionResults(fileName):
    text = [line.strip() for line in open('../mainScripts/predictionResults/' + fileName + '.csv')]

    labels = {}

    if fileName.find('ADNI') == -1 and fileName.find('OASIS3') == -1:
        oidx = -2
    else:
        oidx = -3

    for line in text:
        items = line.split(',')

        if items[oidx] == 'CN':
            ori = 0
        else:
            ori = 1

        if items[-1] == 'CN':
            pred = 0
        else:
            pred = 1

        labels[items[0] + '#' + items[1]] = (ori, pred)

    return labels


def getEmebeddingAndLabel(folderName, dataBase):
    fileName = getFileName(folderName, dataBase)

    idx, embeddings = getFileEmbeddings(fileName)
    labels = getPredictionResults(fileName)

    lo = []
    lp = []
    for id in idx:
        lo.append(labels[id][0])
        lp.append(labels[id][1])

    lo = np.array(lo)
    lp = np.array(lp)

    return embeddings, lo, lp


def analyzeEmbeddings(embeddings, labels):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    tsne = PCA(n_components=2)

    emb = tsne.fit_transform(embeddings)

    emb_1 = emb[labels == 0]
    emb_2 = emb[labels == 1]

    plt.scatter(emb_1[:, 0], emb_1[:, 1], marker='+', color='r')
    plt.scatter(emb_2[:, 0], emb_2[:, 1], marker='+', color='b')

    plt.show()


if __name__ == '__main__':
    # folderNames = ['aug_fancy', 'PGD', 'vanilla']
    folderNames = ['PGD']
    dataBaseNames = ['ADNI', 'AIBL', 'MIRIAD', 'OASIS3']

    for dbn in dataBaseNames:
        for fn in folderNames:
            emb, lo, lp = getEmebeddingAndLabel(fn, dbn)

            analyzeEmbeddings(emb, lo)
