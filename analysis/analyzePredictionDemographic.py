__author__ = 'Haohan Wang'

import sys

import numpy as np

from matplotlib import pyplot as plt


def analyzePredictionResult(folderName, dataBase):
    if folderName == 'aug_fancy':
        fileName = folderName + '/result_aug_fancy_fold_0_seed_1_epoch_120_' + dataBase + '.csv'
    elif folderName == 'PGD':
        fileName = folderName + '/result_aug_pgd_0.001_fold_0_seed_1_epoch_45_' + dataBase + '.csv'
    elif folderName == 'vanilla':
        fileName = folderName + '/result_aug_fold_0_seed_1_epoch_50_' + dataBase + '.csv'
    else:
        sys.exit()

    if dataBase == 'ADNI':
        gsIdx = 4
        pdIdx = 6
        agIdx = 2
        gdIdx = 3
    elif dataBase == 'AIBL' or dataBase == 'MIRIAD':
        gsIdx = 4
        pdIdx = 5
        agIdx = 2
        gdIdx = 3
    elif dataBase == 'OASIS3':
        gsIdx = 2
        pdIdx = 4
        agIdx = None
        gdIdx = None
    else:
        sys.exit()


    text = [line.strip() for line in open('../mainScripts/predictionResults/' + fileName)]

    gs = []
    pd = []
    age = []
    gender = []

    for line in text:
        items = line.split(',')
        if items[gsIdx] == 'CN':
            gs.append(0)
        else:
            gs.append(1)

        if items[pdIdx] == 'CN':
            pd.append(0)
        else:
            pd.append(1)

        age.append(float(items[agIdx]))

        if items[gdIdx].startswith('F'):
            gender.append(0)
        else:
            gender.append(1)

    gs = np.array(gs)
    pd = np.array(pd)
    age = np.array(age)
    gender = np.array(gender)

    age_results = []
    gender_results = []

    age_results.append(age[np.logical_and(gs==0, pd==0)]) # age_cn2cn
    age_results.append(age[np.logical_and(gs==1, pd==1)]) # age_ad2ad
    age_results.append(age[np.logical_and(gs==0, pd==1)]) # age_cn2ad
    age_results.append(age[np.logical_and(gs==1, pd==0)]) # age_ad2cn

    gender_results.append(gender[np.logical_and(gs==0, pd==0)]) # gender_cn2cn
    gender_results.append(gender[np.logical_and(gs==1, pd==1)]) # gender_ad2ad
    gender_results.append(gender[np.logical_and(gs==0, pd==1)]) # gender_cn2ad
    gender_results.append(gender[np.logical_and(gs==1, pd==0)]) # gender_ad2cn

    plt.boxplot(age_results)
    plt.xticks([1, 2, 3, 4], ['cn2cn', 'ad2ad', 'cn2ad', 'ad2cn'])
    plt.savefig('predictionAnalysisFigs/detail_' + folderName + '_' + dataBase + '.png')
    plt.clf()

    age_preds = []
    age_preds.append(age[pd==0])
    age_preds.append(age[pd==1])

    plt.boxplot(age_preds)
    plt.xticks([1, 2], ['pred_CN', 'pred_AD'])
    plt.savefig('predictionAnalysisFigs/prediction_' + folderName + '_' + dataBase + '.png')
    plt.clf()

if __name__ == '__main__':
    folderNames = ['aug_fancy', 'PGD', 'vanilla']
    dataBaseNames = ['ADNI', 'AIBL', 'MIRIAD']

    for dbn in dataBaseNames:
        for fn in folderNames:
            analyzePredictionResult(fn, dbn)
