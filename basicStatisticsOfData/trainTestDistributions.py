__author__ = 'Haohan Wang'

import numpy as np

def checkMajorityPredictionAccuracy_ADNI(csvFile):
    text = [line.strip() for line in open(csvFile)][1:]

    diag = {'AD': 0, 'CN': 0}

    for line in text:
        items = line.split(',')

        if items[-1] =='test':
            if items[-2] == 'AD':
                diag['AD'] += 1
            elif items[-2] == 'CN':
                diag['CN'] += 1


    t = diag['AD'] + diag['CN']

    print ('AD', diag['AD']/float(t), end='\t')
    print ('CN', diag['CN']/float(t))

def checkMajorityPredictionAccuracy(csvFile):
    text = [line.strip() for line in open(csvFile)][1:]

    diag = {'AD': 0, 'CN': 0}

    for line in text:
        items = line.split(',')

        if items[-2] == 'AD':
            diag['AD'] += 1
        elif items[-2] == 'CN':
            diag['CN'] += 1

        if items[-1] == 'AD':
            diag['AD'] += 1
        elif items[-1] == 'CN':
            diag['CN'] += 1

    t = diag['AD'] + diag['CN']

    print ('AD', diag['AD']/float(t), end='\t')
    print ('CN', diag['CN']/float(t))


def checkAgeGenderDistributionOfLabels(csvFile, onlyTraining = False):

    from scipy.stats import spearmanr
    from scipy.stats import ttest_ind

    if csvFile.find('OASIS3') != -1:
        return None

    text = [line.strip() for line in open(csvFile)][1:]

    diags = []
    ages = []
    genders = []

    for line in text:
        items = line.split(',')

        if onlyTraining:
            if items[-1] != 'train':
                continue

        diag = None
        if items[4] == 'AD':
            diag = 0
        elif items[4] == 'CN':
            diag = 1

        age = float(items[2])
        gender = 1 if items[3].startswith('F') else 0

        if diag is not None:
            diags.append(diag)
            ages.append(age)
            genders.append(gender)

    diags = np.array(diags)
    ages = np.array(ages)
    genders = np.array(genders)

    # print (diags)
    # print (ages)
    # print (genders)

    ages1, ages2 = ages[diags==0], ages[diags==1]
    genders1, genders2 = genders[diags==0], genders[diags==1]

    _, p_age = ttest_ind(ages1, ages2)
    _, p_gender = ttest_ind(genders1, genders2)

    print ('age', p_age)
    print ('gender', p_gender)

if __name__ == '__main__':
    csvFile1 = '/media/haohanwang/Storage/AlzheimerImagingData/ADNI_CAPS/split.pretrained.0.csv'
    csvFile2 = '/media/haohanwang/Storage/AlzheimerImagingData/AIBL_CAPS/aibl_info.csv'
    csvFile3 = '/media/haohanwang/Storage/AlzheimerImagingData/MIRIAD_CAPS/miriad_test_info.csv'
    csvFile4 = '/media/haohanwang/Storage/AlzheimerImagingData/OASIS3_CAPS/oasis3_test_info_2.csv'

    # checkMajorityPredictionAccuracy_ADNI(csvFile=csvFile1)
    # checkMajorityPredictionAccuracy(csvFile=csvFile2)
    # checkMajorityPredictionAccuracy(csvFile=csvFile3)
    # checkMajorityPredictionAccuracy(csvFile=csvFile4)

    checkAgeGenderDistributionOfLabels(csvFile=csvFile1, onlyTraining=True)
    # so it seems there are dependencies between both age and gender and the diagnosis in the training data
    checkAgeGenderDistributionOfLabels(csvFile=csvFile2)
    checkAgeGenderDistributionOfLabels(csvFile=csvFile3)
    checkAgeGenderDistributionOfLabels(csvFile=csvFile4)