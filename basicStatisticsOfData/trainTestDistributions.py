__author__ = 'Haohan Wang'


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


if __name__ == '__main__':
    csvFile1 = '/media/haohanwang/Storage/AlzheimerImagingData/ADNI_CAPS/split.pretrained.0.csv'
    csvFile2 = '/media/haohanwang/Storage/AlzheimerImagingData/AIBL_CAPS/aibl_info.csv'
    csvFile3 = '/media/haohanwang/Storage/AlzheimerImagingData/MIRIAD_CAPS/miriad_test_info.csv'
    csvFile4 = '/media/haohanwang/Storage/AlzheimerImagingData/OASIS3_CAPS/oasis3_test_info_2.csv'

    checkMajorityPredictionAccuracy_ADNI(csvFile=csvFile1)
    checkMajorityPredictionAccuracy(csvFile=csvFile2)
    checkMajorityPredictionAccuracy(csvFile=csvFile3)
    checkMajorityPredictionAccuracy(csvFile=csvFile4)