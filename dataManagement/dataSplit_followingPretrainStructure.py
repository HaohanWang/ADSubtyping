__author__ = 'Haohan Wang'

def getDemographicInformation():
    text = [line.strip() for line in open('/media/haohanwang/Info/ADNI_CAPS/split.stratified.0.csv')]
    demo = {}
    for line in text[1:]:
        items = line.split(',')
        if items[0] not in demo:
            demo[items[0]] = {}
        demo[items[0]][items[1]] = (items[2], items[3], items[4])
    return demo

def getCurrentSplitFile(fold):

    fileFolder = '../pretrainModels/performances/fold_' + str(fold) + '/'
    trainSplit = {}
    valSplit = {}
    testSplit = {}
    trainText = [line.strip() for line in open(fileFolder + 'train_subject_level_result.tsv')]
    for line in trainText[1:]:
        items = line.split('\t')
        if items[0] not in trainSplit:
            trainSplit[items[0]] = {}
        trainSplit[items[0]][items[1]] = 'train'
    valText = [line.strip() for line in open(fileFolder + 'valid_subject_level_result.tsv')]
    for line in valText[1:]:
        items = line.split('\t')
        if items[0] not in valSplit:
            valSplit[items[0]] = {}
        valSplit[items[0]][items[1]] = 'val'

    testText = [line.strip() for line in open(fileFolder + 'test-ADNI_subject_level_result.tsv')]
    for line in testText[1:]:
        items = line.split('\t')
        if items[0] not in testSplit:
            testSplit[items[0]] = {}
        testSplit[items[0]][items[1]] = 'test'

    return trainSplit, valSplit, testSplit

def generateNewSplit(fold):

    trainSplit, valSplit, testSplit = getCurrentSplitFile(fold)
    demo = getDemographicInformation()

    f = open('/media/haohanwang/Info/ADNI_CAPS/split.pretrained.' + str(fold) +'.csv', 'w')
    f.writelines('participant_id,session_id,age,sex,diagnosis,split\n')
    for sub in demo:
        if sub in trainSplit:
            for sess in demo[sub]:
                ms = demo[sub][sess]
                f.writelines(sub + ',' + sess + ',')
                for m in ms:
                    f.writelines(m + ',')
                f.writelines('train\n')
        elif sub in valSplit:
            for sess in demo[sub]:
                ms = demo[sub][sess]
                f.writelines(sub + ',' + sess + ',')
                for m in ms:
                    f.writelines(m + ',')
                f.writelines('val\n')
        elif sub in testSplit:
            for sess in demo[sub]:
                ms = demo[sub][sess]
                f.writelines(sub + ',' + sess + ',')
                for m in ms:
                    f.writelines(m + ',')
                f.writelines('test\n')
        else:
            for sess in demo[sub]:
                ms = demo[sub][sess]
                f.writelines(sub + ',' + sess + ',')
                for m in ms:
                    f.writelines(m + ',')
                f.writelines('train\n')


if __name__ == '__main__':
    for i in range(5):
        generateNewSplit(i)