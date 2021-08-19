__author__ = 'Haohan Wang'

fileFolder = '/media/haohanwang/Info/ADNI_CAPS/'

def loadTrainTestInformation():
    text = [line.strip() for line in open(fileFolder + 'split.stratified.0.csv')]

    train = {}
    test = {}
    val = {}

    for line in text[1:]:
        items = line.split(',')

        if items[-1] == 'train':
            if items[-2] not in train:
                train[items[-2]] = 1
            else:
                train[items[-2]] += 1

        elif items[-1] == 'val':
            if items[-2] not in val:
                val[items[-2]] = 1
            else:
                val[items[-2]] += 1

        elif items[-1] == 'test':
            if items[-2] not in test:
                test[items[-2]] = 1
            else:
                test[items[-2]] += 1

    print (train, test, val)

    train_s = 0
    test_s = 0
    val_s = 0

    for k in train:
        train_s += train[k]
    for k in test:
        test_s += test[k]
    for k in val:
        val_s += val[k]

    for k in train:
        train[k] = train[k]/float(train_s)
    for k in test:
        test[k] = test[k]/float(test_s)
    for k in val:
        val[k] = val[k]/float(val_s)

    print (train, test, val)


if __name__ == '__main__':
    loadTrainTestInformation()