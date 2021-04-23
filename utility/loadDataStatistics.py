__author__ = 'Haohan Wang'

filePath = '/media/haohanwang/Elements/attack_visualization/'

def loadInformation(trainFlag=True):
    if trainFlag:
        text = [line.strip() for line in open(filePath + 'train_attack_info.csv')][1:]
    else:
        text = [line.strip() for line in open(filePath + 'test_attack_info.csv')][1:]

    labels = []
    predictions = []
    for line in text:
        items = line.split(',')
        labels.append(int(items[2]))
        predictions.append(float(items[3]))
    return labels, predictions

def sortInformation(trainFlag=True):
    if trainFlag:
        text = [line.strip() for line in open(filePath + 'train_attack_info.csv')][1:]
    else:
        text = [line.strip() for line in open(filePath + 'test_attack_info.csv')][1:]

    info = []

    for line in text:
        info.append(line.split(','))
    info = sorted(info)

    for line in info:
        print (line)

    f = open('../outputs/sortedInfo.csv', 'w')
    for line in info:
        f.writelines(','.join(line)+'\n')
    f.close()

if __name__ == '__main__':
    sortInformation(False)