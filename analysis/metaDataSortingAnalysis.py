__author__ = 'Haohan Wang'

def checkMCIvAD():
    text = [line.strip() for line in open('../data/split_1.csv')][1:]

    data = {}

    for line in text:
        items = line.split(',')
        k = items[0]
        s = items[1]
        d = items[2]

        if k not in data:
            data[k] = []

        data[k].append((s, d))

    for k in data:
        data[k] = sorted(data[k])

    caseCount = 0
    controlCount = 0

    for k in data:

        MCIFlag = False
        ADFlag = False

        for i in range(len(data[k])):
            d = data[k][i][1]
            if ADFlag:
                break
            if d == 'MCI' and i+1!=len(data[k]):
                MCIFlag = True
            if d == 'AD':
                ADFlag = True

        if MCIFlag and (not ADFlag):
            controlCount += 1
        if MCIFlag and ADFlag:
            caseCount += 1

    print caseCount
    print controlCount


if __name__ == '__main__':
    checkMCIvAD()