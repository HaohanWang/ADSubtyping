__author__ = 'Haohan Wang'

import numpy as np

def id_check():
    text1 = [line.strip() for line in open('../data/split.stratified.0.csv')]
    id1 = []
    for line in text1:
        s = line.split(',')[0][8:]
        id1.append(s)
    text2 = [line.strip() for line in open('../data/samples.txt')]
    id2 = []
    for line in text2:
        id2.append(line.replace('_',''))

    print len(id1)
    print len(id2)

    l = []
    for m in id2:
        if m in id1:
            l.append(m)

    l = set(l)

    print len(l)

if __name__ == '__main__':
    id_check()