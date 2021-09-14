__author__ = 'Haohan Wang'

import numpy as np

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error as mse

from LMM.LMM import LinearMixedModel
from LMM.helpingMethods import *

def multiRegression():
    # this does not really work because the the regression fails to learn anything (maybe the dimension is too high)

    def trainTestSplit(data):
        return data[:124], data[124:]

    mriData = np.load('../data/multiRegression/mriData.npy')*100
    snpData = np.load('../data/multiRegression/snps.npy')
    markers = [line.strip() for line in open('../data/multiRegression/markers.txt')]

    xtr, xte = trainTestSplit(snpData)

    print (np.mean(mriData), np.std(mriData))

    model = Lasso()

    for i in range(mriData.shape[1]):
        print ('target', i,)
        ytr, yte = trainTestSplit(mriData[:, i])

        model.fit(xtr, ytr)
        ypred = model.predict(xte)
        # print '\t test mse', mse(yte, ypred)
        # print '-------------------'
        # print yte.T
        # print ypred.T
        # print '-------------------'

def multiAssociationStudy():
    mriData = np.load('../data/multiRegression/mriData.npy')*100
    X = np.load('../data/multiRegression/snps.npy').astype(float)
    markers = [line.strip() for line in open('../data/multiRegression/markers.txt')]

    lmm = LinearMixedModel()

    K = matrixMult(X, X.T)
    S, U = linalg.eigh(K)

    for i in range(mriData.shape[1]):
        print ('target', i, 'markers',)

        lmm.fit(X=X, K=K, Kva=S, Kve=U, y=mriData[:, i])
        pValue = lmm.getPvalues()
        idx = np.argsort(pValue)
        for t in range(10):
            print ('\t', markers[idx[t]],)

        print ()

if __name__ == '__main__':
    pass
    # multiRegression()
    multiAssociationStudy()