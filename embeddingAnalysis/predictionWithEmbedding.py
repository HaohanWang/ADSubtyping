__author__ = 'Haohan Wang'

from embeddingLoader import embeddingFileLoader

import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as mse

from matplotlib import pyplot as plt

def checkPreformances(model, Xtest, Ytest):
    p = model.predict(Xtest)
    return accuracy_score(Ytest, p)

def checkPreformances_mse(model, Xtest, Ytest):
    p = model.predict(Xtest)
    return mse(Ytest, p)

def fittingCovariates(X, a, b):
    c = np.append(a.reshape([-1, 1]), b.reshape([-1, 1]), 1)

    models = []
    for i in range(X.shape[1]):
        m = LinearRegression()
        m.fit(c, X[:, i])
        models.append(m)
    return models

def calculateCovariatesResidue(modelList, X, a, b):

    c = np.append(a.reshape([-1, 1]), b.reshape([-1, 1]), 1)

    for i in range(X.shape[1]):
        X[:, i] = X[:, i] - modelList[i].predict(c)
    return X

def basicPrediction(folderName):

    print (folderName)
    print ('----------')

    Xtrain, Ytrain, age_train, gender_train = embeddingFileLoader(folderName, 'ADNI', 'train')

    Xval, Yval, age_val, gender_val = embeddingFileLoader(folderName, 'ADNI', 'val')

    Xtest1, Ytest1, age_test1, gender_test1 = embeddingFileLoader(folderName, 'ADNI')

    Xtest2, Ytest2, age_test2, gender_test2 = embeddingFileLoader(folderName, 'AIBL')

    Xtest3, Ytest3, age_test3, gender_test3 = embeddingFileLoader(folderName, 'MIRIAD')

    Xtest4, Ytest4, _, _ = embeddingFileLoader(folderName, 'OASIS3')

    covariateModelList = fittingCovariates(Xtrain, age_train, gender_train)

    Xtrain = calculateCovariatesResidue(covariateModelList, Xtrain, age_train, gender_train)
    Xval = calculateCovariatesResidue(covariateModelList, Xval, age_val, gender_val)
    Xtest1 = calculateCovariatesResidue(covariateModelList, Xtest1, age_test1, gender_test1)
    Xtest2 = calculateCovariatesResidue(covariateModelList, Xtest2, age_test2, gender_test2)
    Xtest3 = calculateCovariatesResidue(covariateModelList, Xtest3, age_test3, gender_test3)

    model = SVC(kernel='rbf')

    model.fit(Xtrain, Ytrain)
    print ('training performances', checkPreformances(model, Xtrain, Ytrain))
    print ('validation performances', checkPreformances(model, Xval, Yval))
    print ('ADNI performances', checkPreformances(model, Xtest1, Ytest1))
    print ('AIBL performances', checkPreformances(model, Xtest2, Ytest2))
    print ('MIRIAD performances', checkPreformances(model, Xtest3, Ytest3))
    print ('OASIS3 performances', checkPreformances(model, Xtest4, Ytest4))

    print ('----------')


def variableSelection(folderName):

    print (folderName)
    print ('----------')

    Xtrain, Ytrain, age_train, gender_train = embeddingFileLoader(folderName, 'ADNI', 'train')

    Xval, Yval, age_val, gender_val = embeddingFileLoader(folderName, 'ADNI', 'val')

    Xtest1, Ytest1, age_test1, gender_test1 = embeddingFileLoader(folderName, 'ADNI')

    Xtest2, Ytest2, age_test2, gender_test2 = embeddingFileLoader(folderName, 'AIBL')

    Xtest3, Ytest3, age_test3, gender_test3 = embeddingFileLoader(folderName, 'MIRIAD')

    Xtest4, Ytest4, _, _ = embeddingFileLoader(folderName, 'OASIS3')

    model = LogisticRegression(penalty='elasticnet', C=1e-1, l1_ratio=0.5, solver='saga')
    # 1e-1 seems to be a reasonable setting that leads to a reasonable sparse, yet powerful embedding set

    model.fit(Xtrain, Ytrain)

    print ('training performances', checkPreformances(model, Xtrain, Ytrain))
    print ('validation performances', checkPreformances(model, Xval, Yval))
    print ('ADNI performances', checkPreformances(model, Xtest1, Ytest1))
    print ('AIBL performances', checkPreformances(model, Xtest2, Ytest2))
    print ('MIRIAD performances', checkPreformances(model, Xtest3, Ytest3))
    print ('OASIS3 performances', checkPreformances(model, Xtest4, Ytest4))

    coef = model.coef_

    print (coef)
    plt.plot(coef)
    plt.show()

    print ('non zeros:', len(np.where(coef!=0)[0]))

    print ('----------')


if __name__ == '__main__':
    folderNames = ['aug_fancy', 'PGD', 'vanilla']

    for fn in folderNames:
        variableSelection(fn)
