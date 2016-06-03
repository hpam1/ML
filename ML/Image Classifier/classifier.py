# -*- coding: utf-8 -*-
"""
classifier.py

Dependencies: sklearn, numpy

Usage: classifier.py <<training data file>> <<test data file>> <<output file>> <<mode>> SVM
Possible values for <<mode>> are 0 and 1
mode = 0: cross validation mode; performs cross validation on training data and computes log loss
          to estimate the performance of the classifier
mode = 1: generates the output for the test data
If mode is 0, the test data file and output file are optional
Example usage:
classifier.py train.csv test.csv output.csv 1 SVM
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import sys


def getData(fileDir):
    """
    Read input data into numpy array
    Input: The location of the input file
    output: The X and Y values    
    """
    print "in get data", fileDir
    Y = np.loadtxt(fileDir, delimiter=",", usecols=[0])
    X = np.loadtxt(fileDir, delimiter=',',usecols=range(1,21))
    print "read data successfully"
    return (X, Y)

    
def trainClassifierMode(trainX, trainY, clf, mode):
    """
    trains the classifier according to the execution mode
    input:
        1. trainX: the input data matrix
        2. trainY: the input labels
        3. clf: classifier
        4. execution mode: 0 (cross validation) 1(train the classifier to predict the test data)
    output:
        the trained classifier
    """
    if mode == 0:
        # perform 3 fold cross validation and compute log loss on the validation set
        print len(trainX)
        kfold = cross_validation.KFold(len(trainX), n_folds=3)
        for train, test in kfold:
            clf.fit(trainX[train], trainY[train])
            actY = clf.predict_proba(trainX[test])
            computeLogLoss(actY, trainY[test])
    else:
        # train the classifier
        clf.fit(trainX, trainY)
    return clf
    
def trainClassifier(trainX, trainY, classifier, mode):
    """
    creates classifier objects
    input:
        1. trainX: the training data 
        2. trainY: training data labels
        3. classifier: the classifier to be created
        4. execution mode: 0 or 1
    output:
        trained classifier
    """
    print "in train classifier ", classifier
    clf = None
    if classifier == 'SVM':
        clf = SVC(kernel='rbf', probability=True, C=1, gamma=0.1, cache_size=1000)
    elif classifier == 'RF':
        clf = RandomForestClassifier(n_estimators= 200, min_samples_leaf=20)
    elif classifier == 'adaDT':
       clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=200,learning_rate=1.5,algorithm="SAMME.R")
    elif classifier == 'adaRF':
       clf = AdaBoostClassifier(base_estimator = RandomForestClassifier(n_estimators = 20), algorithm = 'SAMME.R')    
    elif classifier == 'LR':
        clf = linear_model.LogisticRegression(C=1)
    elif classifier == 'NB':
        clf = GaussianNB()
    return trainClassifierMode(trainX, trainY, clf, mode)
    
def test(testX, classifier):
    """
    predicts the output of the test data
    input: 
        1. testX: test data features
        2. classifier: the trained classifier
    output:
        [19648 * 8] matrix specifying the probability of a specific image belonging to the 8 classes
    """
    actY = classifier.predict_proba(testX)
    return actY

def computeLogLoss(actY, expY):
    """
    computes log loss on validation set
    input:
        1. actY: the Y value obtained from the classifier
        2. expY: the ground truth
    output:
        multi-class log loss
    Reference:
        https://www.kaggle.com/wiki/MultiClassLogLoss
    """
    print "computing log loss"
    N = len(actY)
    # adjust values for 0 and 1
    index = np.where(actY == 0)
    actY[index] = 0.000000000000001
    index = np.where(actY == 1)
    actY[index] = 0.999999999999999
    # log of the input probabilities
    actY = np.log(actY)
    Y = []
    for val in expY:
        output = np.zeros(8)
        v = int(val)
        output[v-1] = 1
        Y.append(output)
    Y = np.array(Y)
    logLoss = np.multiply(actY, Y)
    loss = np.sum(logLoss) / N
    loss = loss * -1
    print "log loss: ", loss

def main(trainFile, testFile, outputFile, mode, classifier):
    """
    input:
        1. trainFile: the training data features file
        2. testFile: the test data file
        3. outputFile: the file where the output of the test data has to be written
        4. classifier: the classifier to be used
    """
    # scale the input data
    scaler = StandardScaler()
    trainingData = getData(trainFile)
    trainX = trainingData[0]
    trainY = trainingData[1]
    trainX = scaler.fit_transform(trainX)
    testX = []
    testY = []
    # train the classifier
    clf = trainClassifier(trainX, trainY, classifier, mode)
    # if test mode, get test data and predict the output classes
    if mode == 1:
        testData = getData(testFile)
        testX = testData[0]
        testY = testData[1]
        testX = scaler.transform(testX)
        actY = test(testX, clf)
        testY = testY.reshape(len(testY), 1)
        # write the predicted class probabilities
        output = np.concatenate((testY, actY), axis = 1)
        np.savetxt(outputFile, output, fmt='%s', delimiter=',')
    
if __name__ == "__main__":
    if len(sys.argv) != 6:
        print "Usage: classifier.py <training Feature file> <test feature file> <output file> <execution mode> <classifier code>"
        print "Execution mode: 0 - validation mode; 1 - produce output for test data"
        print "Execution Mode 0 : <test feature file> and <output file> can be given as None"
        print "Sample usage with Execution mode 0: classifier.py traincsv.csv None None 0 SVM"
        print "Execution mode 1: predicts the output for test data input (test feature file> and is written to <output file>"
        print "Sample usage with Execution mode 1: classifier.py traincsv.csv test.csv output.csv SVM"
    else:
        trainFile = sys.argv[1]
        testFile = sys.argv[2]
        outputFile = sys.argv[3]
        mode = int(sys.argv[4])
        classifier = sys.argv[5]
        main(trainFile, testFile, outputFile, mode, classifier)

