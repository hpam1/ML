"""
FeatureExtractor.py

Dependencies: opencv version 3, numpy, sklearn

Usage: FeatureExtractor.py <training image folder> <training label file> <test image folder> <training feature output file> <test feature output file>
Sample usage: FeatureExtract.py train traincsv.csv test trainFr.csv testFr.csv

This program processes the training and testing image files and creates a edge based feature
representing the images. They are finally written to the files which can be used by the
classifier module for classifying the images.

The training feature csv file is of the form: image label, << 20 features>>. Similarly, the
test feature csv file is of the form: image id, <<20 features>> 
"""

import cv2
import os
import numpy as  np
from sklearn import decomposition
import sys

def imagePreprocessing(fname):
    """
    Performs image preprocessing
    input:
        fname: The path to the input jpg image
    output:
        feature vector (size: 65025) corresponding to the image
    Logic:
        OpenCV is used for reading the input file. The input image is read in grayscale
        and is resized to 255 * 255 pixels to ensure uniform image size.
        Bilateral filtering is applied to reduce the noise the image and to sharpen the edges
        Canny edge detector is applied to extract the edge features from the image and this is 
        flattened to a 1-D vector
    """
    img = cv2.imread(fname, 0)
    img = cv2.resize(img, (255, 255))
    img = cv2.bilateralFilter(img,5,100,100)
    img = cv2.Canny(img,100,50)
    img = img.flatten()
    return img

def getTrainingData(dirPath, labelDict):
    """
    Reads the input image files and creates a feature vector for each image
    Input:
        1. labelDict: a dictionary that has the label value of the image file
    output:
        training data feature and label value
    Logic:
        For each input image, apply image preprocessing to get its features
        When 1000 input data have been read, apply PCA on the input data to reduce the dimensionality
        and memory consumption.
    """
    feature = []
    Y = []
    i = 0
    buffer = []
    for name in os.listdir(dirPath):
        fname = os.path.join(dirPath, name)
        # get the image file name with any extension
        imgId = name.split('.')[0]
        if imgId in labelDict:
            try:
                # get feature X for the image
                fv = imagePreprocessing(fname)
                # add the image class to the Y variable
                Y.append(labelDict[imgId])
                buffer.append(fv)
                i = i + 1
            except:
                # in case of corrupt file, the file is ignored from being considered for training
                print "Warning: Error opening ", fname , " Ignored"
            # apply PCA on the features
            if i % 1000 == 0:
                data = np.array(buffer)
                if i == 1000:
                    pca.fit(data)
                X = pca.transform(data)
                feature.append(X)
                buffer = []
    # apply pca on the last remaining files
    if len(buffer) > 0:
        data = np.array(buffer)
        X = pca.transform(data)
        feature.append(X)
        buffer = []

    print "completed feature extraction"
    feature = np.concatenate(feature, axis=0)
    Y = np.array(Y).reshape(len(Y),1)
    return (feature, Y)

def getTrainingLabels(trainingFile):
    """
    Finds the class of the input image files
    Reads in the train.csv file that contains the class values of the image files 
    and creates a dictionary of the form: <image file id: class value>
    output:
        label dictionary
    """
    # read the train.csv file. The first column of the file specifies the image file
    # id and the remaining columns specify the class value
    fileId = np.loadtxt(trainingFile, delimiter=",", dtype=str, usecols=[0])
    classProb = np.loadtxt(trainingFile, delimiter=',',usecols=range(1,9))
    # find the index which is non-zero. the index corresponds to the class label id
    nonzeroInd = np.nonzero(classProb)
    fileId = fileId[nonzeroInd[0]]
    label = nonzeroInd[1] + 1
    # create a dictionary structure
    fileId = fileId.reshape(len(fileId), 1)
    label = label.reshape(len(label), 1)
    fileLabels = np.concatenate((fileId, label), axis=1)
    test = dict(fileLabels)
    return test

def getTestData(testPath):
    """
    Generates the features for the test images
    Input:
        dirpath: the folder containing the test image files
    output:
        the features for the test images, the image ids
    Logic:
        Image preprocessing is performed on each image and PCA is applied to reduce the dimensionality
        In case of corrupt files, the features are all assumed to be 0.
    """
    feature = []
    idLst = []
    i = 0
    buffer = []
    for name in os.listdir(testPath):
        fname = os.path.join(testPath, name)
        imgId = name.split('.')[0]                    
        idLst.append(imgId)
        try:
            # get the edge features for the test file
            fv = imagePreprocessing(fname)
            buffer.append(fv)
        except:
            # in case of corrupt files, create a numpy matrix with all values to be zeros
            print "Warning: error reading ", fname
            fv = np.zeros(65025)
            buffer.append(fv)
        i = i + 1
        # perform pca
        if i % 1000 == 0:
                data = np.array(buffer)
                X = pca.transform(data)
                feature.append(X)
                buffer = []
                
    # perform pca on last remaining data    
    if len(buffer) > 0:
        data = np.array(buffer)
        X = pca.transform(data)
        feature.append(X)
        buffer = []
    # return the features and the image id
    feature = np.concatenate(feature, axis=0)
    idLst = np.array(idLst).reshape(len(idLst),1)
    return (feature, idLst)

pca = decomposition.RandomizedPCA(n_components=20, whiten=True)
if __name__ == "__main__":
    if len(sys.argv) != 6:
        print "Usage: FeatureExtractor.py <training image folder> <training label file> <test image folder> <training feature output file> <test feature output file>"
        print "Sample usage: FeatureExtract.py train traincsv.csv test trainFr.csv testFr.csv"
    else:
        trDirPath = sys.argv[1]
        trLabelFile = sys.argv[2]
        testPath = sys.argv[3]
        trFrFile = sys.argv[4]
        testFrFile = sys.argv[5]
        
        print "processing training image"
        labels = getTrainingLabels(trLabelFile)
        data = getTrainingData(trDirPath,labels)
        trainX = data[0]
        trainY = data[1]
        # the output training feature file is of the form label, <<20 features>>
        output = np.concatenate((trainY, trainX), axis = 1)
        np.savetxt(trFrFile, output, fmt='%s', delimiter=',')

        print "processing testing image"
        data = getTestData(testPath)
        testX = data[0]
        fileId = data[1]
        # the output test feature file is of the form label, <<20 features>>
        output = np.concatenate((fileId, testX), axis = 1)
        np.savetxt(testFrFile, output, fmt='%s', delimiter=',')
