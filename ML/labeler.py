"""
labeler.py

This program contains implementations for Quora Labeler challenge.
https://www.quora.com/challenges#labeler
 
Usage: labeler.py

Packages used: Numpy, nltk, sklearn

Each of the question text is preprocessed to remove stop words, to convert to lower case
and stemming algorithm is applied. Then a TF-IDF matrix is created representing 2000 features
for the data. Logistic Regression is applied to train a model and predict the labels of the
test data
"""
from nltk.tokenize import wordpunct_tokenize
from nltk import stem
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


stop_words = set(['why', 'what', 'how', 'when', 'where', 'whom','which', 'who','whose', 'should', 'must', 'would', 'could', 'not','best', 'all', 'me', 'I', 'you', 'yourself', 'yourselves', 'themselves', 'anyone', 'she', 'he', 'our','did', 'am', 'like', 'given', 'etc', 'the', 'is', 'are', 'was', 'were', 'a', 'an', 'much', 'most', 'way', 'given', 'say', 'said', 'done', 'gave', 'great', 'greater', 'small', 'smaller', 'smallest', 'greatest', 'use', 'll', 'to', 'while', 'these', 'them', 'that', 'those', 'ma', 'having', 'giving', 'just', 'over','very', 'before', 'after', 'dont', 'don', 'once', 'up', 'upon', 'even', 'such', 'wouldnt', 'wouldn', 'shouldnt', 'shouldn', 't', 'since', 'because', 'ever', 'about', 'it', 'actually', 'actual', 'advice', 'advise', 'and', 'or', 'otherwise', 'wise', 'fast', 'quick', 'faster', 'quicker', 'as', 'ask', 'at', 'average', 'back', 'bad', 'be', 'become', 'been', 'behind',' better', 'between', 'biggest', 'but', 'can', 'cant', 'choose', 'common', 'consider', 'everyone', 'example', 'exist', 'my', 'name', 'need', 'never', 'their', 'there', 'they', 've', 'vc', 'think', 'we', 'will', 'wont', 'shall', 'with', 'without', 'out', 'your', 'by', 'make', 'made', 'want', 'so','go', 'into', 'if', 'have','know', 'known', 'of', 'on', 'one', 'same', 'see', 'seen', 'still', 'than', 'then', 'vs', 'too', 'it', 'someone', 'something', 'hi', 'hello', 'better', 'had', 'has', 'have', 'hadnt', 'hasnt', 'across', 'again', 'be', 'big', 'him', 'her', 'hard', 'hardest', 'instead', 'yet', 'though', 'although'])
porter = stem.porter.PorterStemmer()
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 2000)
 
def readInput():
    """
    This function reads in the question text and remvoes all characters that are not alphabets.
    The question text is tokenized and stop words are removed from it. Porter's Stemming 
    algorithm is then applied
    """
    questionText = raw_input()
    lettersOnly = re.sub("[^a-zA-Z]", " ", questionText)
    meaningfulWords = [porter.stem(tokens.lower()) for tokens in wordpunct_tokenize(lettersOnly) if tokens.lower() not in stop_words]
    return " ".join(meaningfulWords)

def generateTrainFeatures(L):
    """
    This function generates the training data features and its target labels.
    Input: L : The number of training data
    Output: trainX -> a (L * 2000) numpy matrix representing the 2000 features for each of the
                        L training samples
            trainY -> (L * 185) numpy matrix representing the target class of the training samples
    Logic:
    The input text is read, preprocessed to remove stop words, and is appended to a list.
    Similarly, each of the target class values are read into a list.
    Sklearn package TFIDF vectorizer is used for generating TFIDF matrix for the 2000 frequent
    words. 
    The multi-label classification algorithms require a target Y variable of the form,
    (nsamples * nclasses), multilabel binarizer is used for converting the list of classes
    to a matrix form.
    """
    global classOrder
    X = []
    Y = []
    # read the input
    for i in range(L):
        categories = raw_input()
        target = [int(y) for y in categories.split(" ")]
        del target[0]
        meaningfulWords = readInput()
        Y.append(target)
        X.append(meaningfulWords)
    # construct TF-IDF matrix representing the features
    trainX = vectorizer.fit_transform(X).toarray()
    # convert the target label list to a suitable matrix form
    mlb = MultiLabelBinarizer()
    trainY = mlb.fit_transform(Y)
    # for representing the order of the classes
    classOrder = mlb.classes_
    return (trainX, trainY)
      
def generateValidFeatures(E):
    """
    This function generates the test data features
    Input: E : The number of testing data
    Output: trainX -> a (E * 2000) numpy matrix representing the 2000 features for each of the
                        E testing samples
    Logic:
    The input text is read and the preprocessing to remove stop words is performed and each text
    is appended to a list. The TF-IDF vectorizer learnt from the training data is applied to 
    generate the features for the test data
    """
    X = []
    for i in range(E):
        meaningfulWords = readInput()
        X.append(meaningfulWords)
    testX = vectorizer.transform(X).toarray()
    return testX
    
def trainAndPredictLR(trainX, trainY, testX):
    """
    Logistic regression is used for predicting the target labels of the test data
    The probability of belonging to each of the labels is predicted for every test
    data and the labels with the top 10 probability values are extracted
    
    Input:
        1. trainX: ntrainingSamples * 2000 numpy matrix representing training data features
        2. trainY: ntrainingSamples * 185 numpy matrix representing the training data labels
        3. testX: ntestSamples * 2000 numpy matrix representing test data features
    
    Output:
        testY: ntestSamples * 19 numpy matrix representing the labels for the test data
    
    """
    clf = OneVsRestClassifier(LogisticRegression(C = 1.0))
    clf.fit(trainX, trainY)
    actY = clf.predict_proba(testX)
    testY = []
    # fetch the labels with max probability
    for prob in actY:
        y = []
        for i in range(10):
            index = np.argmax(prob, axis=0)
            classVal = classOrder[index]
            y.append(classVal)
            prob[index] = -1
        testY.append(y)
    return np.array(testY)

def main():
    T, E = [int(i) for i in raw_input().split(" ")]
    trainX, trainY = generateTrainFeatures(T)
    validX = generateValidFeatures(E)
    validY = trainAndPredictLR(trainX, trainY, validX)
    validYout = re.sub('[\[\]]', '', "\n".join(map(str, validY)))
    print validYout
    
    
classOrder = []
main()