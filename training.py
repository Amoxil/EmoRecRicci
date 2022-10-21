from asyncio.windows_utils import pipe
from dataclasses import dataclass
from mmap import ACCESS_WRITE
from operator import index
from unicodedata import category
from xml.etree.ElementTree import tostring
import pandas
import numpy
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, LeaveOneOut, StratifiedKFold, RepeatedKFold
from sklearn.metrics import accuracy_score

def trainTestKFold(data, classifier):

    
    kFold = KFold(n_splits=10)
    ricciCurvData = pandas.read_csv(data, header=None)
    #print(ricciCurvData)
    df = ricciCurvData.iloc[: , 1:-1]
    labels = ricciCurvData.iloc[:,-1:]

    """
    scores = []
    for train_set, test_set in kFold.split(df):
        classifier.fit(df.loc[train_set], labels.loc[train_set].values.ravel())
        score = classifier.score(df.loc[test_set], labels.loc[test_set].values.ravel())
        print(df.loc[test_set], labels.loc[test_set])
        print("-----------------------------------------------------------")
        scores.append(score)
    
    print(scores)
    print(numpy.array(scores).mean())
    """
    kfResults = cross_val_score(estimator=classifier, X=df, y=labels.values.ravel(), scoring="accuracy", cv=kFold)
    #print(kfResults)
    score = round((kfResults.mean()*100),2)
    print("K-Fold cv accuracy: " + str(score))

def trainTestStratKFold(data, classifier):

    
    stratKFold = StratifiedKFold(n_splits=10)
    ricciCurvData = pandas.read_csv(data, header=None)
    #print(ricciCurvData)
    df = ricciCurvData.iloc[: , 1:-1]
    labels = ricciCurvData.iloc[:,-1:]

    kfResults = cross_val_score(estimator=classifier, X=df, y=labels.values.ravel(), scoring="accuracy", cv=stratKFold)
    #print(kfResults)
    score = round((kfResults.mean()*100),2)
    print("Stratified K-Fold cv accuracy: " + str(score))

def trainTestRepeatKFold(data, classifier):

    repKFold = RepeatedKFold(n_splits=10, n_repeats=10)
    ricciCurvData = pandas.read_csv(data, header=None)
    #print(ricciCurvData)
    df = ricciCurvData.iloc[: , 1:-1]
    labels = ricciCurvData.iloc[:,-1:]
  
    kfResults = cross_val_score(estimator=classifier, X=df, y=labels.values.ravel(), scoring="accuracy", cv=repKFold)
    #print(kfResults)
    score = round((kfResults.mean()*100),2)
    print("Reapeated K-Fold cv accuracy: " + str(score))

def trainTestLoocv(data, classifier):

    looCV = LeaveOneOut()
    ricciCurvData = pandas.read_csv(data, header=None)
    #print(ricciCurvData)
    df = ricciCurvData.iloc[: , 1:-1]
    labels = ricciCurvData.iloc[:,-1:]

    loocvResults = cross_val_score(estimator=classifier, X=df, y=labels.values.ravel(), scoring='accuracy', cv=looCV)
    #print(loocvResults)
    score = round((loocvResults.mean()*100),2)
    print("Leave one out cv accuracy: " + str(score))

def trainTestHoldOut(data, classifier):

    ricciCurvData = pandas.read_csv(data, header=None)
    #print(ricciCurvData)
    df = ricciCurvData.iloc[: , 1:-1]
    labels = ricciCurvData.iloc[:,-1:]

    dfTrain, dfTest, labelsTrain, labelsTest = train_test_split(df, labels, test_size=0.2)
    classifier.fit(dfTrain, labelsTrain.values.ravel())
    predictions = classifier.predict(dfTest)
    accuracy = accuracy_score(labelsTest, predictions)
    print("Hold out accuracy: " + str(accuracy))

def trainTestSubInd(data, classifier):
    ricciCurvData = pandas.read_csv(data, header=None)
    accuracy = []
    currN = 5
    n = str(currN).zfill(3)
    prefix = 'S' + n
    while(currN<=999):
        test = ricciCurvData.loc[ricciCurvData[0].str.startswith(prefix), :]
        if not test.empty:
            train = pandas.concat([ricciCurvData,test]).drop_duplicates(keep=False)
            trainLabels = train.iloc[:,-1:]
            train = train.iloc[: , 1:-1]
            testLabels = test.iloc[:,-1:]
            test = test.iloc[: , 1:-1]
            classifier.fit(train, trainLabels.values.ravel())
            predictions = classifier.predict(test)
            accuracy.append(accuracy_score(testLabels, predictions))

        currN = currN + 1
        n = str(currN).zfill(3)
        prefix = 'S' + n
    
    arr = numpy.array(accuracy)
    print(arr)
    print(arr.mean())

@ignore_warnings(category=UserWarning)
def trainTestAll(data, classifier):
    trainTestKFold(data, classifier)
    trainTestStratKFold(data, classifier)
    trainTestRepeatKFold(data, classifier)
    trainTestLoocv(data, classifier)

#def testImage(data, classifier, image):
