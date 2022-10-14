from asyncio.windows_utils import pipe
from dataclasses import dataclass
from operator import index
from xml.etree.ElementTree import tostring
import pandas
import numpy
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, LeaveOneOut, StratifiedKFold, RepeatedKFold
from sklearn.metrics import accuracy_score

def trainTestKFold(data):

    
    kFold = KFold(n_splits=10)
    ricciCurvData = pandas.read_csv(data, header=None)
    print(ricciCurvData)
    classifier = RandomForestClassifier(n_estimators=50, random_state=1)
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
    print(kfResults)
    print(kfResults.mean())

def trainTestStratKFold(data):

    
    stratKFold = StratifiedKFold(n_splits=10)
    ricciCurvData = pandas.read_csv(data, header=None)
    print(ricciCurvData)
    classifier = RandomForestClassifier(n_estimators=50, random_state=1)
    df = ricciCurvData.iloc[: , 1:-1]
    labels = ricciCurvData.iloc[:,-1:]

    kfResults = cross_val_score(estimator=classifier, X=df, y=labels.values.ravel(), scoring="accuracy", cv=stratKFold)
    print(kfResults)
    print(kfResults.mean())

def trainTestRepeatKFold(data):

    repKFold = RepeatedKFold(n_splits=10, n_repeats=10)
    ricciCurvData = pandas.read_csv(data, header=None)
    print(ricciCurvData)
    classifier = RandomForestClassifier(n_estimators=50, random_state=1)
    df = ricciCurvData.iloc[: , 1:-1]
    labels = ricciCurvData.iloc[:,-1:]
  
    kfResults = cross_val_score(estimator=classifier, X=df, y=labels.values.ravel(), scoring="accuracy", cv=repKFold)
    print(kfResults)
    print(kfResults.mean())

def trainTestLoocv(data):

    looCV = LeaveOneOut()
    ricciCurvData = pandas.read_csv(data, header=None)
    print(ricciCurvData)
    df = ricciCurvData.iloc[: , 1:-1]
    labels = ricciCurvData.iloc[:,-1:]

    classifier = RandomForestClassifier(n_estimators=50, random_state=1)

    print("I'm a really slow pc but i'm computing, gimme time D:")
    loocvResults = cross_val_score(estimator=classifier, X=df, y=labels.values.ravel(), scoring='accuracy', cv=looCV)
    print(loocvResults)
    print(loocvResults.mean())

def trainTestHoldOut(data):

    ricciCurvData = pandas.read_csv(data, header=None)
    print(ricciCurvData)
    df = ricciCurvData.iloc[: , 1:-1]
    labels = ricciCurvData.iloc[:,-1:]

    classifier = RandomForestClassifier(n_estimators=50, random_state=1)

    dfTrain, dfTest, labelsTrain, labelsTest = train_test_split(df, labels, test_size=0.2)
    classifier.fit(dfTrain, labelsTrain.values.ravel())
    predictions = classifier.predict(dfTest)
    accuracy = accuracy_score(labelsTest, predictions)
    print(accuracy)
    
    