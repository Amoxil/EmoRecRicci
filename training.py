
from cProfile import label
import time
from tkinter import Y
import pandas
import numpy
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, LeaveOneOut, StratifiedKFold, RepeatedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report


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

    real = []
    preds = []
    accuracy = []
    arr = []

    sm = SMOTE()
    X_res, y_res = sm.fit_resample(df, labels.values.ravel())

    for i in range(0, len(X_res)):
        #gets the testing row
        test = X_res.iloc[i]
        #Removes the testing row from the training set
        train = X_res.drop(X_res.index[i])
        #Separates the curvature values from the label
        testLabel = y_res[i]
        trainLabels = numpy.delete(y_res, i)
        #Training of the classifier
        classifier.fit(train, trainLabels)
        
        arr.append(test)
        prediction = classifier.predict(arr)
        arr.clear()
        #Saves the accuracy of each subject tested
        preds.append(prediction)
        real.append(testLabel)
    
    print(classification_report(real, preds))
    conf_mat = confusion_matrix(real, preds)
    print(conf_mat)
    print("Loocv")

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

#@ignore_warnings(category=UserWarning)
def trainTestSubInd(data, classifier):
    start_time = time.time()
    ricciCurvData = pandas.read_csv(data, header=None)
    accuracy = []
    preds = []
    real = []
    currN = 1
    n = str(currN).zfill(3)
    #Prefix obtained: Sxyz where xyz are 0-9
    #Ex: the prefix of all the images of the first distinct subject is S005
    prefix = 'S' + n
    while(currN<=999):
        #Gets all the rows that starts with that prefix
        test = ricciCurvData.loc[ricciCurvData[0].str.startswith(prefix), :]
        if not test.empty:
            #Removes the testing rows from the training set
            train = pandas.concat([ricciCurvData,test]).drop_duplicates(keep=False)
            #Separates the curvature values from the label
            trainLabels = train.iloc[:,-1:]
            train = train.iloc[: , 1:-1]
            testLabels = test.iloc[:,-1:]
            test = test.iloc[: , 1:-1]
            #Training of the classifier
            classifier.fit(train, trainLabels.values.ravel())
            predictions = classifier.predict(test)
            #Saves the accuracy of each subject tested
            accuracy.append(accuracy_score(testLabels, predictions))
            for p in predictions:
                preds.append(p)
            for r in testLabels.values.ravel():
                real.append(r)


        #Gets the next distinct subject
        currN = currN + 1
        n = str(currN).zfill(3)
        prefix = 'S' + n
    
    acc = numpy.array(accuracy)

    print(classification_report(real, preds))
    conf_mat = confusion_matrix(real, preds)
    print(conf_mat)

    #print acc
    print("Standard deviation: " + str(acc.std()))

    print(numpy.round(acc,2))
    print("Running time: %s seconds" % round(time.time() - start_time, 2))

    return conf_mat

#@ignore_warnings(category=UserWarning)
def trainTestAll(data, classifier):
    trainTestKFold(data, classifier)
    trainTestStratKFold(data, classifier)
    trainTestRepeatKFold(data, classifier)
    trainTestLoocv(data, classifier)

#def testImage(data, classifier, image):
