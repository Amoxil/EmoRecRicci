import re
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, LeaveOneOut, train_test_split
import numpy
import pandas
import seaborn as sns
from sklearn.metrics import accuracy_score, plot_precision_recall_curve, classification_report

"""
accuracy  = [81, 81, 80, 80, 80, 80, 72]
precision = [74, 75, 71, 69, 76, 70, 64]
f1        = [75, 75, 73, 72, 73, 72, 64]
"""




def accPartPlot():

    labels = ['Default','Nose', 'Eyes','Left eye + eyebrow', 'Right eye + eyebrow', 'Eyebrows', 'Lips']
    accuracy  = [87, 87, 85, 85, 85, 85, 81]
    precision = [84, 82, 77, 80, 80, 80, 75]
    f1        = [84, 84, 80, 81, 81, 81, 77]
    index = ['Accuracy', "Precision", "F1"]
    df = pandas.DataFrame([accuracy, precision, f1],
                    columns=labels, index=index)

    df = df.transpose()
    
    print(df)
    ax = df.plot.bar()
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1),
            fancybox=True, shadow=True, ncol=5)

    for container in ax.containers:
        ax.bar_label(container)

    ax.tick_params(labelsize=10)
    ax.set_ylabel('Score in %')

    plt.title("FC - Subject independent testing SVM")

    plt.ylim(50, 100)
    plt.show()

def accPlot(csvLoc):
    labels = ['Random Forest', 'Nearest Neighbor', 'Naive Bayes', 'Support Vector', 'Decision Tree', 'Multi-layer Perceptron']
    df = pandas.read_csv(csvLoc, index_col=0)
    print(df)
    ax = df.plot.bar()
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1),
            fancybox=True, shadow=True, ncol=5)

    for container in ax.containers:
        ax.bar_label(container)

    ax.tick_params(labelsize=10)
    ax.set_ylabel('Accuracy in %')

    plt.title("Mediapipe landmark extraction")

    plt.ylim(25, 100)
    plt.show()


def confusionPlot(data, classifier):
    looCV = LeaveOneOut()
    ricciCurvData = pandas.read_csv(data, header=None)
    #print(ricciCurvData)
    df = ricciCurvData.iloc[: , 1:-1]
    labels = ricciCurvData.iloc[:,-1:]

    pred = cross_val_predict(estimator=classifier, X=df, y=labels.values.ravel(), cv=looCV)
    conf_mat = confusion_matrix(labels.values.ravel(), pred)

    return conf_mat
    
def confusionPlotSbjInd(data, classifier):
    ricciCurvData = pandas.read_csv(data, header=None)
    preds = []
    real = []
    currN = 1
    n = str(currN).zfill(3)
    #Prefix obtained: Sxyz where xyz are 0-9
    #Ex: the prefix of all the images of a distinct subject is S005
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
            #Saves prediction and real label
            for p in predictions:
                preds.append(p)
            for r in testLabels.values.ravel():
                real.append(r)

        #Gets the next distinct subject
        currN = currN + 1
        n = str(currN).zfill(3)
        prefix = 'S' + n

    print(classification_report(real, preds))
    conf_mat = confusion_matrix(real, preds)
    print(conf_mat)

    return conf_mat

def precisionRecall(conf_mat, labels=None):
    #outdated, see subjInd testing in traintest module


    if(labels==None):
        labels = ['Anger','Contempt','Disgust','Fear','Happiness','Neutral','Sadness','Surprise']
    sns.heatmap(conf_mat, cmap=sns.color_palette("rocket_r", as_cmap=True), annot=True, fmt='g', xticklabels=labels, yticklabels=labels)
 

    plt.show()


    
    