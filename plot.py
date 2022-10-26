import re
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, LeaveOneOut, train_test_split
import numpy
import pandas
import seaborn as sns
from sklearn.metrics import accuracy_score, plot_precision_recall_curve, classification_report

"""
labels = ['Random Forest', 'Nearest Neighbor', 'Naive Bayes', 'Support Vector', 'Decision Tree', 'Multi-layer Perceptron']
euclidean = [75.9, 74.7, 66.4, 79.2, 59.9, 80.1]
manhattan = [78.8, 77, 66.7, 80.5, 65.3, 81.3]
cosine = [82, 69.1, 43.4, 79.9, 69.8, 81.6]
chebyshev = [77.7, 75, 70.9, 80, 65.8, 80.3]
index = ['Euclidean', "Manhattan", "Cosine", "Chebyshev"]
df = pandas.DataFrame([euclidean, manhattan, cosine, chebyshev],
                  columns=labels, index=index)

df = df.transpose()
print(df)
"""
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

    precision=[]
    recall=[]
    tpfp = 0

    if(labels==None):
        labels = ['Anger','Contempt','Disgust','Fear','Happiness','Neutral','Sadness','Surprise']
    sns.heatmap(conf_mat, cmap="Greens", annot=True, fmt='g', xticklabels=labels, yticklabels=labels)
    for i in range(0, len(labels)):
            for j in range(0, len(labels)):
                tpfp =  tpfp + conf_mat[j][i]
            precision.append(conf_mat[i][i]/tpfp)
            tpfp = 0
    
    for i in range(0, len(labels)):
            for j in range(0, len(labels)):
                tpfp =  tpfp + conf_mat[i][j]
            recall.append(conf_mat[i][i]/tpfp)
            tpfp = 0
    
    print(precision)
    print(recall)

    classification_report

    plt.show()


    
    