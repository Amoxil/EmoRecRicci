import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict, LeaveOneOut
import numpy as np
import pandas as pd
import seaborn as sns

"""
labels = ['Random Forest', 'Nearest Neighbor', 'Naive Bayes', 'Support Vector', 'Decision Tree', 'Multi-layer Perceptron']
euclidean = [75.9, 74.7, 66.4, 79.2, 59.9, 80.1]
manhattan = [78.8, 77, 66.7, 80.5, 65.3, 81.3]
cosine = [82, 69.1, 43.4, 79.9, 69.8, 81.6]
chebyshev = [77.7, 75, 70.9, 80, 65.8, 80.3]
index = ['Euclidean', "Manhattan", "Cosine", "Chebyshev"]
df = pd.DataFrame([euclidean, manhattan, cosine, chebyshev],
                  columns=labels, index=index)

df = df.transpose()
print(df)
"""
def accPlot(csvLoc):
    labels = ['Random Forest', 'Nearest Neighbor', 'Naive Bayes', 'Support Vector', 'Decision Tree', 'Multi-layer Perceptron']
    df = pd.read_csv(csvLoc, index_col=0)
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
    ricciCurvData = pd.read_csv(data, header=None)
    #print(ricciCurvData)
    df = ricciCurvData.iloc[: , 1:-1]
    labels = ricciCurvData.iloc[:,-1:]

    pred = cross_val_predict(estimator=classifier, X=df, y=labels.values.ravel(), cv=looCV)
    conf_mat = confusion_matrix(labels.values.ravel(), pred)
    print(conf_mat)
    labels = ['Anger','Contempt','Disgust','Fear','Happiness','Neutral','Sadness','Surprise']
    sns.heatmap(conf_mat, cmap="Greens", annot=True, fmt='g', xticklabels=labels, yticklabels=labels)
    

    plt.show()
    

