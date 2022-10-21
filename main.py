import data_preprocessing
import graph_builder_mp
import feature_extraction
import graph_builder_dlib
import feature_extraction
import networkx
import cv2
import pandas
import training
import plot
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

rfc = RandomForestClassifier(n_estimators=50, random_state=1, criterion='gini')
knn = KNeighborsClassifier(n_neighbors=10, weights='uniform')
nb = GaussianNB()
svc = SVC(C=1.0, kernel='linear')
dt = DecisionTreeClassifier(criterion='gini', splitter='best')
mlp = MLPClassifier()
da = QuadraticDiscriminantAnalysis()

#training.trainTestAll("curvature_values/cur_w_eyebrows_si/cosineDlib.csv", rfc)

#feature_extraction.dlibExtract("C:\\Users\\Raffocs\\Desktop\\CK+SI", ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"], "chebyshev")
#feature_extraction.dlibExtract("C:\\Users\\Raffocs\\Desktop\\CK+SI", ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"], "euclidean")
#feature_extraction.dlibExtract("C:\\Users\\Raffocs\\Desktop\\CK+SI", ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"], "cosine")
#feature_extraction.dlibExtract("C:\\Users\\Raffocs\\Desktop\\CK+SI", ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"], "manhattan")
#plot.confusionPlot("curvature_values/cur_w_eyebrows_si/cosineDlib.csv", rfc)
"""
df = pandas.read_csv("curvature_values/cur_w_eyebrows/cosineDlib.csv", header=None)
df = df.loc[df[0].str.startswith('S005'), :]
print(df)
if df.empty:
    print("im empty")"""

training.trainTestSubInd("curvature_values/cur_w_eyebrows/cosineDlib.csv", rfc)