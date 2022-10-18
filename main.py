import data_preprocessing
import graph_builder_mp
import feature_extraction
import graph_builder_dlib
import feature_extraction
import networkx
import cv2
import pandas
import training
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
"""
rfc = RandomForestClassifier(n_estimators=50, random_state=1, criterion='gini')
knn = KNeighborsClassifier(n_neighbors=10, weights='uniform')
nb = GaussianNB()
svc = SVC(C=1.0, kernel='linear')
dt = DecisionTreeClassifier(criterion='gini', splitter='best')
mlp = MLPClassifier()
da = QuadraticDiscriminantAnalysis()

training.trainTestAll("dataChebyshevMP.csv", rfc)
"""
#feature_extraction.dlibExtract("C:\\Users\\Raffocs\\Desktop\\CK+", ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"], "chebyshev")

