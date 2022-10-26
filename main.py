from statistics import LinearRegression
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

rfc = RandomForestClassifier(n_estimators=100, random_state=1, criterion='gini')
knn = KNeighborsClassifier(n_neighbors=10, weights='uniform')
nb = GaussianNB()
svc = SVC(C=1.0,kernel='linear')
dt = DecisionTreeClassifier(criterion='gini', splitter='best')
mlp = MLPClassifier()
da = QuadraticDiscriminantAnalysis()


#training.trainTestAll("cosine.csv", rfc)

#feature_extraction.dlibExtract("C:\\Users\\Raffocs\\Desktop\\CK+", ["anger", "contempt", "neutral", "disgust", "fear", "happiness", "sadness", "surprise"], "cosine")
#plot.confusionPlotSbjInd("cosine.csv", rfc)

"""
df = pandas.read_csv("curvature_values/cur_w_eyebrows/cosineDlib.csv", header=None)
df = df.loc[df[0].str.startswith('S005'), :]
print(df)
if df.empty:
    print("im empty")

image = cv2.imread("face.png")
graph_builder_dlib.showGraph(image)

flm = []
edges = []

for fl in graph_builder_mp.FC_FACE_LANDMARKS:
    flm.append(fl)

for i in range(0,len(graph_builder_mp.FC_FACE_LANDMARKS)):
    for j in range(i+1,len(graph_builder_mp.FC_FACE_LANDMARKS)):
        edges.append((flm[i],flm[j]))

data_preprocessing.normalizeDf("manhattan.csv", edges)

feature_extraction.FCExtract("C:\\Users\\Raffocs\\Desktop\\CK+", ["anger", "neutral", "contempt", "disgust", "fear", "happiness", "sadness", "surprise"], "euclidean")
"""

#training.trainTestSubInd("curvature_values/fc/chebyshevNorm.csv", svc) #0.87
training.trainTestSubInd("curvature_values/fc/euclideanNorm.csv", svc) #0.87
#training.trainTestSubInd("curvature_values/fc/manhattanNorm.csv", svc) 0.86
#training.trainTestSubInd("curvature_values/fc/manhattanNorm.csv", mlp) 0.86, best AVG(0.59,0.84)

#conf_mat = plot.confusionPlotSbjInd("curvature_values/fc/euclideanNorm.csv", svc)
#plot.precisionRecall(conf_mat)

