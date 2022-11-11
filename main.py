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
import time
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
svc = SVC(kernel='linear')
dt = DecisionTreeClassifier(criterion='gini', splitter='best')
mlp = MLPClassifier()
da = QuadraticDiscriminantAnalysis()


#training.trainTestAll("cosine.csv", rfc)
#feature_extraction.FCExtract("C:\\Users\\Raffocs\\Desktop\\CK+SI", ["anger", "contempt", "neutral", "disgust", "fear", "happiness", "sadness", "surprise"], "euclidean")
#plot.confusionPlotSbjInd("cosine.csv", rfc)


#feature_extraction.FCExtract("C:\\Users\\Raffocs\\Desktop\\CK+SI", ["anger", "neutral", "contempt", "disgust", "fear", "happiness", "sadness", "surprise"], "euclidean")


#feature_extraction.FCExtract("C:\\Users\\Raffocs\\Desktop\\CK+SI", ["anger", "neutral", "contempt", "disgust", "fear", "happiness", "sadness", "surprise"], "euclidean")
#data_preprocessing.normalizeDf("curvature_values/fc_undersampled_enhanced","euclidean.csv")
#plot.accPartPlot()

#data_preprocessing.normRes("","euclidean.csv")



#start_time = time.time()
#training.trainTestSubInd("euclideanResampled.csv", rfc)
#print("Running time: %s seconds" % round(time.time() - start_time, 2))
#training.trainTestLoocv("curvature_values/fc_undersampled/euclideanNorm.csv", svc)
#conf_mat = plot.confusionPlotSbjInd("curvature_values/fc/euclideanNorm.csv", svc)
#plot.precisionRecall(conf_mat)

