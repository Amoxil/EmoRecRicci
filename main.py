from data_preprocessing import processImages
import graph_builder_mp
import feature_extraction
import graph_builder_dlib
import feature_extraction
import networkx
import cv2
import pandas
import training
from sklearn.ensemble import RandomForestClassifier

feature_extraction.dlibExtract("C:\\Users\\Raffocs\\Desktop\\CK+", ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"], "chebyshev")
feature_extraction.dlibExtract("C:\\Users\\Raffocs\\Desktop\\CK+", ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"], "euclidean")
feature_extraction.dlibExtract("C:\\Users\\Raffocs\\Desktop\\CK+", ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"], "manhattan")
feature_extraction.dlibExtract("C:\\Users\\Raffocs\\Desktop\\CK+", ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"], "cosine")
#classifier = RandomForestClassifier(n_estimators=50, random_state=1)
#print("Chebyshev")
#training.trainTestAll("dataChebyshev.csv", classifier)


#image = cv2.imread("face2.png")
#graph_builder_dlib.showGraph(image)
