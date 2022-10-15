from data_preprocessing import processImages
import graph_builder_mp
import feature_extraction_mp
import graph_builder_dlib
import feature_extraction_dlib
import networkx
import cv2
import pandas
import training

#dataFrame = feature_extraction_dlib.extractFeatureFrom("C:\\Users\\Raffocs\\Desktop\\CK+", ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"])

training.trainTestKFold("dataDlib.csv")

#image = cv2.imread("face2.png")
#graph_builder_dlib.showGraph(image)