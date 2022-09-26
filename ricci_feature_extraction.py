from ctypes import sizeof
import os
import ricci_graph_builder
import pandas
import cv2
from ricci_graph_builder import FACE_EDGES as edges

def extractFeatureFrom(dir, labels):

    #labels are not actually aligned with the data, columns=edges.union(["label"]) is just for sizing purpuses, will get removed in csv
    ricciData = pandas.DataFrame(columns=edges.union(["label"])) 
    row = []

    for label in labels:
        path = os.path.join(dir, label)
        for image in os.listdir(path):
            currImage = cv2.imread(os.path.join(path, image))
            print("Computing: " + image + "...")
            graph = ricci_graph_builder.buildFormanRicciGraph(currImage)
            if(graph is not None):
                for edge in edges:
                    row.append(graph.G[edge[0]][edge[1]]["formanCurvature"])
                row.append(label)
                ricciData.loc[image] = row
                row.clear()
            #print(ricciData)
    
    ricciData.to_csv("data.csv", header=False,index=False)
    return ricciData
    
    