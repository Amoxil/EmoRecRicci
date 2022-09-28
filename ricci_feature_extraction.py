from ctypes import sizeof
import os
import ricci_graph_builder
import pandas
import cv2
import gc
from ricci_graph_builder import FACE_EDGES as edges

def extractFeatureFrom(dir, labels):

    row = []

    #edges are not actually aligned with the data, columns=edges.union(["label"]) is just for sizing purpuses, will get removed in csv
    ricciCurvData = pandas.DataFrame(columns=edges.union(["label"])) 
    i=0

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
                ricciCurvData.loc[image] = row
                row.clear()
            
            i=i+1
           
            if(i > 250): 
                ricciCurvData.to_csv('data.csv', mode='a', header=False,index=False)
                ricciCurvData = ricciCurvData.iloc[0:0]
                i = 0
    
    ricciCurvData.to_csv('data.csv', mode='a', header=False,index=False)

    