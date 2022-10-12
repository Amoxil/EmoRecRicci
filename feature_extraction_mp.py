from ctypes import sizeof
import os
import shutil
import graph_builder_mp
import pandas
import cv2
from graph_builder_mp import FACE_EDGES as edges

def extractFeatureFrom(dir, labels):

    row = []
    if not os.path.exists(os.path.join(dir, "err")):
        os.makedirs(os.path.join(dir, "err"))

    #Edges are not actually aligned with the data, columns=edges.union(["label"]) is just for sizing purpuses, will get removed in csv
    ricciCurvData = pandas.DataFrame(columns=edges.union(["label"])) 
    i=0

    #Builds path for each label + image
    for label in labels:
        path = os.path.join(dir, label)

        for image in os.listdir(path):

            currImage = cv2.imread(os.path.join(path, image))
            print("Computing: " + image + "...")
            graph = graph_builder_mp.buildFormanRicciGraph(currImage)

            #Appends all edges in a row with the relative label and inserts it in a DataFrame
            if(graph is not None):
                for edge in edges:
                    row.append(graph.G[edge[0]][edge[1]]["formanCurvature"])
                row.append(label)
                ricciCurvData.loc[image] = row
                row.clear()
            else:
                shutil.copy(os.path.join(path, image), os.path.join(dir, "err", image))
            
            i=i+1
           
            if(i > 250): 
                ricciCurvData.to_csv('data.csv', mode='a', header=False)
                ricciCurvData = ricciCurvData.iloc[0:0]
                i = 0
    
    ricciCurvData.to_csv('data.csv', mode='a', header=False)
    

    