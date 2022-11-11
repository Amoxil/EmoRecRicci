import os
import shutil
import graph_builder_mp
import graph_builder_dlib
import pandas
import cv2



def mpExtract(dir, labels, distType):

    row = []
    if not os.path.exists(os.path.join(dir, "err")):
        os.makedirs(os.path.join(dir, "err"))

    #Edges are not actually aligned with the data, columns=edges.union(["label"]) is just for sizing purpuses, will get removed in csv
    ricciCurvData = pandas.DataFrame(columns=graph_builder_mp.FACE_EDGES.union(["label"])) 
    i=0

    #Builds path for each label + image
    for label in labels:
        path = os.path.join(dir, label)

        for image in os.listdir(path):

            currImage = cv2.imread(os.path.join(path, image))
            print("Computing: " + image + "...")
            graph = graph_builder_mp.buildFormanRicciGraph(currImage, distType)

            #Appends all edges in a row with the relative label and inserts it in a DataFrame
            if(graph is not None):
                for edge in graph_builder_mp.FACE_EDGES:
                    row.append(graph.G[edge[0]][edge[1]]["formanCurvature"])
                row.append(label)
                ricciCurvData.loc[image] = row
                row.clear()
            else:
                shutil.copy(os.path.join(path, image), os.path.join(dir, "err", image))
            
            i=i+1
           
            if(i > 250): 
                ricciCurvData.to_csv(distType+".csv", mode='a', header=False)
                ricciCurvData = ricciCurvData.iloc[0:0]
                i = 0
    
    ricciCurvData.to_csv(distType+".csv", mode='a', header=False)
    
def dlibExtract(dir, labels, distType):

    row = []
    if not os.path.exists(os.path.join(dir, "err")):
        os.makedirs(os.path.join(dir, "err"))

    #Edges are not actually aligned with the data, columns=edges.union(["label"]) is just for sizing purpuses, will get removed in csv
    ricciCurvData = pandas.DataFrame(columns=graph_builder_dlib.FACE_EDGES.union(["label"])) 
    i=0

    #Builds path for each label + image
    for label in labels:
        path = os.path.join(dir, label)

        for image in os.listdir(path):

            currImage = cv2.imread(os.path.join(path, image))
            print("Computing: " + image + "...")
            graph = graph_builder_dlib.buildFormanRicciGraph(currImage, distType)

            #Appends all edges in a row with the relative label and inserts it in a DataFrame
            if(graph is not None):
                for edge in graph_builder_dlib.FACE_EDGES:
                    row.append(graph.G[edge[0]][edge[1]]["formanCurvature"])
                row.append(label)
                ricciCurvData.loc[image] = row
                row.clear()
            else:
                shutil.copy(os.path.join(path, image), os.path.join(dir, "err", image))
            
            i=i+1
           
            if(i > 250): 
                ricciCurvData.to_csv(distType+".csv", mode='a', header=False)
                ricciCurvData = ricciCurvData.iloc[0:0]
                i = 0
    
    ricciCurvData.to_csv(distType+".csv", mode='a', header=False)
    
def FCExtract(dir, labels, distType):

    row = []
    edges = []
    flm = []

    for fl in graph_builder_mp.FC_FACE_LANDMARKS:
        flm.append(fl)
    
    for i in range(0,len(graph_builder_mp.FC_FACE_LANDMARKS)):
        for j in range(i+1,len(graph_builder_mp.FC_FACE_LANDMARKS)):
            edges.append((flm[i],flm[j]))
        
    cols = edges.copy()
    cols.append('label')
    
    if not os.path.exists(os.path.join(dir, "err")):
        os.makedirs(os.path.join(dir, "err"))

    ricciCurvData = pandas.DataFrame(columns=cols) 
    i=0
    print(cols)

    #Builds path for each label + image
    for label in labels:
        path = os.path.join(dir, label)

        for image in os.listdir(path):

            currImage = cv2.imread(os.path.join(path, image))
            print("Computing: " + image + "...")
            graph = graph_builder_mp.buildFormanRicciFCGraph(currImage, distType)

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
                ricciCurvData.to_csv(distType+".csv", mode='a', header=False)
                ricciCurvData = ricciCurvData.iloc[0:0]
                i = 0
    
    ricciCurvData.to_csv(distType+".csv", mode='a', header=False)
    
    
    