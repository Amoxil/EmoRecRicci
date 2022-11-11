import time
import cv2
import os
import pandas
import numpy
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

def processImages(dir, labels):
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

    for label in labels:
        path = os.path.join(dir, label)
        print(path)

        for image in os.listdir(path):
            print(image)
            imagePath = os.path.join(path, image)
            currImage = cv2.imread(imagePath)
            if(currImage is not None):
            
                grayImage = cv2.cvtColor(currImage, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(grayImage)
                
                if(len(faces)>1):
                    i=0
                    for (x,y,w,h) in faces:
                        face = grayImage[y:y+h, x:x+w] 
                        resizedImage = cv2.resize(face, (128,128))

                        fileName = os.path.basename(imagePath)
                        dupFileName = os.path.splitext(fileName)[0] + str(i) + ".png"
                        print(dupFileName)
                        print(os.path.join(path, dupFileName))

                        cv2.imwrite(os.path.join(path, dupFileName), resizedImage)
                        i=i+1
                    os.remove(imagePath)
                else:
                    x,y,w,h = faces[0]
                    face = grayImage[y:y+h, x:x+w] 
                    resizedImage = cv2.resize(face, (128,128))
                    image.split()
                    cv2.imwrite(imagePath, resizedImage)

def normalizeDf(path, data):

    loc = os.path.join(path, data)
    
    df = pandas.read_csv(loc, header=None)
    images = df.iloc[:, :1]
    norm = df.iloc[: , 1:-1]
    label = df.iloc[:,-1:]
    cols = len(norm.columns)


    scaler = MinMaxScaler()
    #print(norm)
    norm = scaler.fit_transform(norm)
    norm = pandas.DataFrame(norm, columns=numpy.arange(cols))
    #print(images)
    result = pandas.concat([images, norm], axis=1, join="inner")
    result = pandas.concat([result, label], axis=1, join="inner")
    #print(result)

    normDataLoc = os.path.join(path, data[:-4] + "Norm.csv")
    result.to_csv(normDataLoc, index=False, header=False)
    return data[:-4] + "Norm.csv"

def resampleDf(path, data):
    loc = os.path.join(path, data)
    ricciCurvData = pandas.read_csv(loc, header=None)

    labels = ricciCurvData.iloc[:, -1:]
    values = ricciCurvData.iloc[:,1:-1]
    imageName = ricciCurvData.iloc[:,0:1]
    sm = SMOTE()
    X_res, y_res = sm.fit_resample(values, labels.values.ravel())

    i=0
    while(len(imageName)<len(y_res)):
        imageName.loc[len(imageName.index)] = ['dummy'+str(i)]
        i=i+1
    

    s = pandas.Series(y_res)
    df = pandas.DataFrame({'labels':s.values})

    result = pandas.concat([X_res, df], axis=1, join="inner")
    result = pandas.concat([imageName, result],axis=1, join="inner")
    ricciCurvData = result
    ricciCurvData = ricciCurvData.set_index(0)
    normDataLoc = os.path.join(path, data[:-8] + "Resampled.csv")
    ricciCurvData.to_csv(normDataLoc, header=False)

def normRes(path, data):
    data = normalizeDf(path, data)
    resampleDf(path, data)