import cv2
import os
import pandas
import numpy
from sklearn.preprocessing import MinMaxScaler

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

def normalizeDf(df, edges):
    #BE SURE TO REMOVE DATAFRAME HEADER!!! (Or first entry will not be considered)
    images = df.iloc[:, :1]
    norm = df.iloc[: , 1:-1]
    label = df.iloc[:,-1:]


    scaler = MinMaxScaler()
    #print(norm)
    norm = scaler.fit_transform(norm)
    norm = pandas.DataFrame(norm, columns=numpy.arange(len(edges)))
    #print(images)
    result = pandas.concat([images, norm], axis=1, join="inner")
    result = pandas.concat([result, label], axis=1, join="inner")
    #print(result)
    result.to_csv('normalizedData.csv', index=False, header=False)