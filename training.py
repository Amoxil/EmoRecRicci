from dataclasses import dataclass
from operator import index
from xml.etree.ElementTree import tostring
import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ricciCurvData = pandas.read_csv("normalizedDataManhattan.csv")
print(ricciCurvData)
df = ricciCurvData.iloc[: , 1:-1]
labels = ricciCurvData.iloc[:,-1:]

i=0
sum=0

while(i<100):
    dfTrain, dfTest, labelsTrain, labelsTest = train_test_split(df, labels, test_size=0.3)

    model = DecisionTreeClassifier()
    model.fit(dfTrain, labelsTrain)
    predictions = model.predict(dfTest)
    sum = sum + accuracy_score(labelsTest, predictions)
    print("Accuracy: " + str(accuracy_score(labelsTest, predictions)*100))
    i=i+1

print("Average accuracy over 100 tests: " + str(sum))