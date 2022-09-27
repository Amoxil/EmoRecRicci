from dataclasses import dataclass
from xml.etree.ElementTree import tostring
import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ricciData = pandas.read_csv("data.csv")
df = ricciData.iloc[: , :-1]
labels = ricciData.iloc[:,-1:]

dfTrain, dfTest, labelsTrain, labelsTest = train_test_split(df, labels, test_size=0.1)

model = DecisionTreeClassifier()
model.fit(dfTrain, labelsTrain)
predictions = model.predict(dfTest)
accuracy_score(labelsTest, predictions)
print("Accuracy: " + str(accuracy_score(labelsTest, predictions)*100))
