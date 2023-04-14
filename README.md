# Emotion Recognition using Ricci Curvature Descriptor
## Overview
This project aims to explore the feasibility of using Ricci Curvature as a mathematical descriptor for emotion recognition.
The project includes methods to pre-process data, extract features, train with various classifiers, test, and evalue the results. 
The code is written in Python and uses various libraries, such as sklearn, matplotlib, seaborn, pandas, numpy, imblearn, networkx, and mediapipe.

## Data
The dataset used is CK+. 
The images in the dataset are of different sizes and qualities, so some pre-processing is required before they can be used for training.

## Feature Extraction
The main focus of this project is on the use of Ricci Curvature as a mathematical descriptor for emotion recognition. 
The Ricci Curvature is a geometric property that describes the curvature of a surface at every point. 
In this project, the Ricci Curvature is computed on the face image and used as a feature for emotion recognition.

## Classification
Various classifiers are trained on the feature vectors extracted from the pre-processed images. 
The classifiers used in this project includes the majority of models available in the sklearn library such as Support Vector Machines, Random Forest, and K-Nearest Neighbors.

## Evaluation
The performance of the classifiers is evaluated using various metrics such as accuracy, precision, recall, and F1 score.
The evaluation is done using a Leave-one-subject-out cross-validation technique to ensure that the results are not biased.
