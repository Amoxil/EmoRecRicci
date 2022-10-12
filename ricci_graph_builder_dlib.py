import dlib
import math
import networkx
import cv2
from scipy.spatial import distance
from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci

import cv2
import dlib

img = cv2.imread("face.png")

hog_face_detector = dlib.get_frontal_face_detector()

dlib_facelandmark = dlib.shape_predictor("ASM/shape_predictor_68_face_landmarks.dat")


height, width, _ = img.shape

faces = hog_face_detector(img)
for face in faces:

    face_landmarks = dlib_facelandmark(img, face)

    for n in range(0, 68):
        x = face_landmarks.part(n).x
        y = face_landmarks.part(n).y
        cv2.circle(img, (x, y), 1, (0, 0, 255), 1)


cv2.imshow("image", img)
cv2.waitKey(0)