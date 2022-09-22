import mediapipe
import math
import cv2
import networkx as nx
from face_landmarks import FACE_EDGES as baseEdges;
from face_landmarks import FACE_LANDMARKS as baseLandmarks;

image = cv2.imread("face.png")
height, width, _ = image.shape
faceModule = mediapipe.solutions.face_mesh
processedImage = faceModule.FaceMesh(static_image_mode=True).process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
graph = nx.Graph()

#Adds node to the graph and draws them on the image
for baseLandmark in baseLandmarks:
    landmark =  processedImage.multi_face_landmarks[0].landmark[baseLandmark]
    pos = (int(landmark.x * width), int(landmark.y * height))
    cv2.circle(image, pos, 2, (0,0,0))
    cv2.putText(image, str(baseLandmark), pos, 0, 0.4, (0,0,0))
    graph.add_node(baseLandmark, pos=pos)

nodesPosition = nx.get_node_attributes(graph,"pos")

#Adds edges to the graph and draws them on the image
for baseEdge in baseEdges:
    graph.add_edge(baseEdge[0],baseEdge[1], weight = math.dist(nodesPosition[baseEdge[0]], nodesPosition[baseEdge[1]]))   
    cv2.line(image, nodesPosition[baseEdge[0]], nodesPosition[baseEdge[1]], (0,0,255), 1)

cv2.imshow("image",image)
print(graph)

cv2.waitKey(0)


   