from distutils.command.build import build
import mediapipe
import math
import networkx
import cv2
from GraphRicciCurvature.FormanRicci import FormanRicci
from face_landmarks import FACE_EDGES as baseEdges;
from face_landmarks import FACE_LANDMARKS as baseLandmarks;

class GraphBuilder:

    def buildGraph(self, image):
        height, width, _ = image.shape
        faceModule = mediapipe.solutions.face_mesh
        processedImage = faceModule.FaceMesh(static_image_mode=True).process(image)
        graph = networkx.Graph()

        #Adds node to the graph
        for baseLandmark in baseLandmarks:
            landmark =  processedImage.multi_face_landmarks[0].landmark[baseLandmark]
            pos = (int(landmark.x * width), int(landmark.y * height))
            graph.add_node(baseLandmark, pos=pos)

        nodesPosition = networkx.get_node_attributes(graph,"pos")

        #Adds edges to the graph
        for baseEdge in baseEdges:
            graph.add_edge(baseEdge[0],baseEdge[1], weight = math.dist(nodesPosition[baseEdge[0]], nodesPosition[baseEdge[1]]))   
        
        return graph

    def computeRicci(self, graph):

        #Computes Ricci curv
        ricciCurvGraph = FormanRicci(graph)
        ricciCurvGraph.compute_ricci_curvature()
        return ricciCurvGraph

    def showGraph(self, image):

        graph = self.buildGraph(image)
        
        nodesPositions = networkx.get_node_attributes(graph,"pos")

        for baseEdge in baseEdges:
            cv2.line(image, nodesPositions[baseEdge[0]], nodesPositions[baseEdge[1]], (0,0,255), 1)

        for baseLandmark in baseLandmarks:
            cv2.circle(image, nodesPositions[baseLandmark], 2, (0,0,0))
            cv2.putText(image, str(baseLandmark), nodesPositions[baseLandmark], 0, 0.5, (255,0,0))

        cv2.imshow("image",image)
        cv2.waitKey(0)


