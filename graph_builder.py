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
        frc = FormanRicci(graph)
        frc.compute_ricci_curvature()
        return graph

    def showGraph(self, image):

        graph = self.buildGraph(image)
        
        nodesPositions = networkx.get_node_attributes(graph,"pos")

        for node in graph.nodes:
            neighbors = graph.neighbors(node)
            if(neighbors!= None):
                for neighbor in neighbors:
                    cv2.line(image,nodesPositions[node],nodesPositions[neighbor],(255,0,0),1)
                    cv2.circle(image, nodesPositions[node], 2, (0,0,0))
                    cv2.putText(image, str(node), nodesPositions[node], 0, 0.4, (0,0,0))

        cv2.imshow("image",image)
        cv2.waitKey(0)


