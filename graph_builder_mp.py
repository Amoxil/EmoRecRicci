import mediapipe
import math
import networkx
import cv2
from scipy.spatial import distance
from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci

def buildGraph(image, distType):
    height, width, _ = image.shape
    faceModule = mediapipe.solutions.face_mesh
    processedImage = faceModule.FaceMesh(static_image_mode=True).process(image)
    if(processedImage.multi_face_landmarks is None):
        return
    graph = networkx.Graph()

    #Adds node to the graph
    for faceLandmark in FACE_LANDMARKS:
        #print("have this landmark!!! HYA!!" + landmark)
        landmark =  processedImage.multi_face_landmarks[0].landmark[faceLandmark]
        pos = (int(landmark.x * width), int(landmark.y * height))
        graph.add_node(faceLandmark, pos=pos)

    nodesPosition = networkx.get_node_attributes(graph,"pos")

    #Adds edges to the graph
    for faceEdge in FACE_EDGES:

        match distType:
            case 'manhattan':
                weight = distance.cityblock(nodesPosition[faceEdge[0]], nodesPosition[faceEdge[1]])
            case 'euclidean':
                weight = distance.euclidean(nodesPosition[faceEdge[0]], nodesPosition[faceEdge[1]])
            case 'cosine':
                weight = distance.cosine(nodesPosition[faceEdge[0]], nodesPosition[faceEdge[1]])
            case 'chebyshev':
                weight = distance.chebyshev(nodesPosition[faceEdge[0]], nodesPosition[faceEdge[1]])
            case _:
                    weight = distance.euclidean(nodesPosition[faceEdge[0]], nodesPosition[faceEdge[1]])


        if(weight != 0):
            graph.add_edge(faceEdge[0],faceEdge[1], weight = weight)   
        else:
            return 

    return graph

def buildFormanRicciGraph(image, distType):

    if(image is None):
        return

    graph = buildGraph(image, distType)

    if(graph is None):
        return
    #Computes Forman-Ricci curv
    ricciCurvGraph = FormanRicci(graph)
    ricciCurvGraph.compute_ricci_curvature()
    return ricciCurvGraph

def buildOllivierRicciGraph(image, distType):

    if(image is None):
        return

    graph = buildGraph(image, distType)
    
    if(graph is None):
        return
    #Computes Ollivier-Ricci curv
    ricciCurvGraph = OllivierRicci(graph)
    ricciCurvGraph.compute_ricci_curvature()
    return ricciCurvGraph

def showGraph(image):

    graph = buildGraph(image, "euclidean")

    if(graph is None):
        return
    
    nodesPositions = networkx.get_node_attributes(graph,"pos")

    for faceEdge in FACE_EDGES:
        cv2.line(image, nodesPositions[faceEdge[0]], nodesPositions[faceEdge[1]], (0,0,255), 1)

    for faceLandmark in FACE_LANDMARKS:
        cv2.circle(image, nodesPositions[faceLandmark], 2, (0,0,0))
        cv2.putText(image, str(faceLandmark), nodesPositions[faceLandmark], 0, 0.5, (255,0,0))

    cv2.imshow("image",image)
    cv2.waitKey(0)


FACE_LANDMARKS = frozenset([
    # Lips.
    61, 91, 84, 314, 321, 40, 37, 267, 270, 88, 87, 317, 318, 80, 82, 312, 310, 291,
    # Left eye.
    390, 374, 381, 263, 388, 386, 384, 362,
    # Left eyebrow.
    276, 283, 282, 295, 285, 300, 293, 334, 296, 336,
    # Right eye.
    163, 145, 154, 33, 161, 159, 157, 133,
    # Right eyebrow.
    46, 53, 52, 65, 55, 70, 63, 105, 66, 107,
    # Nose.
    6, 195, 4, 166, 19, 392
])

FACE_EDGES = frozenset([
    # Lips.
    (61, 91),
    (91,84),
    (84,314),
    (314,321),
    (321,291),
    (61, 40),
    (40, 37),
    (37,267),
    (267,270),
    (270,291),
    (61,88),
    (88,87),
    (87,317),
    (317,318),
    (318,291),
    (61,80),
    (80,82),
    (82,312),
    (312,310),
    (310,291),
    # Left eye.
    (263, 390),
    (390,374),
    (374,381),
    (381,362),
    (263,388),
    (388,386),
    (386,384),
    (384,362),
    # Left eyebrow.
    (276, 283),
    (283, 282),
    (282, 295),
    (295, 285),
    (300, 293),
    (293, 334),
    (334, 296),
    (296, 336),
    # Right eye.
    (33,163),
    (163,145),
    (145,154),
    (154,133),
    (33,161),
    (161,159),
    (159,157),
    (157,133),
    # Right eyebrow.
    (46, 53),
    (53, 52),
    (52, 65),
    (65, 55),
    (70, 63),
    (63, 105),
    (105, 66),
    (66, 107),
    # Nose.
    (6,195),
    (195,4),
    (4,19),
    (19,166),
    (19,392)
])




