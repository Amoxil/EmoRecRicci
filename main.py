import ricci_graph_builder
import networkx
import cv2

image = cv2.imread("face.png")

graph = ricci_graph_builder.buildGraph(image)
print(graph)

ricci_graph_builder.showGraph(image)

ricciCurvGraph = ricci_graph_builder.buildFormanRicciGraph(image)
print(ricciCurvGraph.G[37][267]["formanCurvature"])

