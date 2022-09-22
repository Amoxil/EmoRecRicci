from graph_builder import GraphBuilder
import networkx
import cv2

image = cv2.imread("face.png")
graphBuilder = GraphBuilder()
graph = graphBuilder.buildGraph(image)
print(graph)

nodesPositions = networkx.get_node_attributes(graph,"pos")

graphBuilder.showGraph(image)
