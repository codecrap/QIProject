from Graph import Graph

#Modify file Edges.txt to introduce the edges of the graph
G = Graph(9) #The argument is the total number of vertices of the graph
G.Plot_G()
G.Optimizer(2) #The argument for Optimizer is p, the number of layers for the QAOA
G.Plot_C()
G.Plot_S()
