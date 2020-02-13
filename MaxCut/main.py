import numpy as np
import matplotlib.pyplot as plt
from Graph import Graph
import time

# r1 = np.arange(0.001, 0.021, 0.001)
# r2 = np.arange(0.01, 0.21, 0.01)
# M1 = []
# M2 = []

t0 = time.perf_counter()

#Modify file Edges.txt to introduce the edges of the graph
G = Graph(9) #The argument is the total number of vertices of the graph
G.Plot_G()
#The first argument for Optimizer is p, the number of layers for the QAOA
#The second argument for Optimizer is a boolean to use or not noise in our simulation
#The third and fourth arguments are optional and are the error rate of 1 and 2 qubits

G.Optimizer(2, True, 1e-3, 1e-2)

G.Plot_C()
G.Plot_S()

t1 = time.perf_counter()
print("Total execution time: %.2f s" % (t1-t0))

# for i in r1:
#     G.Optimizer(2, True, i, 0.01)
#     M1.append(G.F)
#
# plt.plot(r1,M1)
# plt.xlabel('1-qubit error rate')
# plt.ylabel('Average cost function')
# plt.savefig('Error1.png')
# plt.clf()
#
# for i in r2:
#     G.Optimizer(2, True, 0.001, i)
#     M2.append(G.F)
#
# plt.plot(r2,M2)
# plt.xlabel('2-qubit error rate')
# plt.ylabel('Average cost function')
# plt.savefig('Error2.png')
# plt.clf()
