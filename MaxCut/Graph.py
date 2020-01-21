#import math tools
import numpy as np

# We import the tools to handle general Graphs
import networkx as nx

# We import plotting tools
import matplotlib.pyplot as plt
from   matplotlib import cm
from   matplotlib.ticker import LinearLocator, FormatStrFormatter

# importing Qiskit
from qiskit import Aer, IBMQ
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute

from qiskit.providers.ibmq      import least_busy
from qiskit.tools.monitor       import job_monitor
from qiskit.visualization import plot_histogram

from sympy import Symbol, Matrix, init_printing, pprint, sin, cos, simplify, pi, zeros, lambdify, ones
from sympy.physics.quantum import TensorProduct as TP
from sympy.physics.quantum.dagger import Dagger as Dag


class Graph():
    def __init__(self, ):
        self.n = 4
        self.step_size   = 0.1; #Accuracy of grid search

        #Simulation parameters
        self.backend      = Aer.get_backend("qasm_simulator")
        self.shots        = 10000

        self.G = nx.Graph()
        self._Assign()
        print('Assign')
        self._Degree()
        print('Degree')
        self._F()
        print('F')
        self._Evaluate_F()
        print('Eval_F')
        self._Grid()
        print('Grid')
        self._Build()
        print('Build')

    #Assign nodes and edges to the graph
    def _Assign(self):
        self.V = np.arange(0,self.n,1)
        self.E = self._Read_E()

        self.G.add_nodes_from(self.V)
        self.G.add_weighted_edges_from(self.E)

    #Read file with edges
    def _Read_E(self):
        x = np.genfromtxt(r'Edges.txt', delimiter=',')
        return x

    #How many edges of degree 2 and 4 do we have?
    def _Degree(self):
        self.deg2 = 0
        self.deg4 = 0

        for i in self.E:
            k = self.G.degree(i[0]) + self.G.degree(i[1]) - 2
            if k == 2:
                self.deg2 += 1
            elif k == 4:
                self.deg4 += 1

    # Generate plot of the Graph
    def Plot_G(self):
        colors       = ['r' for node in self.G.nodes()]
        default_axes = plt.axes(frameon=True)
        pos          = nx.spring_layout(self.G)

        nx.draw_networkx(self.G, node_color=colors, node_size=600, alpha=1, ax=default_axes, pos=pos)

        plt.savefig('Graph.png')
        plt.clf()


    # Evaluate the function
    def _Evaluate_F(self):

        a_gamma         = np.arange(0, np.pi, self.step_size)
        a_beta          = np.arange(0, np.pi, self.step_size)
        a_gamma, a_beta = np.meshgrid(a_gamma,a_beta)

        a = self.deg2
        b = self.deg4

        #Expectation value
        self.F1 = self.F(a_beta, a_gamma)

        #a*self._fA(a_gamma, a_beta) + b*self._fB(a_gamma, a_beta)


    def _F(self):

        b = Symbol('b')
        g = Symbol('g')

        Id = Matrix([[1,0],
            [0,1]])

        Z = Matrix([[1,0],
            [0,-1]])

        X = Matrix([[0,1],
            [1,0]])

        H = zeros(2**self.n)

        for e in self.E:
            i = 1
            a = Id
            if e[0] == 0 or e[1] == 0:
                c = Z
            else:
                c = Id
            while i < self.n:
                a = TP(a, Id)
                if e[0] == i or e[1] == i:
                    c = TP(c,Z)
                else:
                    c = TP(c,Id)
                i +=1
            H += 1/2*(a-c)

        i = 0
        B = zeros(2**self.n)

        while i < self.n:
            k = 0
            while k < self.n:
                if k == 0 and i == 0:
                    d = X
                elif k == 0 and i != 0:
                    d = Id
                elif k < i or k > i:
                    d = TP(d,Id)
                elif k == i:
                    d = TP(d,X)
                k += 1
            i += 1
            B += d

        Pl = ones(2**self.n, 1)/(2**(self.n/2))

        Eh = (-1j*g*H).exp()
        Ex = (-1j*b*B).exp()

        Psi = Ex * Eh * Pl
        F = (Dag(Psi) * H * Psi)[0,0]

        self.F = lambdify([b,g], F, 'numpy')



    #Expectation value for an edge of degree 2
    def _fA(self, g, b):
        return 1/2*(np.sin(4*g)*np.sin(4*b) + np.sin(2*b)**2*np.sin(2*g)**2)

    #Expectation value for an edge of degree 4
    def _fB(self, g, b):
        return 1/2*(1-np.sin(2*b)**2*np.sin(2*g)**2*np.cos(4*g)**2-1/4*np.sin(4*b)*np.sin(4*g)*(1+np.cos(4*g)**2))

    # Grid search for the minimizing variables
    def _Grid(self):
        result = np.where(self.F1 == np.amax(self.F1))
        a      = list(zip(result[0],result[1]))[0]

        self.gamma  = a[0]*self.step_size;
        self.beta   = a[1]*self.step_size;

    #The smallest paramters and the expectation can be extracted
    def Print_results(self):
        print('\n --- OPTIMAL PARAMETERS --- \n')
        print('The maximal expectation value is:  M1 = %.03f' % np.amax(self.F1))
        print('This is attained for gamma = %.03f and beta = %.03f' % (self.gamma,self.beta))


    #Building the circuit
    def _Build(self):
        # preapre the quantum and classical resisters
        QAOA = QuantumCircuit(len(self.V), len(self.V))

        # apply the layer of Hadamard gates to all qubits
        QAOA.h(range(len(self.V)))
        QAOA.barrier()

        # apply the Ising type gates with angle gamma along the edges in E
        for edge in self.E:
            k = int(edge[0])
            l = int(edge[1])

            QAOA.cu1(-2*self.gamma, k, l)
            QAOA.u1(self.gamma, k)
            QAOA.u1(self.gamma, l)

        # then apply the single qubit X - rotations with angle beta to all qubits
        QAOA.barrier()
        QAOA.rx(2*self.beta, range(len(self.V)))

        # Finally measure the result in the computational basis
        QAOA.barrier()
        QAOA.measure(range(len(self.V)),range(len(self.V)))

        self.QAOA = QAOA

    #Draw circuit
    def Plot_C(self):
        self.QAOA.draw(output='mpl')
        plt.savefig('Circuit.png')
        plt.clf()


    # Compute the value of the cost function
    def cost_function_C(self, x):
        if( len(x) != len(self.V)):
            return np.nan

        C = 0;
        for index in self.E:
            e1 = int(index[0])
            e2 = int(index[1])

            w      = self.G[e1][e2]['weight']
            C = C + w*x[e1]*(1-x[e2]) + w*x[e2]*(1-x[e1])

        return C

    #Simulate the circuit
    def Simulate(self):
        simulate = execute(self.QAOA, backend=self.backend, shots=self.shots)
        QAOA_results = simulate.result()

        plot_histogram(QAOA_results.get_counts(),figsize = (8,6),bar_labels = False)
        plt.xticks(fontsize = 8)

        plt.savefig('Simulator_counts_1.png')
        plt.clf()

        # Evaluate the data from the simulator
        counts = QAOA_results.get_counts()

        avr_C = 0
        max_C = [0,0]
        hist = {}

        for k in range(len(self.G.edges())+1):
            hist[str(k)] = hist.get(str(k),0)

        for sample in list(counts.keys()):

            # use sampled bit string x to compute C(x)
            x         = [int(num) for num in list(sample)]
            tmp_eng   = self.cost_function_C(x)

            # compute the expectation value and energy distribution
            avr_C     = avr_C    + counts[sample]*tmp_eng
            hist[str(int(tmp_eng))] = hist.get(str(round(tmp_eng)),0) + counts[sample]

            # save best bit string
            if( max_C[1] < tmp_eng):
                max_C[0] = sample
                max_C[1] = tmp_eng

        M1_sampled   = avr_C/self.shots

        print('\n --- SIMULATION RESULTS ---\n')
        print('The sampled mean value is M1_sampled = %.02f while the true value is M1 = %.02f \n' % (M1_sampled,np.amax(self.F1)))
        print('The approximate solution is x* = %s with C(x*) = %d \n' % (max_C[0],max_C[1]))
        print('The cost function is distributed as: \n')
        plot_histogram(hist,figsize = (8,6),bar_labels = False)

        plt.savefig('Simulator_counts_2.png')
        plt.clf()
