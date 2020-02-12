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
import qiskit.providers.aer.noise as noise

from qiskit.providers.ibmq      import least_busy
from qiskit.tools.monitor       import job_monitor
from qiskit.visualization import plot_histogram

from sympy import Symbol, Matrix, init_printing, pprint, sin, cos, simplify, pi, zeros, lambdify, ones
from sympy.physics.quantum import TensorProduct as TP
from sympy.physics.quantum.dagger import Dagger as Dag

from scipy.optimize import differential_evolution


class Graph():
    def __init__(self, n):
        self.n = n


        #Simulation parameters
        self.backend      = Aer.get_backend("qasm_simulator")
        self.shots        = 10000

        self.G = nx.Graph()
        self._Assign()

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


    # Generate plot of the Graph
    def Plot_G(self):
        colors       = ['r' for node in self.G.nodes()]
        default_axes = plt.axes(frameon=True)
        pos          = nx.spring_layout(self.G)

        nx.draw_networkx(self.G, node_color=colors, node_size=600, alpha=1, ax=default_axes, pos=pos)

        plt.savefig('Graph.png')
        plt.clf()


    #Optimize gamma and beta using the simulator
    def Optimizer(self, p, n, *args):
        bounds = []
        self.noise = n
        self.p = p
        # Error probabilities
        if self.noise:
            self.prob_1 = args[0]  # 1-qubit gate
            self.prob_2 = args[1]   # 2-qubit gate
        else:
            pass

        for i in range(0,2*p):
            bounds.append((0,2*np.pi))
        result = differential_evolution(self._Simulate, bounds)
        print(result.x, -1*result.fun)
        self.F = -1*result.fun


    #Building the circuit
    def _Build(self):
        # preapre the quantum and classical resisters
        QAOA = QuantumCircuit(len(self.V), len(self.V))

        # apply the layer of Hadamard gates to all qubits
        QAOA.h(range(len(self.V)))
        QAOA.barrier()

        for i in range(self.p):
            # apply the Ising type gates with angle gamma along the edges in E
            for edge in self.E:
                k = int(edge[0])
                l = int(edge[1])

                QAOA.cu1(-2*self.gamma[i], k, l)
                QAOA.u1(self.gamma[i], k)
                QAOA.u1(self.gamma[i], l)

                # then apply the single qubit X - rotations with angle beta to all qubits
                QAOA.barrier()
                QAOA.rx(2*self.beta[i], range(len(self.V)))
                QAOA.barrier()

        # Finally measure the result in the computational basis
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
    def _Simulate(self, params):
        self.beta = []
        self.gamma = []
        for i in range(self.p):
            self.beta.append(params[2*i])
            self.gamma.append(params[2*i+1])

        self._Build()

        if self.noise:
            # Depolarizing quantum errors
            error_1 = noise.errors.standard_errors.depolarizing_error(self.prob_1, 1)
            error_2 = noise.errors.standard_errors.depolarizing_error(self.prob_2, 2)

            # Add errors to noise model
            noise_model = noise.NoiseModel()
            noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
            noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
            basis_gates = noise_model.basis_gates

            self._Build()
            simulate = execute(self.QAOA, backend=self.backend, shots=self.shots,
                basis_gates=basis_gates,
                noise_model=noise_model)
        else:
            self._Build()
            simulate = execute(self.QAOA, backend=self.backend, shots=self.shots)
            job_monitor(simulate)

        QAOA_results = simulate.result()
        self.QAOA_results = QAOA_results

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

        self.max_C = max_C
        self.hist = hist
        M1_sampled   = avr_C/self.shots

        return -1*M1_sampled


    def Plot_S(self):
        plot_histogram(self.QAOA_results.get_counts(),figsize = (8,6),bar_labels = False)
        if self.n == 5:
            plt.xticks(fontsize = 10)
        elif self.n >5:
            plt.xticks(fontsize = 8)

        plt.suptitle('Probability to measure each subgraph', fontsize = 20)
        plt.savefig('Simulator_counts_1.png')
        plt.xlabel('Measurement outcome')
        plt.clf()


        print('\n --- SIMULATION RESULTS ---\n')
        print('The sampled mean value is M1_sampled = %.02f' % (self.F))
        print('The approximate solution is x* = %s with C(x*) = %d \n' % (self.max_C[0],self.max_C[1]))
        print('The cost function is distributed as: \n')

        plot_histogram(self.hist,figsize = (8,6),bar_labels = False)
        plt.axvline(x=self.F, color='r')
        plt.xlabel('Number of links', fontsize = 12)
        plt.title(' Links cut by the subgraph', fontsize = 20)

        plt.savefig('Simulator_counts_2.png')
        plt.clf()
