import matplotlib.pyplot as plt
from qiskit import *
from qiskit.tools.visualization import plot_histogram, circuit_drawer, plot_bloch_multivector
from qiskit.tools.monitor import job_monitor


circuit = QuantumCircuit(2)

circuit.h(0)
circuit.rz(0.5, 0)

circuit.rzz(1.5, 0, 1)

circuit_drawer(circuit, filename='circuit.png', output='mpl')
