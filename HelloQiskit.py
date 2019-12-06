import matplotlib.pyplot as plt
from qiskit import *
from qiskit.tools.visualization import plot_histogram, circuit_drawer, plot_bloch_multivector
from qiskit.tools.monitor import job_monitor

#Please, fill this with your own token, which is supposed to be secret
IBMQ.save_account('######', overwrite=True)

circuit = QuantumCircuit(1,1)
circuit.h(0)

circuit_drawer(circuit, filename='circuit.png', output='mpl')

simulator = Aer.get_backend('statevector_simulator')
result = execute(circuit, backend=simulator).result()
statevector = result.get_statevector()
plot_bloch_multivector(statevector)
plt.savefig('Bloch_Sphere.png')
plt.clf()

circuit.measure(0,0)

simulator = Aer.get_backend('qasm_simulator')
result = execute(circuit, backend=simulator).result()
plot_histogram(result.get_counts(circuit))
plt.savefig('Simulator_counts.png')
plt.clf()



#IBMQ.load_account()
#provider = IBMQ.get_provider('ibm-q')
#qcomp = provider.get_backend('ibmq_essex')

#job = execute(circuit, backend=qcomp)
#job_monitor(job)
#result=job.result()
#plot_histogram(result.get_counts(circuit))
#plt.savefig('Essex_counts.png')
#plt.clf()
