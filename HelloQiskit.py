import matplotlib.pyplot as plt
from qiskit import *
from qiskit.tools.visualization import plot_histogram, circuit_drawer
from qiskit.tools.monitor import job_monitor

#Please, fill this with your own token, which is supposed to be secret
IBMQ.save_account('######', overwrite=True)

qr = QuantumRegister(2)
cr = ClassicalRegister(2)
circuit = QuantumCircuit(qr,cr)
circuit.h(qr[0])
circuit.measure(qr,cr)
circuit_drawer(circuit, filename='circuit.png', output='mpl')

simulator = Aer.get_backend('qasm_simulator')
result = execute(circuit, backend=simulator).result()
plot_histogram(result.get_counts(circuit))
plt.savefig('Simulator_counts.png')
plt.clf()

IBMQ.load_account()
provider = IBMQ.get_provider('ibm-q')
qcomp = provider.get_backend('ibmq_essex')

job = execute(circuit, backend=qcomp)
job_monitor(job)
result=job.result()
plot_histogram(result.get_counts(circuit))
plt.savefig('Essex_counts.png')
plt.clf()
