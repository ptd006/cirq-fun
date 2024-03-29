"""
Compare "classical" estimation of a Bernoulli random variable (i.e. simulate lots of times and take average) 
vs quantum phase estimate
Most useful reference is Cirq/examples/phase_estimator.py
"""

import cirq
import matplotlib.pyplot as plt
import numpy as np


# Probability for |1>
p = 0.08

# Angle of rotation to be applied to |0> to give superposition
theta_p = 2*np.arcsin(np.sqrt(p))
print(theta_p)

# Construct the rotation operator (unitary matrix)
rotation = cirq.Ry(theta_p)
print(rotation)


### Setup the quantum circuit to sample 
# grab a qubit
q0 = cirq.GridQubit(0, 0)
print(q0)

# create a circuit that simply applies the rotation and measures the outcome
# note to self: the slick way to gets operators on a qubit is blah.on(qubit), e.g. cirq.Ry(theta_p).on(q0)

circuit = cirq.Circuit()
circuit.append(rotation(q0))
circuit.append(cirq.measure(q0, key='m'))
print(circuit)

# "Classical" estimation of p (i.e. sample lots of times and take average)
def MCSampler(n, circuit, _seed=None): 
	if _seed is not None:
		np.random.seed(_seed)
	
	simulator = cirq.Simulator()
	result = simulator.run(circuit, repetitions=100)
	results = np.ndarray.flatten(result.measurements['m'])
	return np.mean(results)

# Show histogram of results
# The central limit theorem implies it is a normal distribution centred around p.
bernoulli_samples = [ MCSampler(70, circuit, seed=160719)  for i in range(1000) ]
plt.hist(bernoulli_samples,bins=15)
plt.show()


## double check all the linear algebra (used for video)
# it is just a real matrix with determinant 1 (unitary)
cirq.unitary(rotation)
np.linalg.det(cirq.unitary(rotation))

# Eigen decomposition
# eigenvalues should be
(np.exp(-1j* theta_p/2), np.exp(+1j* theta_p/2) )
# Check:
eigen_rotation = np.linalg.eig(cirq.unitary(rotation))

# check abs is 1
abs(eigen_rotation[0])

# finally.. we should get back to theta_p
np.angle(eigen_rotation[0][0])*2, theta_p



###################################
## Quantum approach to estimation

# Inverse QFT copy-pasted from from Cirq/examples/phase_estimator.py
class QftInverse(cirq.Gate):
	"""Quantum gate for the inverse Quantum Fourier Transformation
	"""
	
	def __init__(self, num_qubits):
		super(QftInverse, self)
		self._num_qubits = num_qubits
	
	def num_qubits(self):
		return self._num_qubits
	
	def _decompose_(self, qubits):
		"""A quantum circuit (QFT_inv) with the following structure.

		---H--@-------@--------@----------------------------------------------
			  |       |        |
		------@^-0.5--+--------+---------H--@-------@-------------------------
					  |        |            |       |
		--------------@^-0.25--+------------@^-0.5--+---------H--@------------
							   |                    |            |
		-----------------------@^-0.125-------------@^-0.25------@^-0.5---H---
	
		The number of qubits can be arbitrary.
		"""
	
		qubits = list(qubits)
		while len(qubits) > 0:
			q_head = qubits.pop(0)
			yield cirq.H(q_head)
			for i, qubit in enumerate(qubits):
				yield (cirq.CZ**(-1/2.0**(i+1)))(qubit, q_head)
				


def phase_estimate(unknown_rotation, qnum, repetitions):
	# setup QIFT output and ancilliary qubit
	qubits = [None] * qnum
	for i in range(len(qubits)):
		qubits[i] = cirq.GridQubit(0, i)
	
	ancilla = cirq.GridQubit(0, len(qubits))		
	#print('Got qubits', ancilla, qubits)

	# should actually start from an e-vector in general but we are cheating slightly with knowledge of the platform!


	# reference fo controlled gate
	# https://cirq.readthedocs.io/en/stable/generated/cirq.ControlledGate.html
	# neater notation described in https://quantumcomputing.stackexchange.com/questions/5521/how-to-add-control-to-gates-in-cirq
	# N.b. syntax is controlled_gate.on(control_qubit, target_qubit)
	# The control_qubit is left alone.  If it is |1> the control is applied to target qubit.
    # note the powers here could be computed directly easily as they are just rotations!
	# i.e. we could just replace the above with cirq.Ry(theta_p * (2**i) ) 
	circuit = cirq.Circuit.from_ops(
        cirq.H.on_each(*qubits), # H gate on all qubits		
		cirq.H.on_each(ancilla), # H gate on all qubits
        [cirq.ControlledGate( unknown_rotation ** (2**i) ).on( qubits[qnum-i-1], ancilla) for i in range(qnum)],
        QftInverse(qnum)(*qubits),
        cirq.measure(*qubits, key='phase') # don't bother measuring ancilliary bit
	)
	
	print(circuit)
	simulator = cirq.Simulator()	
	result = simulator.run(circuit, repetitions=repetitions)
	return result 


def to_dec(bin):
	dec_estimate = sum([float(s)*0.5**(order+1) for order, s in enumerate(np.flip(bin,0))])
	if dec_estimate > 0.5:
		dec_estimate = 1 - dec_estimate
	return dec_estimate

def run_phase_estimate(qnum, what, repetitions=100, _seed=None): 
	if _seed is not None:
		np.random.seed(_seed)

	result = phase_estimate( what, qnum, repetitions)
	estimates = [to_dec(estimate_bin) for estimate_bin in result.measurements['phase']]
	theta_estimates = np.array(estimates) * 4 * np.pi
	
	p_estimates = np.sin(theta_estimates/2)**2
	plt.hist(p_estimates,bins=50)
	plt.show()


run_phase_estimate(8, rotation, 100, _seed=160719)



from scipy.stats import binom
100*(1-binom.cdf(3, 7, 0.81))
(2 * np.sqrt(0.1 * 0.9 / 0.01) ) **2
U = np.array( [[np.exp(2*np.pi*1.0j*theta_p), 0], [0, 1]] )
np.linalg.eig(U)

m = cirq.unitary(rotation)
np.dot(m, [1,0])
np.sin(theta_p/2)
