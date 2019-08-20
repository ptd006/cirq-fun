import cirq
import matplotlib.pyplot as plt
import numpy as np


# probability for |1>
p = 0.125

# angle
theta_p = 2*np.arcsin(np.sqrt(p))
print(theta_p)

# make a rotation operator (matrix)
# (This is denoted A in the papers)
rotation = cirq.Ry(theta_p)
print(rotation)

# grab some qubits
# ONLY n=2 is supported right now in the adder
n = 2
qubits = [None] * n
for i in range(len(qubits)):
	qubits[i] = cirq.GridQubit(0, i)

print(qubits)

#### Simple addition of the 2 qubits ( AND / XOR )

# Idea is explained in https://quantumcomputing.stackexchange.com/questions/1654/how-do-i-add-11-using-a-quantum-computer

circuit_with_addition = cirq.Circuit()
for qubit in qubits:
    circuit_with_addition.append(rotation(qubit))

# First compute the carry bit with a CCNOT - like AND
carry_qubit = cirq.GridQubit(0, 2)
circuit_with_addition.append( cirq.CCX( qubits[0], qubits[1], carry_qubit ) )

# Now CNOT on qubit 2 controlled by qubit 1 - like XOR
circuit_with_addition.append( cirq.CNOT( control=qubits[0], target=qubits[1] ) )

# Measurements
circuit_with_addition.append(cirq.measure(*qubits, key='xor'))
circuit_with_addition.append(cirq.measure(carry_qubit, key='and'))

print(circuit_with_addition)
result = simulator.run(circuit_with_addition, repetitions=5000)
#result

# The pair (carry_bit, qubit[1]) now represents the sum of our 2 "Bernoulli" cubits
low_bits = np.ndarray.flatten(np.array(result.measurements['xor']).T[1]) # each row represents the measurements from a single qubit
high_bits = np.ndarray.flatten(np.array(result.measurements['and']))
bin_sum = low_bits + 2 * high_bits

# check probabilities for each outcome
np.mean(bin_sum == 0),np.mean(bin_sum == 1),np.mean(bin_sum == 2)

# Should be
(1-p)**2, 2*p*(1-p) , p**2

# hist
plt.bar([0,1,2],[sum(bin_sum == 0),sum(bin_sum == 1), sum(bin_sum == 2)] )
plt.xticks([0,1,2], size=10)
plt.ylabel('Number of observations')
plt.show()





############################################################
# Quantum estimation for the probability that the sum is 1
# It amounts to estimating the phase on the carry qubit!

# circuit represented as a unitary matrix
A = circuit_with_addition.to_unitary_matrix()
A



print(circuit_with_addition)
# inverse
# np.linalg.inv(A)

# Recall we need to form an operator of the form 
# A S_0 A^-1 S_phi0



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
				


def phase_estimate(unknown_operator, qnum, repetitions):
	# setup QIFT output and ancilliary qubit
	qubits = [None] * qnum
	for i in range(len(qubits)):
		qubits[i] = cirq.GridQubit(0, i)
	
	ancilla = cirq.GridQubit(0, len(qubits))		
	#print('Got qubits', ancilla, qubits)

	# reference fo controlled gate
	# https://cirq.readthedocs.io/en/stable/generated/cirq.ControlledGate.html
	# neater notation described in https://quantumcomputing.stackexchange.com/questions/5521/how-to-add-control-to-gates-in-cirq
	# N.b. syntax is controlled_gate.on(control_qubit, target_qubit)
	# The control_qubit is left alone.  If it is |1> the control is applied to target qubit.
    # note the powers here could be computed directly easily as they are just rotations!
	# i.e. we could just replace the above with cirq.Ry(theta_p * (2**i) ) 
	circuit = cirq.Circuit.from_ops(
        cirq.H.on_each(*qubits), # H gate on all qubits
        [cirq.ControlledGate( unknown_operator ** (2**i) ).on( qubits[qnum-i-1], ancilla) for i in range(qnum)],
        QftInverse(qnum)(*qubits),
        cirq.measure(*qubits, key='phase') # don't bother measuring ancilliary bit
	)
    # print(circuit)
	simulator = cirq.Simulator()	
	result = simulator.run(circuit, repetitions=repetitions)
	return result 


def to_dec(bin):
	dec_estimate = sum([float(s)*0.5**(order+1) for order, s in enumerate(np.flip(bin,0))])
	if dec_estimate > 0.5:
		dec_estimate = 1 - dec_estimate
	return dec_estimate

def run_phase_estimate(qnum, what, repetitions=100, seed=None): 
	if seed is not None:
		np.random.seed(160719)

	result = phase_estimate( what, qnum, repetitions)
	estimates = [to_dec(estimate_bin) for estimate_bin in result.measurements['phase']]
	theta_estimates = np.array(estimates) * 4 * np.pi
	
	p_estimates = np.sin(theta_estimates/2)**2
	plt.hist(p_estimates,bins=50)
	plt.show()


run_phase_estimate(8, rotation, 1000, seed=160719)

