"""
Compare "classical" estimation of a Bernoulli random variable (i.e. simulate lots of times and take average) 
vs quantum phase estimate
Most useful reference is Cirq/examples/phase_estimator.py
"""


import cirq

import matplotlib.pyplot as plt
import numpy as np


# probability for |1>
p = 0.1

# angle
theta_p = 2*np.arcsin(np.sqrt(p))
print(theta_p)

# make a rotation operator (matrix)
# (This is denoted A in the papers)
rotation = cirq.Ry(theta_p)
print(rotation)
# it is just a real matrix with determinant 1 (unitary)
cirq.unitary(rotation)
np.linalg.det(cirq.unitary(rotation))


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
simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=100)
results = np.ndarray.flatten(result.measurements['m'])
sum(results * 1.0) / len(results)

# histogram
plt.bar([0,1],[sum(results*1),sum(1-results)] )
plt.xticks([0,1], size=10)
plt.ylabel('Number of observations')
plt.show()


###################################
## Quantum estimation

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
				


def phase_estimate(what, qnum, repetitions):
	# setup QIFT output and ancilliary qubit
	qubits = [None] * qnum
	for i in range(len(qubits)):
		qubits[i] = cirq.GridQubit(0, i)
	
	ancilla = cirq.GridQubit(0, len(qubits))		
	print('Got qubits', ancilla, qubits)

	# reference fo controlled gate
	# https://cirq.readthedocs.io/en/stable/generated/cirq.ControlledGate.html
	# neater notation described in https://quantumcomputing.stackexchange.com/questions/5521/how-to-add-control-to-gates-in-cirq
	# N.b. syntax is controlled_gate.on(control_qubit, target_qubit)
	# The control_qubit is left alone.  If it is |1> the control is applied to target qubit.
    # note the powers here could be computed directly easily as they are just rotations!
	# i.e. we could just replace the above with cirq.Ry(theta_p * (2**i) ) 
	circuit = cirq.Circuit.from_ops(
        cirq.H.on_each(*qubits), # H gate on all qubits
        [cirq.ControlledGate( what ** (2**i) ).on( qubits[qnum-i-1], ancilla) for i in range(qnum)],
        QftInverse(qnum)(*qubits),
        cirq.measure(*qubits, key='phase') # don't bother measuring ancilliary bit
	)
    # print(circuit)
	simulator = cirq.Simulator()	
	result = simulator.run(circuit, repetitions=repetitions)
	return result 




def run_phase_estimate(qnum, what, repetitions=100):
    """Execute the phase estimator cirquit with multiple settings and
    show results.
    """
    print('Estimation with {}qubits.'.format(qnum))
    print('Estimation (Raw binary)')
    fold_func = lambda ms: ''.join(np.flip(ms, 0).astype(int).astype(str))
    result = phase_estimate( what, qnum, repetitions)
    hist = result.histogram(key='phase', fold_func=fold_func)
    estimate_bin = hist.most_common(1)[0][0]
    estimate = (sum([float(s)*0.5**(order+1) for order, s in enumerate(estimate_bin)]))
    if estimate > 0.5:
        estimate = 1- estimate

    print('{:0.4f} ({})'.format(estimate, estimate_bin))
    
    
    return estimate


pe = run_phase_estimate(8, rotation, 1)
qae = pe * np.pi * 8 / 2
np.sin(qae/2)**2


def run_phase_estimate_v2(qnum, what, repetitions=100):
    fold_func = lambda ms: ''.join(np.flip(ms, 0).astype(int).astype(str))
    result = phase_estimate( what, qnum, repetitions)
    
    def to_dec(bin):
        dec_estimate = sum([float(s)*0.5**(order+1) for order, s in enumerate(np.flip(bin,0))])
        if dec_estimate > 0.5:
            dec_estimate = 1 - dec_estimate
        return dec_estimate
    
    estimates = [to_dec(estimate_bin) for estimate_bin in result.measurements['phase']]

    theta_estimates = np.array(estimates) * np.pi * 8 / 2
    p_estimates = np.sin(theta_estimates/2)**2

    plt.hist(p_estimates,bins=50)
    plt.show()

run_phase_estimate_v2(8, rotation, 100)

