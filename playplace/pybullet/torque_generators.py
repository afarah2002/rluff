import numpy as np

class TorqueGenFuncs(object):

	def torque_generator(prev_sine_terms, new_amps, new_freqs, t):

		product = np.outer(t, new_freqs)
		sine_term = np.sin(product)
		full_sine_term = np.multiply(new_amps, sine_term)
		output = np.add(prev_sine_terms, full_sine_term)
		
		return output