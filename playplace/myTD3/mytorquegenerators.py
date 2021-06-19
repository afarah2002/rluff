import numpy as np
import random
from sklearn.preprocessing import minmax_scale


class TorqueGenFuncs(object):

	def brute_force_torque_gen(no_terms, new_amps, new_freqs, t, action_dim):
		product = np.outer(new_freqs,t).reshape(action_dim,
												no_terms,
												len(t))
		sine_term = np.sin(product).reshape(no_terms*action_dim, len(t))
		new_amps = new_amps.reshape(no_terms*action_dim)
		A = np.tile(new_amps, len(t)).reshape(len(t), no_terms*action_dim).T
		individ_terms = np.multiply(A, sine_term).reshape(action_dim,
														  no_terms,
														  len(t))
		output = np.sum(individ_terms, axis=1)

		for j in range(action_dim):
			if max(output[j,:]) > 1:
				output[j,:] = minmax_scale(output[j,:], feature_range=(-1,1))


		return output

	def complexity_subgoal_torque_gen(prev_sine_terms, new_amps, new_freqs, t):

		product = np.outer(t, new_freqs)
		sine_term = np.sin(product)
		full_sine_term = np.multiply(new_amps, sine_term)
		output = np.add(prev_sine_terms, full_sine_term)

		for j in range(6):
			if max(output[:,j]) > 1:
				output[:,j] = minmax_scale(output[:,j], feature_range=(-1,1))
		
		return output

