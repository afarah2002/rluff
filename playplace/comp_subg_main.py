import numpy as np
import matplotlib.pyplot as plt
import time 
import math
from queue import Queue
from threading import Thread
import random

import pybullet_envs
import pybullet as p
import gym

from pybullet_tests.test2_live_plotter import MyDataClass, MyPlotClass
from pybullet_tests.torque_generators import TorqueGenFuncs

class ComplexitySubgoals:

	def __init__(self):

		#-----------------set general-----------------#
		self.env_name = "HalfCheetahBulletEnv-v0"
		p.connect(p.DIRECT)
		self.env = gym.make(self.env_name)
		self.env.render()
		self.env.reset()

		self.num_states = self.env.observation_space.shape[0]
		self.num_actions = self.env.action_space.shape[0]*2

		self.upper_bound = 1.
		self.lower_bound = -1.

		#-----------------set components-----------------#

		#-----------------set hyperparams-----------------#
		self.total_episodes = 5000

		#-----------------set torque gen params-----------------#

		self.torque_generator = TorqueGenFuncs.torque_generator_1

		self.f_s = 50

		t_0 = np.linspace(0, 2*np.pi, self.f_s, endpoint=False)
		amps_0 = [random.uniform(-1., 1.) for i in range(6)]
		freqs_0 = [random.uniform(0., 10.) for i in range(6)]
		self.torques_0 = np.zeros((len(t_0), 6))
		T_init = self.torque_generator(torques_0, amps_0, freqs_0, t_0)

		self.torques = T_init
		self.prev_torque_term = T_init
		self.new_amps = amps_0
		self.new_freqs = freqs_0
		self.primes = np.concatenate((self.new_amps, self.new_freqs))

		self.term_count = 1 
		self.new_term_factor = 1.
		self.term_reduction_factor = 0.99
		self.term_bound = 0

		self.cycle_counter = 0
		self.max_cycles = 5

		#-----------------set display params-----------------#
		self.data = MyDataClass(6)
		self.plotter = MyPlotClass(self.data)
		self.q = Queue()
		self._sentinel = object()
		

	def producer(self):
		try:
			for ep in range(self.total_episodes):
				prev_state = env.reset()
				episodic_reward = 0
				reward = 0
				self.t = t_0

				while True:
					tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
					for i in range(len(self.t)):
						# update cycle counter at end of t
						if i = len(self.t) - 1:
							self.cycle_counter += 1

						if self.cycle_counter >= self.max_cycles:
							# after the previous f and A values 
							# have been explored, ask for new ones
							self.cycle_counter = 0
							print("\n ASK \n")

							# self.primes = ASK FOR NEW f and A VALUES HERE
							if not math.isnan(float(self.primes[0])):
								self.new_amps = new_term_factor*self.primes[:int(len(self.primes)/2)]
								self.new_freqs = primes[int(len(self.primes)/2):]
							else:
								print("NAAAAAAAAAAAAAN")
								self.new_amps = [random.uniform(-1., 1.) for i in range(6)]
								self.new_freqs =  [random.uniform(0., 10.) for i in range(6)]

							self.torques = self.torque_generator(self.prev_torque_term,
																 self.new_amps,
																 self.new_freqs,
																 self.t)
						torques_set = self.torques[i,:]
						action = torques_set
						self.q.put(self.t[i], action, ep, episodic_reward)
						
						try:
							state, reward, done, info = env.step(action)
						except:
							break

						# record in buffer
						episodic_reward += reward
						# lEarNiNG...........

					if done and self.cycle_counter > 0:
						break

					prev_state = state
					self.t = np.linspace(max(t), max(t) + 2*np.pi, self.f_s, endpoint=False)

				ep_reward_list.append(episodic_reward)
				# Mean of last 40 episodes
				avg_reward = np.mean(ep_reward_list[-40:])

				# if we have mastered the first term, add another one
				if avg_reward > self.term_bound:
					print("NEW TERM")
					self.term_count += 1
					self.torques_0 = torques
					self.term_bound += 100
					self.new_term_factor *= self.term_reduction_factor
					self.prev_torque_term = self.torques

				print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
				avg_reward_list.append(avg_reward)

				###### SAVE MODEL HERE #####
				MODEL_NUM_TAG = 2
				model_loc = "saved_models/actor_" + str(MODEL_NUM_TAG)
				# actor_model.save(model_loc)			


		except KeyboardInterrupt:
			print("stopped")
		pass

	def consumer(self):
		while True:
			for new_data in iter(self.q.get, self._sentinel):
				self.data.XData.append(new_data[0])
				self.data.actions_data.append(new_data[1])

				if len(self.data.XData) >= 100:
					del self.data.XData[0]
					del self.data.actions_data[0]

				if new_data[2] == self.data.episode_numbers[-1]:
					self.data.episodic_reward_data[-1] = new_data[3]
				if new_data[2] != self.data.episode_numbers[-1]:
					self.data.episode_numbers.append(new_data[2])
					self.data.episodic_reward_data.append(new_data[3])

	def run(self):
		t1 = Thread(target=self.consumer)
		t2 = Thread(target=self.producer)
		t1.start()
		t2.start()
		plt.show()
		

if __name__ == '__main__':
	ComplexitySubgoals().run()




