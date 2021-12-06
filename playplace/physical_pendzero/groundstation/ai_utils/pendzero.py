import copy
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.stats
import time


from groundstation.ai_utils.pendzero_utils import PendZeroUtils, NN

class PendZero(object):

	def __init__(self,
				test_num,
				target,
				N_timesteps,
				action_state_combo_queue, 
				action_queue, 
				data_classes, 
				pi_client,
				pc_server):

		self.test_num = test_num
		self.target = target
		self.N_timesteps = int(N_timesteps)
		self.action_state_combo_queue = action_state_combo_queue
		self.action_queue = action_queue
		self.data_classes = data_classes
		self.pi_client = pi_client
		self.pc_server = pc_server

		self.run()

	def run(self):

		self.env_model = NN(dims=[2+1,2]).model # takes in real previous state + action, outputs predicted next state
		self.actor_model = NN(dims=[2,1]).model # takes in real previous state, outputs chosen action mu
		mu = 0.0
		std = 0.01
		k = 0.001 # proportionality constant relating std to state diff
		max_action = 0.1
		start_timestep = 2000

		self.n_actors = 10

		actions_buffer = np.zeros([self.N_timesteps, 1]) # torque
		states_buffer = np.zeros([self.N_timesteps, 2]) # theta theta_dot
		real_states_buffer = np.zeros([self.N_timesteps, 2]) # theta theta_dot
		avg_theta_dot_buffer = np.zeros([self.N_timesteps, 1]) # avg theta_dot

		real_rewards_buffer = np.zeros([self.N_timesteps,1])
		std_buffer = np.zeros([self.N_timesteps,1])

		for t in range(self.N_timesteps):
			print(real_states_buffer[t-1])
			if t < start_timestep:
				# randomly sample actions, no training yet
				action_dist = PendZeroUtils.my_distribution(-max_action, max_action, mu, std) # not actually used here
				actions = action_dist.rvs(size=self.n_actors)

				# randomly guess states
				states = np.random.uniform(low=-100*np.pi/180, high=100*np.pi/180, size=(self.n_actors,2))

				reward = PendZeroUtils.reward_function(self.target, states, actions, real_states_buffer, actions_buffer, t)
				# get maximum reward value and its index
				max_reward = np.max(reward)
				max_reward_index = reward.argmax(axis=0)
				# choose corresponding action, predicted state
				# chosen_action = actions[max_reward_index]

				chosen_action = np.random.uniform(-max_action, max_action)
				chosen_state = np.random.uniform(low=-100*np.pi/180, high=100*np.pi/180)

				# Add chosen action/state to buffers
				actions_buffer[t,:] = chosen_action
				states_buffer[t,:] = chosen_state

				# get state from real environment due to chosen action, add to buffer
				real_state, combo_data_pack = self.step(np.array([chosen_action]), t)
				real_states_buffer[t,:] = real_state

				# get reward from real environment, add to buffer
				real_reward = PendZeroUtils.reward_function(self.target, real_state, chosen_action, real_states_buffer, actions_buffer, t)
				real_rewards_buffer[t] = real_reward

				# update combo data pack with reward
				combo_data_pack["reward"] = {"Time" : t, 
											 "Reward" : real_reward}

				self.action_state_combo_queue.put(combo_data_pack)
				print(f"Timestep: {t} \n")


			if t > 100:
				avg_theta_dot_buffer[t] = np.mean(np.abs(real_states_buffer[t-20:t,1]))

			if t >= start_timestep:
				start = time.time()
				print(std, mu)
				# Generate new distribution from updated mu, std
				try:
					action_dist = PendZeroUtils.my_distribution(-max_action-0.005, max_action+0.005, mu, std)
				except ValueError:
					print("                                         Couldn't generate a new distribution")	

				# sample actions 
				actions = action_dist.rvs(size=self.n_actors)

				# get predicted states from env model
				# print(f"1: {time.time() - start}")
				states = np.zeros([self.n_actors,2])
				for actor_index in range(self.n_actors): # for each actor
					action = actions[actor_index]
					env_input = tf.reshape([real_state[0], real_state[1], action],[1,3])
					state = self.env_model(env_input)
					states[actor_index,:] = state
				# print(f"2: {time.time() - start}")

				# returns array of corresponding reward values
				reward = PendZeroUtils.reward_function(self.target, states, actions, real_states_buffer, actions_buffer, t)
				# get maximum reward value and its index
				max_reward = np.max(reward)
				max_reward_index = reward.argmax(axis=0)

				# choose corresponding action, predicted state
				chosen_action = actions[max_reward_index]

				chosen_state = states[max_reward_index,:]
				# Add chosen action/state to buffers
				actions_buffer[t,:] = chosen_action
				states_buffer[t,:] = chosen_state
				
				# get state from real environment due to chosen action, add to buffer
				# print(f"3: {time.time() - start}")
				real_state, combo_data_pack = self.step(np.array([chosen_action]), t)
				# print(f"4: {time.time() - start}")
				real_states_buffer[t,:] = real_state

				# get reward from real environment, add to buffer
				real_reward = PendZeroUtils.reward_function(self.target, real_state, chosen_action, real_states_buffer, actions_buffer, t)
				real_rewards_buffer[t] = real_reward

				# update combo data pack with reward
				combo_data_pack["reward"] = {"Time" : t, 
											 "Reward" : real_reward}

				# print(f"5: {time.time() - start}")
				self.action_state_combo_queue.put(combo_data_pack)
				# print(f"6: {time.time() - start}")

				# get difference between predicted chosen state and real observed state due to chosen action
				state_diff = real_state - chosen_state

				# get updated mu from actor model
				actor_input = tf.reshape(real_state,[1,2])
				mu = np.clip(self.actor_model(actor_input), -max_action+0.005, max_action-0.005)	
				# print(f"7: {time.time() - start}")

				# update std with state diff
				std = k*np.mean(np.abs(state_diff))
				std_buffer[t] = std

				# train the env and the actor at the end of the timestep
				# train env 
				env_xdata = np.concatenate((real_states_buffer, actions_buffer), axis=1)
				env_ydata = np.roll(real_states_buffer,-1,axis=0)
				PendZeroUtils.train(self.env_model, env_xdata, env_ydata, t)

				# train actor
				actor_xdata	= np.roll(real_states_buffer,-1,axis=0)
				actor_ydata = actions_buffer
				PendZeroUtils.train(self.actor_model, actor_xdata, actor_ydata, t)
				print(f"8: {time.time() - start}")

			print(f"Timestep: {t}, std: {std}, Reward: {real_reward},                  Avg theta dot: {avg_theta_dot_buffer[t]}")

	def step(self, chosen_action, t):

		# convert array to action pack (add to corresponding slots in dictionary)
		action_data_pack = PendZeroUtils.convert_action_to_action_pack(chosen_action, t)

		# add action pack to action queue
		self.action_queue.put(action_data_pack)

		# Receive and read action-state combo pack from pi
		combo_data_pack = self.pi_client.receive_data_pack()

		# compute theta dot
		next_state = PendZeroUtils.get_next_state_from_combo(combo_data_pack)
		ang_vel_depth = 5
		d_ang_pos = np.array(self.data_classes["Wing angles"].YData)[:,0]
		d_real_time = np.array(self.data_classes["Real time"].YData)[:,0]
		if len(d_ang_pos) > ang_vel_depth:
			ang_vel = np.mean(np.diff(d_ang_pos[-ang_vel_depth:]))/np.mean(np.diff(d_real_time[-ang_vel_depth:]))
			next_state[1] = ang_vel
			# Update combo_data with new ang_vel
			combo_data_pack["next state"]["Angular velocity"] = [ang_vel]


		# get the next state from the combo pack
		# return next state, combo_pack
		return next_state, combo_data_pack

