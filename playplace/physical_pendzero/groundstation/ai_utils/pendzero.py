import copy
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.stats
import time

import PendZeroUtils

class NN(object):

	def __init__(self, dims=[1,1]):
		n = 16
		self.model = tf.keras.Sequential(
			[
				tf.keras.layers.Dense(dims[0]),
				tf.keras.layers.LeakyReLU(alpha=0.3),
				tf.keras.layers.Dense(n),
				tf.keras.layers.LeakyReLU(alpha=0.3),
				tf.keras.layers.Dense(n),
				tf.keras.layers.LeakyReLU(alpha=0.3),
				tf.keras.layers.Dense(n),
				tf.keras.layers.LeakyReLU(alpha=0.3),
				tf.keras.layers.Dense(n),
				tf.keras.layers.LeakyReLU(alpha=0.3),
				tf.keras.layers.Dense(dims[1]),
			]
		)

		self.model.compile(optimizer='adam', loss='mse')

class PendZero(object):

	def __init__(test_num,
				target,
				N_timesteps,
				action_state_combo_queue, 
				action_queue, 
				data_classes, 
				pi_client,
				pc_server):

		self.test_num = test_num
		self.target = target
		self.N_timesteps = N_timesteps
		self.action_state_combo_queue = action_state_combo_queue
		self.action_queue = action_queue
		self.data_classes = data_classes
		self.pi_client = pi_client
		self.pc_server = pc_server

		self.run()

	def run(self):
		mu = 0.0
		std = 0.01
		k = 0.1
		max_action = 0.1
		start_timestep = 2000

		n_actors = 50

		actions_buffer = np.zeros([N_timesteps, 1]) # torque
		states_buffer = np.zeros([N_timesteps, 2]) # theta theta_dot
		real_states_buffer = np.zeros([N_timesteps, 2]) # theta theta_dot
		avg_theta_dot_buffer = np.zeros([N_timesteps, 1]) # avg theta_dot

		real_rewards_buffer = np.zeros([N_timesteps,1])
		std_buffer = np.zeros([N_timesteps,1])

		for t in range(self.N_timesteps):
			if t < start_timestep:
				# randomly sample actions, no training yet
				action_dist = PendZeroUtils.my_distribution(-max_action, max_action, mu, std) # not actually used here
				actions = action_dist.rvs(size=n_actors)

				# randomly guess states
				states = np.random.uniform(low=-100*np.pi/180, high=100*np.pi/180, size=(n_actors,2))

				reward = PendZeroUtils.reward_function(states, actions, real_states_buffer, actions_buffer, t)
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
				real_state, combo_data_pack = self.step(chosen_action, t)
				real_states_buffer[t,:] = real_state

				# get reward from real environment, add to buffer
				real_reward = PendZeroUtils.reward_function(real_state, chosen_action, real_states_buffer, actions_buffer, t)
				real_rewards_buffer[t] = real_reward

				# update combo data pack with reward
				combo_data_pack["reward"] = {"Time" : t, 
											 "Reward" : real_reward}

			if t > 401:
				avg_theta_dot_buffer[t] = np.mean(np.abs(real_states_buffer[t-400:t,1]))

			if t >= start_timestep:
				print(std, mu)
				# Generate new distribution from updated mu, std
				action_dist = PendZeroUtils.my_distribution(-max_action-0.005, max_action+0.005, mu, std)
				# sample actions 
				actions = action_dist.rvs(size=n_actors)

				# get predicted states from env model
				states = np.zeros([n_actors,2])
				for actor_index in range(n_actors): # for each actor
					action = actions[actor_index]
					env_input = tf.reshape([real_state[0], real_state[1], action],[1,3])
					state = env_model(env_input)
					states[actor_index,:] = state

				# returns array of corresponding reward values
				reward = PendZeroUtils.reward_function(states, actions, real_states_buffer, actions_buffer, t)
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
				real_state, combo_data_pack = self.step(chosen_action, t)
				real_states_buffer[t,:] = real_state

				# get reward from real environment, add to buffer
				real_reward = PendZeroUtils.reward_function(real_state, chosen_action, real_states_buffer, actions_buffer, t)
				real_rewards_buffer[t] = real_reward

				# update combo data pack with reward
				combo_data_pack["reward"] = {"Time" : t, 
											 "Reward" : real_reward}
											 
				# get difference between predicted chosen state and real observed state due to chosen action
				state_diff = real_state - chosen_state

				# get updated mu from actor model
				actor_input = tf.reshape(real_state,[1,2])
				mu = np.clip(actor_model(actor_input), -max_action+0.005, max_action-0.005)	

				# update std with state diff
				std = k*np.mean(np.abs(state_diff))
				std_buffer[t] = std

				print(f"Timestep: {t}, std: {std}, Reward: {real_reward}")

				# train the env and the actor at the end of the timestep
				# train env 
				env_xdata = np.concatenate((real_states_buffer, actions_buffer), axis=1)
				env_ydata = np.roll(real_states_buffer,-1,axis=0)
				PendZeroUtils.train(env_model, env_xdata, env_ydata, t)

				# train actor
				actor_xdata	= np.roll(real_states_buffer,-1,axis=0)
				actor_ydata = actions_buffer
				PendZeroUtils.train(actor_model, actor_xdata, actor_ydata, t)

	def step(self, t, chosen_action):

		# convert array to action pack (add to corresponding slots in dictionary)
		action_data_pack = PendZeroUtils.convert_action_to_action_pack(chosen_action, t)

		# add action pack to action queue
		self.action_queue.put(action_data_pack)

		# Receive and read action-state combo pack from pi
		combo_data_pack = self.pi_client.receive_data_pack()

		# compute theta dot
		ang_vel_depth = 5
		d_ang_pos = np.array(self.data_classes["Wing angles"].YData)[:,0]
		d_real_time = np.array(self.data_classes["Real time"].YData)[:,0]
		if len(d_ang_pos) > ang_vel_depth:
			ang_vel = np.mean(np.diff(d_ang_pos[-ang_vel_depth:]))/np.mean(np.diff(d_real_time[-ang_vel_depth:]))
			# Update combo_data with new ang_vel
			combo_data_pack["next state"]["Angular velocity"] = [ang_vel]


		# get the next state from the combo pack
		next_state = PendZeroUtils.get_next_state_from_combo(combo_data_pack)

		# return next state, combo_pack
		return next_state, combo_data_pack

