import copy
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.stats
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from groundstation.ai_utils.pendzero_utils import PendZeroUtils, NN_torch

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

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.raw_data_loc = f"test_data/{self.test_num}_{self.target}/raw_data/"
		os.system("mkdir " + self.raw_data_loc)

		self.run()

	def run(self):

		start = time.time()
		self.env_model = NN_torch(dims=[2+1,2], max_output=1000).to(self.device) # takes in real previous state + action, outputs predicted next state
		self.actor_model = NN_torch(dims=[2,1]).to(self.device) # takes in real previous state, outputs chosen action mu
		mu = 0.0
		std = 0.01
		k = 0.001 # proportionality constant relating std to state diff
		max_action = 0.1
		start_timestep = 1000

		self.n_actors = 25

		self.actions_buffer = np.zeros([self.N_timesteps, 1]) # torque
		self.states_buffer = np.zeros([self.N_timesteps, 2]) # theta theta_dot
		self.real_states_buffer = np.zeros([self.N_timesteps, 2]) # theta theta_dot
		self.avg_theta_dot_buffer = np.zeros([self.N_timesteps, 1]) # avg theta_dot

		self.real_rewards_buffer = np.zeros([self.N_timesteps,1])
		self.std_buffer = np.zeros([self.N_timesteps, 1])

		self.phase_space_amp_buffer = np.zeros([self.N_timesteps, 1])

		for t in range(self.N_timesteps):
			# save data

			# print(self.real_states_buffer[t-1])
			if t < start_timestep:
				# randomly sample actions, no training yet
				action_dist = PendZeroUtils.my_distribution(-max_action, max_action, mu, std) # not actually used here
				actions = action_dist.rvs(size=self.n_actors*10)

				# randomly guess states
				states = np.random.uniform(low=-100*np.pi/180, high=100*np.pi/180, size=(self.n_actors*10,2))

				reward = PendZeroUtils.reward_function(self.target, states, actions, self.real_states_buffer, self.actions_buffer, t)
				# get maximum reward value and its index
				max_reward = np.max(reward)
				max_reward_index = reward.argmax(axis=0)
				# choose corresponding action, predicted state
				# chosen_action = actions[max_reward_index]

				chosen_action = np.random.uniform(-max_action, max_action)
				chosen_state = np.random.uniform(low=-100*np.pi/180, high=100*np.pi/180)

				# Add chosen action/state to buffers
				self.actions_buffer[t,:] = chosen_action
				self.states_buffer[t+1,:] = chosen_state

				# get state from real environment due to chosen action, add to buffer
				real_state, combo_data_pack = self.step(np.array([chosen_action]), t)
				self.real_states_buffer[t+1,:] = real_state
				phase_space_amp = real_state[0]**2 + real_state[1]**2
				self.phase_space_amp_buffer[t+1] = phase_space_amp

				# get reward from real environment, add to buffer
				real_reward = PendZeroUtils.reward_function(self.target, real_state, chosen_action, self.real_states_buffer, self.actions_buffer, t)
				self.real_rewards_buffer[t] = real_reward

				# update combo data pack with reward
				combo_data_pack["reward"] = {"Time" : t, 
											 "Reward" : real_reward}

				self.action_state_combo_queue.put(combo_data_pack)
				print(f"Timestep: {t}         {time.time() - start}")


			if t > 100:
				self.avg_theta_dot_buffer[t] = np.mean(np.abs(self.real_states_buffer[t-20:t,1]))

			if t >= start_timestep:
				# print(std, mu)
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
					env_input = torch.FloatTensor(np.reshape([real_state[0], real_state[1], action],[1,3])).to(self.device)
					state = self.env_model(env_input).cpu().data.numpy().flatten()
					states[actor_index,:] = state
				# print(f"2: {time.time() - start}")

				# returns array of corresponding reward values
				reward = PendZeroUtils.reward_function(self.target, states, actions, self.real_states_buffer, self.actions_buffer, t)
				# get maximum reward value and its index
				max_reward = np.max(reward)
				max_reward_index = reward.argmax(axis=0)

				# choose corresponding action, predicted state
				chosen_action = actions[max_reward_index]
				# chosen_action = np.random.uniform(-max_action, max_action)
				# chosen_action = 0.1*np.sin(t/10)
				chosen_state = states[max_reward_index,:]

				# chosen_state = np.random.uniform(low=-100*np.pi/180, high=100*np.pi/180)

				# Add chosen action/state to buffers
				self.actions_buffer[t,:] = chosen_action
				self.states_buffer[t+1,:] = chosen_state
				
				# get state from real environment due to chosen action, add to buffer
				# print(f"3: {time.time() - start}")
				real_state, combo_data_pack = self.step(np.array([chosen_action]), t)
				# print(f"4: {time.time() - start}")
				self.real_states_buffer[t+1,:] = real_state
				phase_space_amp = real_state[0]**2 + real_state[1]**2
				self.phase_space_amp_buffer[t] = phase_space_amp

				# get reward from real environment, add to buffer
				real_reward = PendZeroUtils.reward_function(self.target, real_state, chosen_action, self.real_states_buffer, self.actions_buffer, t)
				self.real_rewards_buffer[t] = real_reward

				# update combo data pack with reward
				combo_data_pack["reward"] = {"Time" : t, 
											 "Reward" : real_reward}

				# print(f"5: {time.time() - start}")
				self.action_state_combo_queue.put(combo_data_pack)
				# print(f"6: {time.time() - start}")

				# get difference between predicted chosen state and real observed state due to chosen action
				state_diff = real_state - chosen_state

				# get updated mu from actor model
				actor_input = torch.FloatTensor(np.reshape(real_state,[1,2])).to(self.device)
				mu = np.clip(self.actor_model(actor_input).cpu().data.numpy().flatten(), -max_action+0.005, max_action-0.005)	
				# print(f"7: {time.time() - start}")
				# print(f"Mu: {mu},   Actions: {actions},    Chosen action: {chosen_action}")

				# update std with state diff
				# std = k*np.mean(np.abs(state_diff))
				# std = k*np.abs(real_reward)
				self.std_buffer[t] = std

				if t%50 == 0:
					# train the env and the actor at the end of the timestep
					# train env 
					self.save()
					env_xdata = np.concatenate((self.real_states_buffer[0:t,:], self.actions_buffer[0:t,:]), axis=1)
					env_ydata = np.roll(self.real_states_buffer[0:t,:],-1,axis=0)
					PendZeroUtils.train_torch(self.device, self.env_model, env_xdata, env_ydata, t, learning_rate=0.001)

					# train actor
					actor_xdata	= np.roll(self.real_states_buffer[0:t,:],-1,axis=0)
					actor_ydata = self.actions_buffer[0:t,:]
					PendZeroUtils.train_torch(self.device, self.actor_model, actor_xdata, actor_ydata, t, learning_rate=0.001)

				print(f"8: {time.time() - start}")


			print(f"Timestep: {t}, mu: {mu}, std: {std}, Reward: {real_reward},                  Avg theta dot: {self.avg_theta_dot_buffer[t]}    Current theta dot: {self.real_states_buffer[t,1]}")
			# print(f"Timestep: {t}, std: {std}, Reward: {real_reward},                  PS amp: {phase_space_amp}    Current theta dot: {self.real_states_buffer[t,1]}")

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
		# print(d_ang_pos)
		if len(d_ang_pos) > ang_vel_depth:
			ang_vel = np.mean(np.diff(d_ang_pos[-ang_vel_depth:]))/np.mean(np.diff(d_real_time[-ang_vel_depth:]))
			next_state[1] = ang_vel
			# Update combo_data with new ang_vel
			# combo_data_pack["next state"]["Angular velocity"] = [ang_vel]
			# print(f"                                                                                                Angular velocity: {ang_vel}")


		# get the next state from the combo pack
		# return next state, combo_pack
		return next_state, combo_data_pack

	def save(self):
		# save pendzero buffers

		np.save(self.raw_data_loc + "actions_buffer", self.actions_buffer)
		np.save(self.raw_data_loc + "states_buffer", self.states_buffer) 
		np.save(self.raw_data_loc + "real_states_buffer", self.real_states_buffer) 
		np.save(self.raw_data_loc + "avg_theta_dot_buffer", self.avg_theta_dot_buffer) 
		np.save(self.raw_data_loc + "real_rewards_buffer", self.real_rewards_buffer)
		np.save(self.raw_data_loc + "std_buffer", self.std_buffer)
		np.save(self.raw_data_loc + "phase_space_amp_buffer", self.phase_space_amp_buffer)

		# save models/NNs
		env_path = f"models/envs/{self.test_num}"
		torch.save(self.env_model,env_path)


