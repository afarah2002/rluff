import copy
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.stats
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

class PendZeroUtils:

	def convert_action_to_action_pack(action, t):
		# Takes an action and converts it into the dictionary
		# that is passed into the action queue

		wing_torque = action[0]
		# wing_torque = 0

		action_data_pack = {"Time" : t,
							"Wing torques" : [wing_torque]
							}

		return action_data_pack

	def get_next_state_from_combo(combo_data):
		# Separates the state from the combo 
		# into a usable 1D array
		next_state = []
		for state_name, state in combo_data["next state"].items():
			if state_name != "Time" and state_name != "Real time":
				for state_value in state:
					next_state.append(state_value)

		return np.array(next_state)

	def append_data(data_classes, combo_data):
		for data_dir, data_class in data_classes.items():
			tab_name = data_class.tab_name

			new_x = [combo_data[tab_name]["Time"]]
			
			if data_class.data_class_name == "Angular velocity" or data_class.data_class_name == "Wing angles":
				new_y = combo_data[tab_name][data_dir]*data_class.num_lines
			else:
				new_y = combo_data[tab_name][data_dir]

			if new_x not in data_class.XData:
				data_class.XData = np.append(data_class.XData, new_x)
				data_class.YData = np.append(data_class.YData, new_y).reshape(
								   len(data_class.XData),len(new_y))
			else:
				data_class.XData[-1] = new_x[0]
				data_class.YData[-1] = new_y

	def my_distribution(min_val, max_val, mean, std):
		# taken from 
		# https://stackoverflow.com/questions/27831923/python-random-number-generator-with-mean-and-standard-deviation#:~:text=def%20my_distribution(min_val,random.seed(100)
		scale = max_val - min_val
		location = min_val
		# Mean and standard deviation of the unscaled beta distribution
		unscaled_mean = (mean - min_val) / scale
		unscaled_var = (std / scale) ** 2
		# Computation of alpha and beta can be derived from mean and variance formulas
		t = unscaled_mean / (1 - unscaled_mean)
		beta = ((t / unscaled_var) - (t * t) - (2 * t) - 1) / ((t * t * t) + (3 * t * t) + (3 * t) + 1)
		alpha = beta * t
		# Not all parameters may produce a valid distribution
		if alpha <= 0 or beta <= 0:
			raise ValueError('Cannot create distribution for the given parameters.')
		# Make scaled beta distribution with computed parameters
		return scipy.stats.beta(alpha, beta, scale=scale, loc=location)

	def train(model, xdata, ydata, t, epochs=1, sample=256):
		# Random sample from buffers
		indices = np.random.randint(t,size=sample)
		x_sample = np.take(xdata, indices, axis=0)
		y_sample = np.take(ydata, indices, axis=0)

		x_train = tf.constant(x_sample)
		y_train = tf.constant(y_sample)
		model.fit(x_train, y_train, epochs=epochs)

	def train_torch(device, model, xdata, ydata, t, learning_rate=0.0001, epochs=25, sample=512):
		criterion = nn.MSELoss()
		optm = Adam(model.parameters(), lr=learning_rate)
		
		for epoch in range(epochs):
			epoch_loss = 0
			correct = 0

			# Random sample from buffers
			indices = np.random.randint(t,size=sample)
			x_sample = np.take(xdata, indices, axis=0)
			y_sample = np.take(ydata, indices, axis=0)

			x_train = torch.FloatTensor(x_sample).to(device)
			y_train = torch.FloatTensor(y_sample).to(device)

			model.zero_grad()
			predictions = model(x_train)
			loss = criterion(predictions, y_train)
			loss.backward()
			optm.step()

	def reward_function(target_vel, states, actions, real_states_buffer, actions_buffer, t):
		target_angle = 40

		target_theta_max = 40
		target_freq = 4.427

		if np.size(states) == 2: # single state
			thetas = states[0]
			theta_dots = states[1]
			actions = [actions]
		else: # states from all the actors
			thetas = states[:,0]
			theta_dots = states[:,1]

		torques = actions
		history_len = 100
		if t < history_len:
			history_len = t
		else:
			pass

		updated_theta_dots = np.zeros([history_len+1,len(actions)])
		prev_theta_dots = real_states_buffer[t-history_len:t,1] # previous real states are used to get the average theta dot
		
		prev_theta_dots_tiled = np.array([prev_theta_dots for i in range(len(actions))]).reshape(history_len,len(actions))
		updated_theta_dots[0:history_len,:] = prev_theta_dots_tiled
		updated_theta_dots[-1,:] = theta_dots


		updated_torques = np.zeros([history_len+1,len(actions)])
		prev_torques = actions_buffer[t-history_len:t]

		prev_torques_tiled = np.array([prev_torques for i in range(len(actions))]).reshape(history_len,len(actions))
		updated_torques[0:history_len,:] = prev_torques_tiled
		updated_torques[-1,:] = torques

		# calculate average theta_dot over last history_len timesteps
		theta_dots_avg = np.mean(np.abs(updated_theta_dots),axis=0)
		torques_avg = np.mean(np.abs(updated_torques), axis=0)

		R_torque = -10*np.power(np.abs(torques) - 0.1,2)
		# R_torque = 0
		R_theta_dot = -1*np.power((theta_dots_avg/target_vel - 1.0),2)
		R_phase_space = -0*np.power((np.power(thetas/target_theta_max,2) + np.power(theta_dots/(target_freq*target_theta_max),2) - 1),2)

		rewards = (R_torque + R_theta_dot + R_phase_space)
		# rewards = 10*(R_theta_dot)

		if np.size(states)==2:
			print(f"                                                               {R_torque}, {R_theta_dot}, {R_phase_space}")


		return rewards

	def phase_space_reward_function(states, actions, t):
		if np.size(states) == 2: # single state
			thetas = states[0]
			theta_dots = states[1]
			actions = [actions]
		else: # states from all the actors
			thetas = states[:,0]
			theta_dots = states[:,1]

		torques = actions

		target_amp = 30000
		phase_space_amp = np.power(thetas, 2) + np.power(theta_dots, 2)

		R_torques = -10*np.power(torques,2)
		R_phase_space = -10*np.power((phase_space_amp/target_amp - 1),2)
		rewards = R_torques + R_phase_space

		return rewards

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

class NN_torch(nn.Module):

	def __init__(self, dims=[1,1], max_output=1):
		super(NN_torch, self).__init__()

		self.fc1 = nn.Linear(dims[0], 256)
		self.fc2 = nn.Linear(256, 256)
		self.fc3 = nn.Linear(256, dims[1])

		self.max_output = max_output

	def forward(self, state):
		output = F.relu(self.fc1(state))
		output = F.relu(self.fc2(output))
		return self.max_output*torch.tanh(self.fc3(output))