import copy
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.stats
import time

import physics

start = time.time()
# Pendulum characteristics
m = 0.1 # kg
r = 0.3
b = 0.1
dt = 0.01
rk4 = physics.RK4(m, b, r, dt)

'''
PendZero Description

There is a NN building a model of the env, taking in s_r(t), a(t), and outputting 
s_r(t+1) (it learns what actions do to the env). 

At each timestep a continuous action distribution is generated, bounded between 
the min/max actions, with mean mu and standard dev std. 

A sample of size n_actors is taken from the sample. Each perturbs the model of the env, 
and the model predicts the next state due to each action taken. 

The a(t), s(t+1) pair that generates the highest reward based on an explicitly calculated reward function is 
allowed to perturb the real env. The real state from the real env both before and after the 
perturbation is what trains the env model. 

A separate actor NN is trained to take in the current real state s_r(t) and output the 
optimal action a(t). It is supervised to fit the real state s_r(t) to the action from the 
sample that yielded the maximum reward, a_max(t). This way, the actor NN learns to find the 
optimal action because all it trains on are that actions that yielded the highest reward in 
the model env. The output of this NN becomes the mean mu of the action distribution.

As data is collected, the env model is able to train on more data and its predictions 
of the real env's response to a given action becomes more accurate. The difference between 
the env model's prediction of the next state and the actual state observed will decrease
with training time. 

The std of the action distribution is proportional to that difference. As the env model
becomes more accurate, the uncertainty in the next state decreases, so it doesn't need 
to explore actions with each new distribution sample. The exploration in the optimal action
becomes the task of the actor NN, which must find the optimal mu around which to center the
continuous action distribution. 

Available hyperparameters:
	- n_actors - the sample size taken from the distribution at each timestep
	
	- k - the constant of proportionality that scales 
		  the env model's loss to get the actor NN's std dev
	
	- min/max actions - bounds for action distribution

'''

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

np.random.seed(100)

def reward_function(states, actions, real_states_buffer, actions_buffer, t):
	target_vel = 10*np.pi/180 # rad/s

	if np.size(states) == 2: # single state
		thetas = states[0]
		theta_dots = states[1]
		actions = [actions]
	else: # states from all the actors
		thetas = states[:,0]
		theta_dots = states[:,1]

	torques = actions
	history_len = 400
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

	R_torque = -20*np.power(torques_avg,2)
	R_theta_dot = -0.1*np.power((theta_dots_avg/target_vel - 1.0),2)
	rewards = 10*(R_torque + R_theta_dot)

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



env_model = NN(dims=[2+1,2]).model # takes in real previous state + action, outputs predicted next state
actor_model = NN(dims=[2,1]).model # takes in real previous state, outputs chosen action mu

def train(model, xdata, ydata, t, epochs=1, sample=256):
	# Random sample from buffers
	indices = np.random.randint(t,size=sample)
	x_sample = np.take(xdata, indices, axis=0)
	y_sample = np.take(ydata, indices, axis=0)

	x_train = tf.constant(x_sample)
	y_train = tf.constant(y_sample)
	model.fit(x_train, y_train, epochs=epochs)

def main():

	mu = 0.0
	std = 0.01
	k = .1 # proportionality constant relating std to state diff
	max_action = 0.1
	episode_length = 1000 # timesteps per episode
	start_timestep = 2000
	N_timesteps = int(start_timestep+10000)+1

	n_actors = 50

	actions_buffer = np.zeros([N_timesteps, 1]) # torque
	states_buffer = np.zeros([N_timesteps, 2]) # theta theta_dot
	real_states_buffer = np.zeros([N_timesteps, 2]) # theta theta_dot
	avg_theta_dot_buffer = np.zeros([N_timesteps, 1]) # avg theta_dot

	real_rewards_buffer = np.zeros([N_timesteps,1])
	std_buffer = np.zeros([N_timesteps,1])

	dt = 0.01
	# initial state
	time = 0.
	y = np.zeros(2) # starts at 0, no motion yet
	# y = [4., 0.] # starts at 0, no motion yet	
	real_state = y 

	for t in range(N_timesteps):
		if t < start_timestep:
			# randomly sample actions, no training yet
			action_dist = my_distribution(-max_action, max_action, mu, std) # not actually used here
			actions = action_dist.rvs(size=n_actors)
			# actions = np.random.uniform(-max_action, max_action, size=n_actors)
			# actions = max_action*np.random.normal(mu, std, n_actors)
			# randomly guess states
			states = np.random.uniform(low=-100*np.pi/180, high=100*np.pi/180, size=(n_actors,2))

			reward = reward_function(states, actions, real_states_buffer, actions_buffer, t)
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
			time, y = rk4.forward(time, y, chosen_action)
			real_state = y
			real_states_buffer[t,:] = real_state
			# get reward from real environment, add to buffer
			real_reward = reward_function(real_state, chosen_action, real_states_buffer, actions_buffer, t)
			real_rewards_buffer[t] = real_reward
			# get difference between predicted chosen state and real observed state due to chosen action
			state_diff = real_state - chosen_state
			# dont update standard deviation in the beginning, just sample, observe and store 
			# print(state_diff)

		if t > 401:
			avg_theta_dot_buffer[t] = np.mean(np.abs(real_states_buffer[t-400:t,1]))

		if t >= start_timestep:
			print(std, mu)
			# Generate new distribution from updated mu, std
			action_dist = my_distribution(-max_action-0.005, max_action+0.005, mu, std)
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
			reward = reward_function(states, actions, real_states_buffer, actions_buffer, t)
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
			time, y = rk4.forward(time, y, chosen_action)
			real_state = y
			real_states_buffer[t,:] = real_state
			print(real_states_buffer[t,:])
			# get reward from real environment, add to buffer
			real_reward = reward_function(real_state, chosen_action, real_states_buffer, actions_buffer, t)
			real_rewards_buffer[t] = real_reward
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
			train(env_model, env_xdata, env_ydata, t)

			# train actor
			actor_xdata	= np.roll(real_states_buffer,-1,axis=0)
			actor_ydata = actions_buffer
			train(actor_model, actor_xdata, actor_ydata, t)

			if t % 100 == 0:
				# plot kinematic data
				# plt.plot(range(t), std_buffer)
				plt.plot(range(t), real_rewards_buffer[:t], label="rewards")
				plt.plot(range(t), actions_buffer[:t], label="torques")
				plt.plot(range(t), avg_theta_dot_buffer[:t], label="avg theta dot")
				plt.plot(range(t), states_buffer[:t,1], label="predicted theta dot")
				plt.plot(range(t), real_states_buffer[:t,1], label="real theta dot")
				plt.legend()

				# plot action distribution with given mu, std
				# x = np.linspace(-max_action, max_action, n_actors)
				# plt.plot(x, action_dist.pdf(x).flatten())
				plt.show()

if __name__ == '__main__':
	main()