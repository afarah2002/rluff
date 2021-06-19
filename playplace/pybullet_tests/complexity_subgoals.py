import gym 
import pybullet_envs
import pybullet as p
import time
import math
from queue import Queue
from threading import Thread
from sklearn.preprocessing import minmax_scale

from test2_live_plotter import MyDataClass, MyPlotClass

# it is different from how MuJoCo renders environments
# it doesn't differ too much to me w/ and w/o mode='human'
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random


_sentinel = object()

p.connect(p.DIRECT)

# check the link below for some common environments
# https://github.com/bulletphysics/bullet3/releases

# env_name = "BipedalWalker-v3"
# env_name = 'AntBulletEnv-v0'
env_name = "HalfCheetahBulletEnv-v0"
# env_name = "Walker2DBulletEnv-v0"
# env_name = "HopperBulletEnv-v0"
# env_name = "MinitaurBulletEnv-v0"
# env_name = "InvertedPendulumBulletEnv-v0"


env = gym.make(env_name)

env.render()
env.reset()

num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]*2

# upper_bound = 2*np.pi
# lower_bound = -2*np.pi

upper_bound = 1.
lower_bound = -1.

class OUActionNoise:
	'''
	Better exploration by the actor
	'''

	def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
		self.theta = theta
		self.mean = mean
		self.std_dev = std_deviation
		self.dt = dt
		self.x_initial = x_initial
		self.reset()

	def __call__(self):
		# Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
		x = (
			self.x_prev
			+ self.theta * (self.mean - self.x_prev) * self.dt
			+ self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
		)
		# Store x into x_prev
		# Makes next noise dependent on current one
		self.x_prev = x
		return x

	def reset(self):
		if self.x_initial is not None:
			self.x_prev = self.x_initial
		else:
			self.x_prev = np.zeros_like(self.mean)


class Buffer:
	def __init__(self, buffer_capacity=100000, batch_size=999):
		# Number of "experiences" to store at max
		self.buffer_capacity = buffer_capacity
		# Num of tuples to train on.
		self.batch_size = batch_size

		# Its tells us num of times record() was called.
		self.buffer_counter = 0

		# Instead of list of tuples as the exp.replay concept go
		# We use different np.arrays for each tuple element
		self.state_buffer = np.zeros((self.buffer_capacity, num_states))
		self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
		self.reward_buffer = np.zeros((self.buffer_capacity, 1))
		self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

	# Takes (s,a,r,s') obervation tuple as input
	def record(self, obs_tuple):
		# Set index to zero if buffer_capacity is exceeded,
		# replacing old records
		index = self.buffer_counter % self.buffer_capacity

		self.state_buffer[index] = obs_tuple[0]
		self.action_buffer[index] = obs_tuple[1]
		self.reward_buffer[index] = obs_tuple[2]
		self.next_state_buffer[index] = obs_tuple[3]

		self.buffer_counter += 1

	# Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
	# TensorFlow to build a static graph out of the logic and computations in our function.
	# This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
	@tf.function
	def update(
		self, state_batch, action_batch, reward_batch, next_state_batch,
	):
		# Training and updating Actor & Critic networks.
		# See Pseudo Code.
		with tf.GradientTape() as tape:
			target_actions = target_actor(next_state_batch, training=True)
			y = reward_batch + gamma * target_critic(
				[next_state_batch, target_actions], training=True
			)
			critic_value = critic_model([state_batch, action_batch], training=True)
			critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

		critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
		critic_optimizer.apply_gradients(
			zip(critic_grad, critic_model.trainable_variables)
		)

		with tf.GradientTape() as tape:
			actions = actor_model(state_batch, training=True)
			critic_value = critic_model([state_batch, actions], training=True)
			# Used `-value` as we want to maximize the value given
			# by the critic for our actions
			actor_loss = -tf.math.reduce_mean(critic_value)

		actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
		actor_optimizer.apply_gradients(
			zip(actor_grad, actor_model.trainable_variables)
		)

	# We compute the loss and update parameters
	def learn(self):
		# Get sampling range
		record_range = min(self.buffer_counter, self.buffer_capacity)
		# Randomly sample indices
		batch_indices = np.random.choice(record_range, self.batch_size)

		# Convert to tensors
		state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
		action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
		reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
		reward_batch = tf.cast(reward_batch, dtype=tf.float32)
		next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

		self.update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
	for (a, b) in zip(target_weights, weights):
		a.assign(b * tau + a * (1 - tau))


def get_actor():
	# Initialize weights between -3e-3 and 3-e3
	last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

	inputs = layers.Input(shape=(num_states,))
	out = layers.Dense(256, activation="relu")(inputs)
	out = layers.Dense(256, activation="relu")(out)
	outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(out)

	# Our upper bound is 2.0 for Pendulum.
	outputs = outputs * upper_bound
	model = tf.keras.Model(inputs, outputs)
	return model


def get_critic():
	# State as input
	state_input = layers.Input(shape=(num_states))
	state_out = layers.Dense(16, activation="relu")(state_input)
	state_out = layers.Dense(32, activation="relu")(state_out)

	# Action as input
	action_input = layers.Input(shape=(num_actions))
	action_out = layers.Dense(32, activation="relu")(action_input)

	# Both are passed through seperate layer before concatenating
	concat = layers.Concatenate()([state_out, action_out])

	out = layers.Dense(256, activation="relu")(concat)
	out = layers.Dense(256, activation="relu")(out)
	outputs = layers.Dense(num_actions)(out)

	# Outputs single value for give state-action
	model = tf.keras.Model([state_input, action_input], outputs)

	return model

def policy(state, noise_object):
	sampled_actions = tf.squeeze(actor_model(state))
	noise = noise_object()
	# Adding noise to action
	sampled_actions = sampled_actions.numpy()

	# We make sure action is within bounds
	legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

	return list(np.squeeze(legal_action))

def torque_generator(prev_sine_terms, new_amps, new_freqs, t):

	product = np.outer(t, new_freqs)
	sine_term = np.sin(product)
	full_sine_term = np.multiply(new_amps, sine_term)
	output = np.add(prev_sine_terms, full_sine_term)
	# normalize between -1 and 1
	for j in range(6):
		if max(output[:,j]) > 1:
			output[:,j] = minmax_scale(output[:,j], feature_range=(-1,1))

	return output		

std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.1
actor_lr = 0.01

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)


total_episodes = 5000
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.05

buffer = Buffer(1000000, 64)

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

all_amps_store = []
all_freqs_store = []

data = MyDataClass(int(num_actions/2))

def producer(out_q):
	f_s = 50
	t_0 = np.linspace(0, 2*np.pi, f_s, endpoint=False)
	t = t_0
	amps_0 = [random.uniform(-1., 1.) for i in range(int(num_actions/2))]
	freqs_0 = [random.uniform(0., 10.) for i in range(int(num_actions/2))]
	torques_0 = np.zeros((len(t), int(num_actions/2)))
	T_init = torque_generator(torques_0, amps_0, freqs_0, t)


	print(T_init.shape)

	torques = T_init
	prev_torque_term = T_init
	new_amps = amps_0
	new_freqs = freqs_0
	primes = np.concatenate((new_amps, new_freqs))

	term_count = 1
	new_term_factor = 1.
	term_reduction_factor = .9
	term_bound = -200

	cycle_counter = 0
	max_cycles = 5

	# fig = plt.figure()
	try:
		for ep in range(total_episodes):

			prev_state = env.reset()
			episodic_reward = 0
			reward = 0
			t = t_0

			while True:

				tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

				# set Torques in action space to be initial torque
				# run torques as actions until the end
				for i in range(len(t)):
					# env.render()
					# print(cycle_counter)
					# print(t[i])
					if i == len(t) - 1:
						cycle_counter += 1

					if cycle_counter >= max_cycles:
						# after the previous f and A values 
						# have been explored, ask for new ones
						cycle_counter = 0
						# done = True
						print(" \n ASK \n")
						primes = np.abs(policy(tf_prev_state, ou_noise))

						if not math.isnan(float(primes[0])):
							new_amps = new_term_factor*primes[:int(len(primes)/2)] 
							new_freqs = primes[int(len(primes)/2):]
						else:
							print("NAAAAAAAAAAAAAAAAAAAAAAAAAAAAAANN")
							new_amps = [random.uniform(-1., 1.) for i in range(int(num_actions/2))]
							new_freqs = [random.uniform(0., 10.) for i in range(int(num_actions/2))]

						torques = torque_generator(prev_torque_term, new_amps, new_freqs, t)

					torques_set = torques[i,:]
					# print("Term bound: ", term_bound, "   Term count: ", term_count)
					# print(cycle_counter)
					# t = np.linspace(t[int(f_s/2)], t[int(f_s/2)] + 2*np.pi, f_s, endpoint=False)

					# print(episodic_reward, " torques = ", torques_set)
					action = torques_set
					# print(action)

					# data.actions_data.append(action)
					# data.XData.append(t[i])

					out_q.put((t[i], action, ep, episodic_reward))
					# print(ep, "		", episodic_reward)
					# print(np.array(data.actions_data).T)


					# Recieve state and reward from environment.
					try:
						state, reward, done, info = env.step(action)
					except:
						break

					buffer.record((prev_state, primes, reward, state))
					episodic_reward += reward
					# print(torques_set)

					buffer.learn()

					update_target(target_actor.variables, actor_model.variables, tau)
					update_target(target_critic.variables, critic_model.variables, tau)

					# print("ep reward = ", episodic_reward, \
					# 	  "bound = ", np.mean(ep_reward_list[-10:]), \
					# 	  "prev terms: ", torques_0[4,0])

				# End this episode when `done` is True
				if done and cycle_counter >= 1:
					break

				prev_state = state
				t = np.linspace(max(t), max(t) + 2*np.pi, f_s, endpoint=False)
			

			ep_reward_list.append(episodic_reward)
			# Mean of last 40 episodes
			avg_reward = np.mean(ep_reward_list[-40:])

			# if we have mastered the first term, add another one
			if avg_reward > term_bound:
				print("NEW TERM")
				term_count += 1
				torques_0 = torques
				term_bound += 100
				new_term_factor *= term_reduction_factor
				prev_torque_term = torques

			print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
			avg_reward_list.append(avg_reward)

			###### SAVE MODEL HERE #####
			MODEL_NUM_TAG = 2
			model_loc = "saved_models/actor_" + str(MODEL_NUM_TAG)
			# actor_model.save(model_loc)

	except KeyboardInterrupt:
		print("stopped")


def consumer(in_q):
	while True:
		for new_data in iter(in_q.get, _sentinel):
			data.XData.append(new_data[0])
			data.actions_data.append(new_data[1])

			if len(data.XData) >= 10:
				del data.XData[0]
				del data.actions_data[0]

			if new_data[2] == data.episode_numbers[-1]:
				data.episodic_reward_data[-1] = new_data[3]
			if new_data[2] != data.episode_numbers[-1]:
				data.episode_numbers.append(new_data[2])
				data.episodic_reward_data.append(new_data[3])
			

plotter = MyPlotClass(data)
if __name__ == '__main__':
	q = Queue()
	t1 = Thread(target=consumer, args=(q,))
	t2 = Thread(target=producer, args=(q,))
	t1.start()
	t2.start()
	plt.show()
