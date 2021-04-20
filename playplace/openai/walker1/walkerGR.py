import gym
import time
import tensorflow.compat.v1 as tf
import random
import numpy as np
import math
import matplotlib.pyplot as plt

class GameRunner:
	# this is where everything the robot does (model dynamics, actions selection, NN training)
	# is organized
	def __init__(self, sess, model, env, memory, max_eps, min_eps, 
				 decay, render=True):
		# initialize a bunch of internal variables

		self._sess = sess # This is a tensorflow session object

		self._env = env # This is an OpenAI gym environment

		self._model = model # This is a NN model
		
		self._memory = memory # This is the memory class instance (2nd class we wrote)

		self._render = render # True/False means game is displayed/not displayed

		self._max_eps = max_eps #---
		self._min_eps = min_eps #   |--- these determine the greediness of the agent during training
		self._decay = decay     #---		(will it choose exploit possibly short-term rewards or explore
								# 				possibly long term rewards? how often will it do each?)
		self._eps = self._max_eps
		self._steps = 0
		self._reward_store = []
		self._max_x_store = []

	def run(self):
		state = self._env.reset() # reset the environment and get a new state

		tot_reward = 0 # initial values to be used in the following loop
		max_x = -100

		while True:
			if self._render:
				self._env.render() # render the env if that's what we want

			action = self._choose_action(state) # see this later, in the runner class
			print(action, "\n\n\n\n")
			# act, = pf(state[None])
			next_state, reward, done, info = self._env.step(action) # run the action using the built-in openai function
			'''
			# next_state = the new state of the agent
			# reward = the reward received by taking "action"
			# done = tells us if the game has finished
			# info = additional info, optional use
			'''
			#-----------------rewards different states different amounts, optional-----------------#
			if next_state[0] >= 0.1:
				reward += 10
			elif next_state[0] >= 0.25:
				reward += 20
			elif next_state[0]:
				reward += 100

			if next_state[0] > max_x:
				max_x = next_state[0]
				# update max_x with next_state value if necessary

			# is the game finished? if so, set the next state to none
			if done:
				next_state = None

			self._memory.add_sample((state, action, reward, next_state))
			self._replay()
			# exponentially decay the eps value
			self._steps += 1
			self._eps = self._min_eps + (self._max_eps - \
								self._min_eps) * math.exp(-.1 * self._steps)
			# move agent to next state and accumulate the reward
			state = next_state
			tot_reward += reward

			# if the game is done, break the loop
			if done:
				self._reward_store.append(tot_reward)
				self._max_x_store.append(max_x)
				break

		print("Step {}, Total reward: {}, Eps: {}".format(self._steps, tot_reward, self._eps))

	def _choose_action(self, state):
		# exectute an epsilon-greedy + Q policy
		if random.random() < self._eps:
			return random.randint(0, self._model._num_actions - 1)
		else:
			return np.argmax(self._model.predict_one(state, self._sess))

	def _replay(self):
		# take a batch of memories to be used in training
		batch = self._memory.sample(self._model._batch_size)
		states = np.array([val[0] for val in batch])
		next_states = np.array([(np.zeros(self._model._num_states)
									if val[3] is None else val[3]) for val in batch])

		# predict Q(s,a) given the batch of states
		q_s_a = self._model.predict_batch(states, self._sess)
		# predict Q(s',a') - so that we can do a gamma * max(Q(s'a')) below
		q_s_a_d = self._model.predict_batch(next_states, self._sess)
		# setup empty training arrays
		x = np.zeros((len(batch), self._model._num_states))
		y = np.zeros((len(batch), self._model._num_actions))

		for i, b in enumerate(batch):
			state, action, reward, next_state = b[0], b[1], b[2], b[3]
			# get the current q values for all actions in the state
			current_q = q_s_a[i]
			# update the q value for action
			if next_state is None:
				# in this case, the game completed completed after action, 
				# so there is no max Q(s',a') prediction possible
				current_q[action] = reward
			else:
				current_q[action] = reward + .1 * np.amax(q_s_a_d[i])

			x[i] = state
			y[i] = current_q
		self._model.train_batch(self._sess, x, y)
