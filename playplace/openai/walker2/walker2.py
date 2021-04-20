import gym
import numpy as np
import tensorflow.compat.v1 as tf
import random 
import math

tf.compat.v1.disable_eager_execution()


class Model:

	def __init__(self, num_actions, num_states):
		self._num_actions = num_actions
		self._num_states = num_states

		self._states = None
		self._actions = None

		self._output = None
		self._optimizer = None
		self._var_init = None

		self.train()

	def train(self):
		# train the NN w/ fully connect layers
		self._states = tf.placeholder(tf.float32,[None, self._num_states])
		self._qsa = tf.placeholder(tf.float32, [None, self._num_actions])
		layer1 = tf.layers.dense(inputs=self._states, units=1024, activation=tf.nn.relu)
		layer2 = tf.layers.dense(inputs=layer1, units=1024, activation=tf.nn.relu)
		self._output = tf.layers.dense(inputs=layer2, units=self._num_actions)

		
		loss = tf.losses.mean_squared_error(self._qsa, self._output)
		self._optimizer = tf.train.AdamOptimizer().minimize(loss)
		self._var_init = tf.global_variables_initializer()

		

	def predict_action(self, state, sess):
		print("state = ", state)
		print("states = ", self._states)
		print("output = ", self._output)
		return sess.run(self._output, feed_dict={self._states: state.reshape(1,self._num_states)})
		


class Memory:

	def __init__(self, max_mem):
		self.max_mem = max_mem
		self.state_storage = []
		self.reward_storage = []

	def store_memory(self, state, reward):
		if len(self.state_storage) <= self.max_mem:
			self.state_storage.append(state)
		else:
			self.state_storage.pop(0)
			self.state_storage.append(state)

		if len(self.reward_storage) <= self.max_mem:
			self.reward_storage.append(reward)
		else:
			self.reward_storage.pop(0)
			self.reward_storage.append(state)

	def sample_memory(self, num_samples):

		if num_samples > len(self.state_storage):
			return random.sample(self.state_storage, len(self.state_storage))
		else:
			return random.sample(self.state_storage, num_samples)


class Runner:

	def __init__(self, env, sess, model, memory,
					  num_actions, num_states,
					  max_eps, min_eps):
		self._env = env
		self._sess = sess
		self._model = model
		self._memory = memory

		self._num_actions = num_actions
		self._num_states = num_states

		self._max_eps = max_eps
		self._min_eps = min_eps
		self._eps = self._max_eps

		self._steps = 0


	def run(self):
		state = self._env.reset()
		tot_reward = 0

		while True:
			self._env.render() # render the visual
			action = self._choose_action(state)
			print("\n\n", action, "\n\n")
			next_state, reward, done, info = self._env.step(action)

			if done:
				next_state = None

			self._memory.store_memory((state, action, reward, next_state), reward)
			self._steps += 1
			self._eps = self._min_eps + (self._max_eps - self._min_eps) * \
										math.exp(0.01 * self._steps)

			state = next_state
			tot_reward += reward

			if done:
				break

		# choose action based on NN

		# take action, get data

		# feed data back to NN

	def _choose_action(self, state):
		if random.random() < self._eps:
			return [random.randrange(*sorted([-1,1])) 
					for i in range(self._model._num_actions)]
			# return random.sample(range(-1,1), self._model._num_actions)
		else:
			print("else")
			return self._model.predict_action(state, self._sess)[0]





def main():
	env = gym.make("BipedalWalker-v3")
	state = env.reset()

	num_actions = len(env.action_space.sample())
	num_states = len(env.reset())
	model = Model(num_actions, num_states)
	mem = Memory(50000000)
	max_eps = 0.1
	min_eps = 0.0001

	with tf.Session() as sess:
		while True:
			sess.run(model._var_init)
			GR = Runner(env, sess, model, mem,
						num_actions, num_states,
						max_eps, min_eps)
			GR.run()

		env.close()

if __name__ == '__main__':
	main()