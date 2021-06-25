import gym
import numpy as np
import tensorflow.compat.v1 as tf
import random 
import math

tf.compat.v1.disable_eager_execution()


class Model:

	def __init__(self, num_actions, num_states, batch_size):
		self._num_actions = num_actions
		self._num_states = num_states
		self._batch_size = batch_size

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
		layer1 = tf.layers.dense(inputs=self._states, units=25, activation=tf.nn.relu)
		layer2 = tf.layers.dense(inputs=layer1, units=25, activation=tf.nn.relu)
		self._output = tf.layers.dense(inputs=layer2, units=self._num_actions)

		
		loss = tf.losses.mean_squared_error(self._qsa, self._output)
		self._optimizer = tf.train.AdamOptimizer().minimize(loss)
		self._var_init = tf.global_variables_initializer()

		

	def predict_action(self, state, sess):
		# print("state = ", state)
		# print("states = ", self._states)
		# print("output = ", self._output)
		return sess.run(self._output, feed_dict={self._states: state.reshape(1,self._num_states)})

	def predict_batch(self, states, sess):
		return sess.run(self._output, feed_dict={self._states: states})

	def train_batch(self, sess, x_batch, y_batch):
		sess.run(self._optimizer, feed_dict={self._states: x_batch, self._qsa: y_batch})
		


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
			if len(self._memory.state_storage) > 1001:
				past_actions = np.array(self._memory.state_storage).T[1]
				dif = np.mean(np.abs(np.subtract(past_actions[-1],past_actions[-1000])))
				# print("dif = ", dif)
				if dif < .1:
					# print("hi")
					action = self._choose_rand_action()
					# break
				else:
					action = self._choose_action(state)					
			# print("action = ", action)
			next_state, reward, done, info = self._env.step(action)
			print(next_state[14], " ", next_state[2])

			height = next_state[14]
			vel_x = next_state[2]
			if vel_x < 0.1:
				reward -= 100
			if height < .3:
				reward -= 10

			# print(reward)
			if done:
				next_state = None

			self._memory.store_memory((state, action, reward, next_state), reward)
			self._replay()
			self._steps += 1
			self._eps = self._min_eps + (self._max_eps - self._min_eps) * \
										math.exp(-0.01 * self._steps)
			# print("eps=  ", self._eps)
			state = next_state
			tot_reward += reward
			# print("total reward = ", tot_reward)

			if done:
				break

		# choose action based on NN

		# take action, get data

		# feed data back to NN

	def _choose_action(self, state):
		if random.random() < self._eps:
			return np.random.uniform(low=-1, high=1, size=self._model._num_actions)
			# return random.sample(range(-1,1), self._model._num_actions)
		else:
			# print("else")
			return self._model.predict_action(state, self._sess)[0]

	def _choose_rand_action(self):
		return np.random.uniform(low=-1, high=1, size=self._model._num_actions)

	def _replay(self):
		batch = self._memory.sample_memory(self._model._batch_size)
		states = np.array([val[0] for val in batch])
		next_states = np.array([(np.zeros(self._model._num_states)
			if val[3] is None else val[3]) for val in batch])

		qsa = self._model.predict_batch(states, self._sess)
		qsad = self._model.predict_batch(next_states, self._sess)
		x = np.zeros((len(batch), self._model._num_states))
		y = np.zeros((len(batch), self._model._num_actions))
		for i, b in enumerate(batch):
			state, action, reward, next_state = b[0], b[1], b[2], b[3]
			# get the current q values for all actions in state
			current_q = qsa[i]
			# update the q value for action
			# if next_state is None:
			# 	# in this case, the game completed after action, so there is no max Q(s',a')
			# 	# prediction possible
			# 	current_q[action] = reward
			# else:
			# 	qsadi_round = np.multiply(10000,qsad[i]).astype(int)
				# current_q[action] = reward + 0.1 * np.max(qsadi_round)
			x[i] = state
			y[i] = current_q
		self._model.train_batch(self._sess, x, y)


def main():
	env = gym.make("BipedalWalker-v3")
	state = env.reset()

	num_actions = len(env.action_space.sample())
	num_states = len(env.reset())
	batch_size = 100
	model = Model(num_actions, num_states, batch_size)
	mem = Memory(500000)
	max_eps = .4
	min_eps = 0.001

	with tf.Session() as sess:
		sess.run(model._var_init)
		GR = Runner(env, sess, model, mem,
					num_actions, num_states,
					max_eps, min_eps)
		while True:

			GR.run()

		env.close()

if __name__ == '__main__':
	main()