'''
Walker3 improves upon walker2
 - Goals
	 - Batch training
	 - More NN layers
	 - Better epsilon stuff (idk yet!)
	 - evolutionary?

'''

import gym
import numpy as np
import tensorflow.compat.v1 as tf
import random 
import math
import time

tf.compat.v1.disable_eager_execution()


class Agent:

	def __init__(self, num_actions, num_states):
		self.num_actions = num_actions
		self.num_states = num_states

		self.max_mem = 50000
		self.memory = []

		self.gamma = 0.95    # discount rate
		self.epsilon = 1.0  # exploration rate
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.001

		self._var_init = None

		self.model = self._build_model()

	def _build_model(self):
		self.states = tf.placeholder(tf.float32,[None, self.num_states])
		self.qsa = tf.placeholder(tf.float32,[None, self.num_actions])

		layer1 = tf.layers.dense(inputs=self.states, units=25, activation=tf.nn.relu)
		layer2 = tf.layers.dense(inputs=layer1, units=25, activation=tf.nn.relu)

		self.output = tf.layers.dense(inputs=layer2,units=self.num_actions)

		loss = tf.losses.mean_squared_error(self.qsa, self.output)
		self.optimizer = tf.train.AdamOptimizer().minimize(loss)
		self._var_init = tf.global_variables_initializer()

	def predict_one_action(self, state, sess):
		return sess.run(self.output, feed_dict={
										self.states: 
										state.reshape(1,self.num_states)})[0]

	def train_batch(self):
		pass

	def predict_batch(self):
		pass

	# memory

	def store_memory(self, data):
		self.memory.append(data)
		pass

	def act(self, state, sess):
		if random.random() < self.epsilon:
			return np.random.uniform(low=-1, high=1, size=self.num_actions)
		action_options = self.predict_one_action(state, sess)
		# print("actions = ", action_options)
		return action_options

	# def replay(self, batch_size, sess):
	# 	minibatch = random.sample(self.memory, batch_size)
	# 	for state, action, reward, next_state, done in minibatch:
	# 		target = reward
	# 		if not done:
	# 			target = (reward + self.gamma * 
	# 					  self.model.predict_action(state, self.sess)[0])
	# 		target_f = self.model.predict_action(state, sess)
	# 		target_f[0][action]
	# 	pass



if __name__ == '__main__':
	env = gym.make("BipedalWalker-v3")

	num_actions = len(env.action_space.sample())
	num_states = len(env.reset())

	agent = Agent(num_actions, num_states)
	batch_size = 32

	EPISODES = 1000

	with tf.Session() as sess:
		sess.run(agent._var_init)
		while True:
			state = env.reset()
			t0 = time.time()
			while True:
				env.render()

				action = agent.act(state, sess)
				next_state, reward, done, _ = env.step(action)

				elapsed_time = time.time() - t0
				t0 = time.time()
				vel_x = next_state[2]
				dist = vel_x*elapsed_time

				print("distance travelled = ", dist)
				if dist < 0.0001:
					reward -= 10
				# height = next_state[14]
				# print(vel_x)
				# if vel_x < 0.05:
				# 	reward -= 100
				# if height < .3:
				# 	break


				data = (state, action, reward, next_state, done)
				agent.store_memory(data)
				state = next_state

				if done:
					print("\nDONE\n")
					break

			# if len(agent.memory) > batch_size:
			# 	agent.replay(batch_size, sess)