import gym
import time
import tensorflow.compat.v1 as tf
import random
import numpy as np
import math
import matplotlib.pyplot as plt
tf.compat.v1.disable_eager_execution()
# tf.disable_v2_behavior()
from memory import Memory
from model import Model
from walkerGR import GameRunner


def main():
	env_name = 'BipedalWalker-v3'
	env = gym.make(env_name)

	num_states = env.env.observation_space.shape[0]
	print(num_states)
	num_actions = 2

	model = Model(num_states, num_actions, 100)
	mem = Memory(50000)

	with tf.Session() as sess:
		# a = tf.constant(5.0)
		# b = tf.constant(6.0)
		# c = a * b
		sess.run(model._var_init)
		gr = GameRunner(sess, model, env, mem, .3, .1,
						.1)

		num_episodes = 300
		cnt = 0
		while cnt < num_episodes:
			if cnt % 10 == 0:
				print('Episode {} of {}'.format(cnt+1, num_episodes))
			gr.run()
			cnt += 1
		plt.plot(gr._reward_store)
		plt.show()
		plt.close("all")
		plt.plot(gr._max_x_store)
		plt.show()
	

# def main():
# 	env = gym.make("BipedalWalker-v3")
# 	observation = env.reset()
# 	for _ in range(100000):
# 		env.render()
# 		action = env.action_space.sample() # your agent here (this takes random actions)
# 		observation, reward, done, info = env.step(action)

if __name__ == '__main__':
	main()