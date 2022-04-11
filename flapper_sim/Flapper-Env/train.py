import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import gym
import time
import os

from stable_baselines3 import TD3, SAC, PPO
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

import flapper_env

def main():
	env = gym.make('Flapper-v0')
	# n_actions = env.action_space.shape[-1]
	# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

	# model = PPO("MlpPolicy", env, verbose=1)
	# # model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
	# # model = TD3("MlpPolicy", env, verbose=1)
	# # model.learn(total_timesteps=25000)
	# model.learn(total_timesteps=50000, log_interval=10)
	# model.save("flapper-v0.0.01")
	# env = model.get_env()

	# del model # remove to demonstrate saving and loading

	# # model = TD3.load("td3_pendulum")
	# model = PPO.load("flapper-v0.0.01")

	ob = env.reset()
	t = np.arange(0,2*np.pi,0.001)
	c = 0
	i = 0
	while True:
		c += 1
		stroke_plane_angle = -np.pi/2*np.sin(20*t[i])*0
		stroke_plane_angle = -30*np.pi/180
		# stroke_plane_angle = np.pi/2
		wing_torque = 0.6*np.cos(20*(t[i]))*0
		i += 1
		if i == len(t):
			i = 0
		# action, _states = model.predict(obs)
		action = np.array([stroke_plane_angle, wing_torque])
		ob, rewards, done, _ = env.step(action, c)
		# if done:
		# 	env.reset()
		# 	i = 0
		env.render()

if __name__ == '__main__':
	main()