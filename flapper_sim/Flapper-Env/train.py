import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import gym
import time
import os

from queue import *
import itertools
import matplotlib.animation as animation
import multiprocessing
import threading

from stable_baselines3 import TD3, SAC, PPO, DQN
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_util import make_vec_env

import flapper_env

import gui.utils as gui_utils
import gui.framework as gui_framework

class Threads:

	def ai_main(data_classes):
		env = gym.make('Flapper-v0')
		n_actions = env.action_space.shape[-1]
		# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

		# model = PPO("MlpPolicy", env, verbose=1)
		# # model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
		model = TD3("MlpPolicy", env, verbose=1)
		# # model.learn(total_timesteps=25000)
		model.learn(total_timesteps=10000, log_interval=10)
		model.save("flapper-v0.0.01")
		env = model.get_env()

		del model # remove to demonstrate saving and loading

		model = TD3.load("flapper-v0.0.01")
		# model = PPO.load("flapper-v0.0.01")

		env = gym.make('Flapper-v0')
		ob = env.reset()
		t = np.arange(0,2*np.pi,0.001)
		i = 0
		while True:
			# stroke_plane_angle = -30*np.pi/180*np.sin(50*t[i] - 10*np.pi/180)
			# stroke_plane_angle = -5*np.pi/180
			# stroke_plane_angle = np.pi/2
			# left_wing_pos = np.cos(50*(t[i]))*(np.pi/180)*30
			# right_wing_pos = np.cos(50*(t[i]))*(np.pi/180)*30
			# wing_torque = -0*np.pi/180
			i += 1
			if i == len(t):
				i = 0
				t = np.arange(max(t),max(t)+2*np.pi,0.001)
			action, _states = model.predict(obs)
			# action = np.array([stroke_plane_angle, left_wing_pos, right_wing_pos])
			
			obs, rewards, done, _ = env.step(action, data_classes, t[i])
			if done:
				env.reset()
			env.render()

	def ai_main_train(data_classes):
		# Parallel environments
		env = make_vec_env("Flapper-v0", n_envs=1)

		model = PPO("MlpPolicy", env, verbose=1)
		print("Training...")
		model.learn(total_timesteps=250000)
		model.save("ppo_flapper")

		del model # remove to demonstrate saving and loading

		model = PPO.load("ppo_flapper")

		env2 = gym.make('Flapper-v0', GUI=True)
		obs = env2.reset()
		while True:
			print("Displaying trained robot")
			action, _states = model.predict(obs)
			obs, rewards, dones, info = env2.step(action)
			env2.render()	

def main():

	mainQueue = Queue()

	GUI = False

	data_classes = {"Stroke" : gui_utils.GUIDataClass("tab1", "Stroke", int(2+3*2)),
					"AoA" : gui_utils.GUIDataClass("tab2", "AoA", 2),
					"Reward" : gui_utils.GUIDataClass("tab3", "Reward", 1)}
					# "Vectors" : gui_utils.GUIDataClass("tab3", "Vectors", 3)}
	if GUI:
		tab1_figs = [gui_utils.NewMPLFigure(data_classes["Stroke"])]
		tab2_figs = [gui_utils.NewMPLFigure(data_classes["AoA"])]
		tab3_figs = [gui_utils.NewMPLFigure(data_classes["Reward"])]

		gui_figs = [tab1_figs, tab2_figs, tab3_figs]

		for gui_fig_type in gui_figs:
			lines_sets = [fig.lines for fig in gui_fig_type]

		gui_app = gui_framework.GUI(gui_figs)

		anis = [animation.FuncAnimation(fig.figure,
										gui_utils.MPLAnimation.animate,
										interval=10,
										fargs=[fig])
										for fig in list(itertools.chain.from_iterable(gui_figs))]

	ai_main_thread = threading.Thread(target=Threads.ai_main_train,
									  args=(data_classes,))
	ai_main_thread.start()

	if GUI:
		gui_app.mainloop()

if __name__ == '__main__':
	main()