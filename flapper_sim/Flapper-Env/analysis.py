import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.stats
import gym
import csv
import time
import os

from queue import *
import itertools
import matplotlib.animation as animation
import multiprocessing
import threading

from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3 import TD3, SAC, PPO, DQN, DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

import flapper_env

def expert_plotter():

	loaded_expert = np.load("/home/nasa01/Documents/UML/willis/rluff/flapper_sim/Flapper-Env/expert_data.npz")
	expert_observations = loaded_expert['expert_observations']
	expert_actions = loaded_expert['expert_actions']

	timesteps = np.arange(0,len(expert_observations))

	pos = expert_observations[:,0:3]
	ori = expert_observations[:,3:7]
	vel = expert_observations[:,7:10]
	ang_vel = expert_observations[:,10:13]
	joint_dyn = expert_observations[:,13:-3]

	# plt.plot(timesteps, joint_dyn[:,4])
	# plt.plot(timesteps, joint_dyn[:,0])
	# plt.plot(timesteps, vel[:,1])
	# plt.plot(timesteps, joint_dyn[:,3])
	plt.plot(timesteps, np.power(ang_vel[:,0],2))
	plt.show()

class TrainedPlotter(object):

	def __init__(self, run):
		self.agent_dir = f"/home/nasa01/Documents/UML/willis/rluff/flapper_sim/Flapper-Env/trained_agents/SAC{run}"
		self.params = np.load(f"{self.agent_dir}/params.npy",allow_pickle='TRUE').item()
		self.converged_data = f"{self.agent_dir}/converged_data.npz"
		print(self.params)
		pass

	def save_converged_data(self):

		# runs the bot and saves the actions and 
		# observations of the converged bot 
		# for a given num of interactions
		save_params = self.params
		save_params["env"]["gui"] = True
		num_interactions = 1000
		env = gym.make('Flapper-v0', params=save_params)

		if isinstance(env.action_space, gym.spaces.Box):
			converged_observations = np.empty((num_interactions,) + env.observation_space.shape)
			converged_actions = np.empty((num_interactions,) + (env.action_space.shape[0],))
		else:
			converged_observations = np.empty((num_interactions,) + env.observation_space.shape)
			converged_actions = np.empty((num_interactions,) + env.action_space.shape)

		model = SAC.load(f"{self.agent_dir}/agent")
		obs = env.reset()
		for timestep in tqdm(range(num_interactions)):
			action, _states = model.predict(obs)
			converged_observations[timestep] = obs
			converged_actions[timestep] = action
			print(obs)
			obs, rewards, done, _ = env.step(action)

		np.savez_compressed(
			f"{self.agent_dir}/converged_data",
			converged_actions=converged_actions,
			converged_observations=converged_observations)

	def plot_converged_data(self):
		if not os.path.exists(self.converged_data):
			# if the converged data has not been saved, run and save it
			print("Saving converged data...")
			self.save_converged_data()

		loaded_converged = np.load(self.converged_data)
		converged_observations = loaded_converged['converged_observations']
		converged_actions = loaded_converged['converged_actions']

		timesteps = np.arange(0,len(converged_observations))

		pos = converged_observations[:,0:3]
		ori = converged_observations[:,3:7]
		vel = converged_observations[:,7:10]
		ang_vel = converged_observations[:,10:13]

		joint_dyn = converged_observations[:,13:-3]
		spm_pos = joint_dyn[:,0]
		wing_pos = joint_dyn[:,[2,5]]
		wing_vel = joint_dyn[:,[3,6]]
		wing_trq = joint_dyn[:,[4,7]]

		plt.plot(timesteps, wing_vel)
		plt.show()

	def progess_plotter(self):
		prog_file = f"{self.agent_dir}/log/progress.csv"
		with open(prog_file) as f:
			reader = csv.reader(f, delimiter=",", quotechar='"')
			# next(reader, None)  # skip the headers
			data_read = [row for row in reader]
		
		data = np.array(data_read).T
		# header = 
		episodes = data[list(data.T[0]).index("time/episodes"),1:].astype(float)
		ep_mean = data[list(data.T[0]).index("rollout/ep_rew_mean"),1:].astype(float)
		# print(data.T[0])
		# print(ep_mean)

		plt.plot(episodes,ep_mean)
		plt.show()

	def display(self):
		disp_params = self.params
		disp_params["env"]["gui"] = True
		# disp_params["env"]["traj reset"] = True

		model = SAC.load(f"{self.agent_dir}/agent")
		env = gym.make('Flapper-v0', params=disp_params)
		obs = env.reset()
		t = 0
		while True:
			# time.sleep(0.5)
			t+=1
			print(t)
			# print("Displaying trained robot")
			action, _states = model.predict(obs)

			obs, rewards, dones, info = env.step(action)
			# print(rewards)
			# env2.render()	

if __name__ == '__main__':
	run = "0001"
	TP = TrainedPlotter(run)
	# TP.progess_plotter()
	# TP.display()
	TP.plot_converged_data()
	# expert_plotter()