import gym
import csv
import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.fftpack import rfft, irfft, fftfreq, fft
import matplotlib as mpl
from pathlib import Path
import scipy.stats
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

plt.rcParams.update({
	"text.usetex": True,
	"font.family": "serif",
	"font.serif": ["Palatino"],
	"font.size" : 18})

# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = "Times New Roman"
# mpl.rcParams.update({'font.size': 24})
# mpl.rc('text', usetex=False)

class TrainedPlotter(object):

	def __init__(self, run):
		self.agent_dir = f"/home/nasa01/Documents/UML/willis/rluff/flapper_sim/Flapper-Env/trained_agents/SAC{run}"
		self.params = np.load(f"{self.agent_dir}/params.npy",allow_pickle='TRUE').item()
		self.converged_data = f"{self.agent_dir}/converged_data.npz"
		print(self.params)
		pass

	def expert_plotter(self):

		loaded_expert = np.load("/home/nasa01/Documents/UML/willis/rluff/flapper_sim/Flapper-Env/expert_data.npz")
		expert_observations = loaded_expert['expert_observations']
		expert_actions = loaded_expert['expert_actions']

		timesteps = np.arange(0,len(expert_observations))

		pos = expert_observations[:,0:3]
		ori = expert_observations[:,3:7]
		vel = expert_observations[:,7:10]
		ang_vel = expert_observations[:,10:13]
		joint_dyn = expert_observations[:,13:-3]
		spm_pos = joint_dyn[:,0]
		wing_pos = joint_dyn[:,[2,5]]
		wing_vel = joint_dyn[:,[3,6]]
		wing_trq = joint_dyn[:,[4,7]]

		# plt.plot(timesteps, joint_dyn[:,4])
		# plt.plot(timesteps, joint_dyn[:,0])
		# plt.plot(timesteps, vel[:,1])
		# plt.plot(timesteps, joint_dyn[:,3])
		# plt.plot(timesteps, np.power(ang_vel[:,0],2))
		# plt.plot(timesteps, wing_trq)
		self.fft_fitter(timesteps[500:1500], wing_trq[500:1500,0])
		plt.show()

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

		timesteps = np.arange(0,len(converged_observations))*0.01 # seconds

		pos = converged_observations[:,0:3]
		ori = converged_observations[:,3:7]
		vel = converged_observations[:,7:10]
		ang_vel = converged_observations[:,10:13]

		joint_dyn = converged_observations[:,13:-3]
		spm_pos = joint_dyn[:,0]
		wing_pos = joint_dyn[:,[2,5]]
		wing_vel = joint_dyn[:,[3,6]]
		wing_trq = joint_dyn[:,[4,7]]

		# f1, axarr1 = plt.subplots(1, 2, figsize=(10, 5))

		# # plt.plot(timesteps, pos)

		# ax0 = axarr1[0]
		# ax1 = axarr1[1]

		# par23 = ax0.twinx()		
		# p01, = ax0.plot(timesteps, 
		# 					  converged_actions[:,0]*180/np.pi,
		# 					  label=r"$\phi$")
		# p02, = par23.plot(timesteps, 
		# 				  converged_actions[:,1]*180/np.pi,
		# 				  label=r"$\Delta\theta_1$")
		# p03, = par23.plot(timesteps, 
		# 				  converged_actions[:,2]*180/np.pi,
		# 				  label=r"$\Delta\theta_2$")
		# ax0.legend()
		# # ax0.axis["left"].label.set_color(p01.get_color())
		# # par23.axis["right"].label.set_color(p023.get_color())
		
		# ax1.plot(timesteps, spm_pos*180/np.pi)
		# ax1.plot(timesteps, wing_pos*180/np.pi)


		# f, axarr = plt.subplots(1, 1, figsize=(10, 5))

		# axarr.plot(timesteps, vel[:,1], linewidth=3)
		# axarr.set_xlabel("Time [s]")
		# axarr.set_ylabel(r'$\dot{y}$ [m]')
		# axarr.set_title("Forward y velocity vs time")
		# axarr.grid()

		# episodes, ep_mean = self.progess_plotter()
		# axarr[1].plot(episodes, ep_mean)
		# axarr[1].set_xlabel("Mean reward")
		# axarr[1].set_ylabel("Episode")

		# for i, label in enumerate(('(a)', '(b)')):
		# 	axarr[i].text(0.5, 1.05, label, transform=axarr[i].transAxes,
		# 	  			  fontweight='bold', va='top', ha='right')
		# 	axarr[i].grid()

		# x = timesteps
		# y = wing_trq
		# self.fft_fitter(x[:500], y[:500])

		# plt.draw()
		plt.plot(wing_pos[:100,1], wing_vel[:100,1])
		plt.show()

	def fft_fitter(self, x, y):

		fit_x = x[0:300]
		fit_y = y[0:300,0] 

		N = len(fit_x)
		T = fit_x[1] - fit_x[0]
		yf = fft(fit_y)
		xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

		f_signal = rfft(fit_y)
		W = fftfreq(fit_y.size, d=fit_x[1]-fit_x[0])

		cut_f_signal = f_signal.copy()
		cut_f_signal[(W>.1/0.01)] = 0  # filter freq

		cut_signal = irfft(cut_f_signal)

		# plot results
		f, axarr = plt.subplots(1, 3, figsize=(10, 5))
		axarr[0].grid()
		axarr[0].plot(x,y)
		axarr[0].set_ylabel(r'$\tau$ [Nm]')
		axarr[0].legend((r'$\tau_1$', r'$\tau_2$'),loc='upper right')
		axarr[0].set_xlabel("Time [s]")

		axarr[1].grid()
		axarr[1].scatter(fit_x, y[0:300,0],s=50, label=r'$\tau_1$')
		axarr[1].plot(fit_x,cut_signal,'r', label="FFT")
		axarr[1].legend(loc="upper right")
		axarr[1].set_xlabel("Time [s]")

		axarr[2].grid()
		axarr[2].plot(xf, 2.0/N * np.abs(yf[:N//2]),'r')
		axarr[2].set_xlabel("FFT: frequency [Hz]")
		axarr[2].set_ylabel("FFT: amplitude")

		for i, label in enumerate(('(a)', '(b)', '(c)')):
			axarr[i].text(0.5, 1.05, label, transform=axarr[i].transAxes,
			  			  fontweight='bold', va='top', ha='right')
		
		f.suptitle("FFT analysis of converged torque")

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
		n_updates = data[list(data.T[0]).index("train/n_updates"),1:].astype(float)
		print(max(n_updates))

		fig = plt.figure()
		ax = fig.add_subplot(111)

		ax.set_xlabel("Episode number")
		ax.set_ylabel("Reward")
		ax.set_title("Reward vs Episode")
		ax.grid()
		ax.plot(episodes,ep_mean,linewidth=3)
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
			# print(t)
			# print("Displaying trained robot")
			action, _states = model.predict(obs)

			obs, rewards, dones, info = env.step(action)
			# print(rewards)
			# env2.render()	

if __name__ == '__main__':
	run = "0022"
	TP = TrainedPlotter(run)
	# TP.expert_plotter()
	# TP.progess_plotter()
	# TP.display()
	TP.plot_converged_data()