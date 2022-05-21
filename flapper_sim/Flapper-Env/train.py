import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.stats
import shutil
import gym
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

import gui.utils as gui_utils
import gui.framework as gui_framework

class Threads:

	def expert_main(data_classes):

		env = gym.make('Flapper-v0')
		num_interactions = int(4e4)

		if isinstance(env.action_space, gym.spaces.Box):
			expert_observations = np.empty((num_interactions,) + env.observation_space.shape)
			expert_actions = np.empty((num_interactions,) + (env.action_space.shape[0],))
		else:
			expert_observations = np.empty((num_interactions,) + env.observation_space.shape)
			expert_actions = np.empty((num_interactions,) + env.action_space.shape)

		obs = env.reset()
		t = np.arange(0,2*np.pi,0.001)
		i = 0
		# A random phase shift in the stroke plane oscillation
		sp_shift = 0 # Will be random w each reset
		for timestep in tqdm(range(num_interactions)):
			stroke_plane_angle = -20*np.pi/180*np.sin(50*t[i] - sp_shift*90*np.pi/180)
			left_wing_pos = np.cos(50*(t[i]))*(np.pi/180)*30
			right_wing_pos = np.cos(50*(t[i]))*(np.pi/180)*30

			i += 1
			if i == len(t):
				i = 0
				t = np.arange(max(t),max(t)+2*np.pi,0.001)
			
			action = np.array([stroke_plane_angle, left_wing_pos, right_wing_pos])
			
			# # only start saving once it has reached a steady state:
			# if i > 600:
			expert_observations[timestep] = obs
			expert_actions[timestep] = action

			obs, rewards, done, _ = env.step(action)
			if done:
				# Different random phase shift in the stroke plane osciallation
				i = 0
				sp_shift = np.random.uniform(-1.,1.)
				print(f"New stroke plane angle phase shift: {sp_shift*90} deg")
				env.reset()

		np.savez_compressed(
			"expert_data",
			expert_actions=expert_actions,
			expert_observations=expert_observations)

	def ai_main_train(run_num):

		# directory organization
		runs_names = ["0"*(4-len(str(i))) + str(i) for i in range(1000)]
		run = runs_names[run_num]
		agent_dir = f"/home/nasa01/Documents/UML/willis/rluff/flapper_sim/Flapper-Env/trained_agents/SAC{run}"
		if os.path.exists(agent_dir) and os.path.isdir(agent_dir):
			shutil.rmtree(agent_dir)
		Path(agent_dir).mkdir(parents=True, exist_ok=True)

		params = {"env" : {"2D" : True, # constraints to 2d plane
						   "full obs" : True, # gives as much data as possible
				  		   "gravity" : [0,0,0], 
				  		   "gui" : False, 
				  		   "max ep steps" : 100*np.random.randint(10,20),
				  		   "run" : run, # run number
				  		   "traj reset" : False,
				  		   "total timesteps" : 1e5}, # resets into a state familiar to student
				  "bird" : {"student" : False, # uses pretrained model
				  			"rewards" : 
				  				{"forward" : 1.,
				  				 "torque" : -np.random.uniform(0,0.2), # penalties must be negative
				  				 "height" : -np.random.uniform(0.001,0.1), # penalties must be negative
				  				 "instability" : -np.random.uniform(0.0001,0.01)}, # penalties must be negative
				  			"max wing ang vel" : 1000,
				  			}
				  }
		print(params)
		# save params for this run
		np.save(f"{agent_dir}/params.npy", params)
		
		env = gym.make('Flapper-v0', params=params)
		n_actions = env.action_space.shape[-1]
		action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

		# Logger 
		log_path = f"{agent_dir}/log"
		logger = configure(log_path, ["stdout", "csv", "tensorboard"])

		if params["bird"]["student"]:
			model = SAC.load("sac_student") # use pretrained model
			model.set_env(env)
		else:
			model = SAC("MlpPolicy", 
						env, 
						action_noise=action_noise, 
						verbose=1, 
						tensorboard_log="./flapper_tensorboard/")
		
		print("Training...")

		model.set_logger(logger)
		model.learn(total_timesteps=params["env"]["total timesteps"], tb_log_name="first_run")
		model.save(f"{agent_dir}/agent")



def main():

	for i in range(2,1000): # num of runs
		Threads.ai_main_train(i)
	# i = int(float(input()))
	# print(i)

	# mainQueue = Queue()

	# GUI = False

	# data_classes = {"Stroke" : gui_utils.GUIDataClass("tab1", "Stroke", int(2+3*2)),
	# 				"AoA" : gui_utils.GUIDataClass("tab2", "AoA", 2),
	# 				"Reward" : gui_utils.GUIDataClass("tab3", "Reward", 1)}
	# 				# "Vectors" : gui_utils.GUIDataClass("tab3", "Vectors", 3)}
	# if GUI:
	# 	tab1_figs = [gui_utils.NewMPLFigure(data_classes["Stroke"])]
	# 	tab2_figs = [gui_utils.NewMPLFigure(data_classes["AoA"])]
	# 	tab3_figs = [gui_utils.NewMPLFigure(data_classes["Reward"])]

	# 	gui_figs = [tab1_figs, tab2_figs, tab3_figs]

	# 	for gui_fig_type in gui_figs:
	# 		lines_sets = [fig.lines for fig in gui_fig_type]

	# 	gui_app = gui_framework.GUI(gui_figs)

	# 	anis = [animation.FuncAnimation(fig.figure,
	# 									gui_utils.MPLAnimation.animate,
	# 									interval=10,
	# 									fargs=[fig])
	# 									for fig in list(itertools.chain.from_iterable(gui_figs))]

	# ai_main_thread = threading.Thread(target=Threads.ai_main_train)
	# ai_main_thread.start()

	# expert_main_thread = threading.Thread(target=Threads.expert_main,
	# 								  args=(data_classes,))
	# expert_main_thread.start()

	# if GUI:
	# 	gui_app.mainloop()

if __name__ == '__main__':
	main()