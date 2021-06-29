import numpy as np
import torch
import gym
import argparse
import os
import pybullet as p
import pybullet_envs
import sys
import random
import time

import myutils
import myTD3
import myOurDDPG
import myDDPG
import myplottingutils
import mytorquegenerators

def eval_policy_raw(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward

def eval_policy(policy, env_name, seed, env_action_dim, no_terms, f_bounds, eval_episodes=10,):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.

	f_s = 200*3
	t_0 = np.linspace(0., 2*np.pi, f_s, endpoint=False)

	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			primes = policy.select_action(np.array(state))
			new_sin_amps = np.clip(primes[:int(len(primes)/3)], 0., 1.)
			new_cos_amps = np.clip(primes[int(len(primes)/3):2*int(len(primes)/3)], 0., 1.)
			new_freqs = np.clip(primes[2*int(len(primes)/3):], f_bounds[0], f_bounds[1])
			torques = mytorquegenerators.TorqueGenFuncs.brute_force_torque_gen(no_terms,
																	 		   new_sin_amps,
																	 		   new_cos_amps,
																			   new_freqs,
																			   t_0,
																			   env_action_dim)
			for time_i in range(len(t_0)):
				action = torques[:, time_i]
				state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward

class Options(object):

	# action_dim = 0

	def __init__(self, RENDER_BOOL=False):
		self.parser = argparse.ArgumentParser()
		self.parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
		self.parser.add_argument("--env", default="HalfCheetahBulletEnv-v0")          # OpenAI gym environment name
		self.parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
		self.parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
		self.parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
		self.parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
		self.parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
		self.parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
		self.parser.add_argument("--discount", default=0.99)                 # Discount factor
		self.parser.add_argument("--tau", default=0.005)                     # Target network update rate
		self.parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
		self.parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
		self.parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
		self.parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
		self.parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name

		self.args = self.parser.parse_args()

		self.file_name = f"{self.args.policy}_{self.args.env}_{self.args.seed}"
		print("---------------------------------------")
		print(f"Policy: {self.args.policy}, Env: {self.args.env}, Seed: {self.args.seed}")
		print("---------------------------------------")

		if not os.path.exists("./results"):
			os.makedirs("./results")

		if self.args.save_model and not os.path.exists("./models"):
			os.makedirs("./models")

		self.env = gym.make(self.args.env)
		if RENDER_BOOL:
			self.env.render()

		# Set seeds
		self.env.seed(self.args.seed)
		self.env.action_space.seed(self.args.seed)
		torch.manual_seed(self.args.seed)
		np.random.seed(self.args.seed)
		
		self.state_dim = self.env.observation_space.shape[0]
		self.action_dim = self.env.action_space.shape[0]
		# self.action_dim	= action_dim 
		self.max_action = float(self.env.action_space.high[0])

		


	def raw_TD3(self, out_q):

		self.kwargs = {
			"state_dim": self.state_dim,
			"action_dim": self.action_dim,
			"max_action": self.max_action,
			"discount": self.args.discount,
			"tau": self.args.tau,
		}

		# Initialize policy
		if self.args.policy == "TD3":
			# Target policy smoothing is scaled wrt the action scale
			self.kwargs["policy_noise"] = self.args.policy_noise * self.max_action
			self.kwargs["noise_clip"] = self.args.noise_clip * self.max_action
			self.kwargs["policy_freq"] = self.args.policy_freq
			self.policy = myTD3.TD3(**self.kwargs)
		elif self.args.policy == "OurDDPG":
			self.policy = myOurDDPG.DDPG(**self.kwargs)
		elif self.args.policy == "DDPG":
			self.policy = myDDPG.DDPG(**self.kwargs)

		if self.args.load_model != "":
			policy_file = self.file_name if self.args.load_model == "default" else self.args.load_model
			self.policy.load(f"./models/{policy_file}")

		# Evaluate untrained policy
		self.evaluations = [eval_policy_raw(self.policy, self.args.env, self.args.seed)]

		self.replay_buffer = myutils.ReplayBuffer(self.state_dim, self.action_dim)

		state, done = self.env.reset(), False
		episode_reward = 0
		episode_timesteps = 0
		episode_num = 0

		# self.data_storage = myplottingutils.MyDataClass(action_dim)
		# self.plotter = myplottingutils.MyPlotClass(self.data_storage)

		for t in range(int(self.args.max_timesteps)):
			print(t)
			episode_timesteps += 1

			# Select action randomly or according to policy
			if t < self.args.start_timesteps:
				action = self.env.action_space.sample()
				out_q.put((t, action, episode_num, episode_reward))
			else:
				action = (
					self.policy.select_action(np.array(state))
					+ np.random.normal(0, self.max_action * self.args.expl_noise, size=self.action_dim)
				).clip(-self.max_action, self.max_action)
			
			# Send action to queue
			out_q.put((t, action, episode_num, episode_reward))
			print(action)
			# Perform action
			next_state, reward, done, _ = self.env.step(action) 
			done_bool = float(done) if episode_timesteps < self.env._max_episode_steps else 0

			# Store data in replay buffer
			self.replay_buffer.add(state, action, next_state, reward, done_bool)

			state = next_state
			episode_reward += reward

			# Train agent after collecting sufficient data
			if t >= self.args.start_timesteps:
				self.policy.train(self.replay_buffer, self.args.batch_size)

			if done: 
				# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
				print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
				# Reset environment
				state, done = self.env.reset(), False
				episode_reward = 0
				episode_timesteps = 0
				episode_num += 1 

			# Evaluate episode
			if (t + 1) % self.args.eval_freq == 0:
				self.evaluations.append(eval_policy_raw(self.policy, self.args.env, self.args.seed))
				np.save(f"./results/{self.file_name}", self.evaluations)
				if self.args.save_model: self.policy.save(f"./models/{self.file_name}")


	def brute_force(self, out_q):

		'''
		Finite # of sine terms, actor outputs A and f for each one, for each joint/main action 
		i.e.
		 - 4 terms
		 - 2 variables (A,f)
		 - 6 joints
		  == 48 (4x2x6) outputs from NN
		'''

		self.env_action_dim = self.env.action_space.shape[0] # num of joints (original action_dim)
		no_terms = 5
		f_bounds = [self.env.action_space.low[0], self.env.action_space.high[0]]
		self.max_action = f_bounds[1]
		self.action_dim = self.env_action_dim * 3 * no_terms

		self.kwargs = {
			"state_dim": self.state_dim,
			"action_dim": self.action_dim,
			"max_action": self.max_action,
			"discount": self.args.discount,
			"tau": self.args.tau,
		}

		# Initialize policy
		if self.args.policy == "TD3":
			# Target policy smoothing is scaled wrt the action scale
			self.kwargs["policy_noise"] = self.args.policy_noise * self.max_action
			self.kwargs["noise_clip"] = self.args.noise_clip * self.max_action
			self.kwargs["policy_freq"] = self.args.policy_freq
			self.policy = myTD3.TD3(**self.kwargs)
		elif self.args.policy == "OurDDPG":
			self.policy = myOurDDPG.DDPG(**self.kwargs)
		elif self.args.policy == "DDPG":
			self.policy = myDDPG.DDPG(**self.kwargs)

		if self.args.load_model != "":
			policy_file = self.file_name if self.args.load_model == "default" else self.args.load_model
			self.policy.load(f"./models/{policy_file}")

		# Evaluate untrained policy
		self.evaluations = [eval_policy(self.policy, self.args.env, self.args.seed, self.env_action_dim, no_terms, f_bounds)]
		self.replay_buffer = myutils.ReplayBuffer(self.state_dim, self.action_dim)

		max_cycles = 3
		f_s = 200*max_cycles
		t_0 = np.linspace(0., max_cycles*2*np.pi, f_s, endpoint=False)
		# f_bounds = [0.001, .1]
		amps_0 = np.array([random.uniform(-1.,1) for _ in range(self.env_action_dim*no_terms)])
		freqs_0 = np.array([random.uniform(f_bounds[0], f_bounds[1]) for _ in range(self.env_action_dim*no_terms)])
		torques_0 = mytorquegenerators.TorqueGenFuncs.brute_force_torque_gen(no_terms,
																			 amps_0,
																			 amps_0,
																			 freqs_0,
																			 t_0,
																			 self.env_action_dim)
		prev_torques = torques_0
		print(torques_0)
		# sys.exit()

		# torques = torques_0
		# new_amps = amps_0
		# new_freqs = freqs_0


		state, done = self.env.reset(), False
		episode_reward = 0
		episode_timesteps = 0
		episode_num = 0

		# self.data_storage = myplottingutils.MyDataClass(action_dim)
		# self.plotter = myplottingutils.MyPlotClass(self.data_storage)

		for t in range(int(self.args.max_timesteps)):
			print("Timestep: ", t)
			episode_timesteps += 1
			cycle_counter = 0
			max_cycles = 2

			# Select action randomly or according to policy
			if t < self.args.start_timesteps:
				new_sin_amps = np.array([random.uniform(-1.,1) for _ in range(self.env_action_dim*no_terms)])
				new_cos_amps = np.array([random.uniform(-1.,1) for _ in range(self.env_action_dim*no_terms)])
				# print(new_amps)
				new_freqs = np.clip(np.array([random.uniform(f_bounds[0], f_bounds[1]) for _ in range(self.env_action_dim*no_terms)]),
									f_bounds[0], f_bounds[1])
				primes = np.concatenate((new_sin_amps, new_cos_amps, new_freqs))
				torques = mytorquegenerators.TorqueGenFuncs.brute_force_torque_gen(no_terms,
																				   new_sin_amps,
																				   new_cos_amps,
																				   new_freqs,
																				   t_0,
																				   self.env_action_dim)
				# action = self.env.action_space.sample()
				# Send action to queue
				# out_q.put((t, action, 0, 0))
			else:
				# THIS IS A NEW SET OF AMPS AND FREQS
				print("NOT RANDOM")
				primes = (
					self.policy.select_action(np.array(state))
					+ np.random.normal(0, self.max_action * self.args.expl_noise, size=self.action_dim)
				).clip(-self.max_action, self.max_action) 
				# Split actor output into new amps and new freqs
				new_sin_amps = np.clip(primes[:int(len(primes)/3)], 0., 1.)
				new_cos_amps = np.clip(primes[int(len(primes)/3):2*int(len(primes)/3)], 0., 1.)
				new_freqs = np.clip(primes[2*int(len(primes)/3):], f_bounds[0], f_bounds[1])

				# generate a new set of joint torques from those amps and freqs
				# if cycle_counter < max_cycles:johnny cheese
				torques = mytorquegenerators.TorqueGenFuncs.brute_force_torque_gen(no_terms,
																		 		   new_sin_amps,
																		 		   new_cos_amps,
																				   new_freqs,
																				   t_0,
																				   self.env_action_dim)
				print("NEW SET")

			for time_i in range(len(t_0)):
				action = torques[:, time_i]
				if time_i == 1:
					print(action)
				# Perform action
				next_state, reward, done, _ = self.env.step(action) 
				# Send action to plotting queue
				
				out_q.put((t_0[time_i], action, episode_num, episode_reward))
				# done_bool = float(done) if episode_timesteps < self.env._max_episode_steps else 0
				# done = False	
				if time_i != len(t_0) - 1:
					done = False
				# if time_i == len(t_0) - 1 and done == True:	
				# 	done = True		
					# t_0 = np.linspace(max(t_0), max(t_0) + max_cycles*2*np.pi, f_s, endpoint=False)

			# print(primes)
			# Store data in replay buffer
			self.replay_buffer.add(state, primes, next_state, reward, done)

			state = next_state
			episode_reward += reward

			# Train agent after collecting sufficient data
			if t >= self.args.start_timesteps:
				self.policy.train(self.replay_buffer, self.args.batch_size)

			if done: 
				# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
				print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
				# Reset environment
				state, done = self.env.reset(), False
				episode_reward = 0
				episode_timesteps = 0
				episode_num += 1 

			# Evaluate episode
			if (t + 1) % self.args.eval_freq == 0:
				self.evaluations.append(eval_policy(self.policy, self.args.env, self.args.seed, self.env_action_dim, no_terms, f_bounds))
				np.save(f"./results/{self.file_name}", self.evaluations)
				if self.args.save_model: self.policy.save(f"./models/{self.file_name}")

		pass

	def complexity_subgoals(self):
		pass

	def consumer(self, in_q, sentinel, data_storage, RPI_BOOL, data_save_obj, servo=None):
		while True:
			for new_data in iter(in_q.get, sentinel):

				action_data_list = [new_data[0]] + [i for i in new_data[1]]
				data_save_obj.save_action(action_data_list)

				if RPI_BOOL:
					servo_angle = 135+135*new_data[1][0]
					servo.turn_with_speed(servo_angle, 70)

				data_storage.XData.append(new_data[0])
				data_storage.actions_data.append(new_data[1])

				if len(data_storage.XData) >= 50:
					del data_storage.XData[0]
					del data_storage.actions_data[0]

				if new_data[2] == data_storage.episode_numbers[-1]:
					data_storage.episodic_reward_data[-1] = new_data[3]
				if new_data[2] != data_storage.episode_numbers[-1]:
					data_storage.episode_numbers.append(new_data[2])
					data_storage.episodic_reward_data.append(new_data[3])