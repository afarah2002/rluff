import os
import random
import pathlib
import numpy as np
import argparse 
import torch
import time
import sys

sys.path.append("groundstation/ai_utils")
import utils
import TD3
import DDPG
import OurDDPG
sys.path.append("../..")


class AITechniques(object):

	def __init__(self, test_num, 
					   target,
					   action_state_combo_queue, 
					   action_queue,
					   action_dim,
					   state_dim,
					   technique,
					   rewards,
					   data_classes, 
					   pi_client,
					   pc_server):
		self.test_num = test_num
		self.target = target
		self.action_state_combo_queue = action_state_combo_queue
		self.action_queue = action_queue
		self.rewards = rewards
		self.data_classes = data_classes
		self.pi_client = pi_client
		self.pc_server = pc_server

		#-----------------class wide arguments-----------------#
		torch.cuda.empty_cache()
		self.parser = argparse.ArgumentParser()
		self.parser.add_argument("--policy", default="TD3") # Policy name (TD3, DDPG or OurDDPG)
		self.parser.add_argument("--start_timesteps", default=2500, type=int)# Time steps initial random policy is used
		self.parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
		self.parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
		self.parser.add_argument("--expl_noise", default=0.2)                # Std of Gaussian exploration noise
		self.parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
		self.parser.add_argument("--discount", default=0.99)                 # Discount factor
		self.parser.add_argument("--tau", default=0.005)                     # Target network update rate
		self.parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
		self.parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
		self.parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates

		self.args = self.parser.parse_args()
		self.dir_name = f"models/{self.test_num}_{self.target}"
		self.file_name = f"{self.test_num}_{self.target}"
		pathlib.Path(f"{self.dir_name}").mkdir(parents=True, exist_ok=True)  # Make dir for storing models

		self.action_dim = action_dim
		self.state_dim = state_dim
		self.max_action = 1.

		techniques = {"infinite res" : self.infinite_res()}

		# Run the chosen technique
		techniques[technique]


	def infinite_res(self):
		
		'''
		States are inputs to the NN, actions are outputs
		No periodic functions, just 1-to-1
		'''

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
			self.policy = TD3.TD3(**self.kwargs)
		elif self.args.policy == "OurDDPG":
			self.policy = OurDDPG.DDPG(**self.kwargs)
		elif self.args.policy == "DDPG":
			self.policy = DDPG.DDPG(**self.kwargs)

		# Evaluate untrained policy
		# self.evaluations = [self.eval_policy()]
		self.replay_buffer = utils.ReplayBuffer(self.state_dim, self.action_dim)

		state, done = self.reset(), False
		episode_reward = 0
		episode_timesteps = 0
		episode_num = 0
		action_num = 0

		for t in range(int(self.args.max_timesteps)):
			# time.sleep(1)
			episode_timesteps += 10
			# Select action randomly or according to policy
			if t < self.args.start_timesteps:
				# Action must be within allowable ranges
				action = np.array([random.uniform(-1., 1.) for i in range(self.action_dim)])
			else:
				# print("NOT RANDOM")
				action = (
					self.policy.select_action(np.array(state))
					+ np.random.normal(0, self.max_action * self.args.expl_noise, size=self.action_dim)
				).clip(-self.max_action, self.max_action)

			# Perform action
			action_num += 1
			action, next_state, reward, done = self.step(action, 
												 		 action_num, 
												 		 t, 
														 episode_num, 
														 state, 
														 episode_reward)
			# Store data in replay buffer
			action = action/0.05 # Convert action back to [-1,1]
			state = next_state
			episode_reward += reward[0]
			self.replay_buffer.add(state, action, next_state, reward, done)

			# Train agent after collecting sufficient data
			if t >= self.args.start_timesteps:
				self.policy.train(self.replay_buffer, self.args.batch_size)
			if done:
				# +1 to account for 0 indexing. 0+ on ep_timesteps since it will increment +1 even if done=True
				# print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
				# Reset env
				# Save model
				print(f"\n\n Episode {episode_num}: Total Reward: {episode_reward}\n\n")
				state, done = self.reset(), False
				episode_reward = 0
				episode_timesteps = 0
				episode_num += 1
				action_num = 0

				# Maybe this will speed up training??

				self.policy.save(f"{self.dir_name}/{self.file_name}")
			torch.cuda.empty_cache()

			# Evaluate episode
			# if (t + 1) % self.args.eval_freq == 0:
			# 	self.evaluations.append(self.eval_policy())




	def step(self, 
			 action, 
			 action_num, 
			 t, 
			 episode_num, 
			 state, 
			 episode_reward):
		# Act here
		# Convert [0,1] action fron NN into usable array
		action = self.convert_NN_action_to_Pi_action(action, state)
		# Convert action array into action data pack
		action_data_pack = self.convert_action_to_action_pack(action, action_num, t)
		# Add action to action queue (send to pi)
		self.action_queue.put(action_data_pack)
		# Read action-state combo from action-state-combo queue (receive from pi)
		combo_data_pack = self.pi_client.receive_data_pack()

		# Data from combo_data_pack
		combo_data = combo_data_pack

		# Compute angular velocity - USING ANG POS DERIV
		ang_vel_depth = 5
		d_ang_pos = np.array(self.data_classes["Wing angles"].YData)[:,0]
		d_real_time = np.array(self.data_classes["Real time"].YData)[:,0]
		if len(d_ang_pos) > ang_vel_depth:
			ang_vel = np.mean(np.diff(d_ang_pos[-ang_vel_depth:]))/np.mean(np.diff(d_real_time[-ang_vel_depth:]))
			# Update combo_data with new ang_vel
			combo_data["next state"	]["Angular velocity"] = [ang_vel]

		next_state = self.get_next_state_from_combo(combo_data)
		# Calculate reward from combo, get episode state
		reward, done = self.get_reward(combo_data)
		# Get the actual action from the combo_data
		action = self.get_action_from_combo(combo_data)
		print(combo_data_pack)
		print("\n")

		# reward = self.get_reward_from_combo(combo_data)
		# Add reward and episode reward to combo pack
		combo_data_pack["reward"] = {"Time" : t,
									 "Reward" : reward}

		combo_data_pack["episode reward"] = {"Time" : episode_num,
									 		 "Episode reward" : [episode_reward + reward[0]]}

		self.action_state_combo_queue.put(combo_data_pack)
		return action, next_state, reward, done

	def convert_NN_action_to_Pi_action(self, action, state):
		# Converts the values between 0 and 1 from the NN
		# output to values that can actually be sent to 
		# the Pi hardware/motors
		action_copy = np.array(action).copy()
		action[0] = 0.05*action_copy[0]
		# action[0] = 0.01*action_copy[0] # For dT instead of T
		# action[0] = 0.25
		return action

	def convert_action_to_action_pack(self, action, action_num, t):
		# Takes an action and converts it into the dictionary
		# that is passed into the action queue

		wing_torque = action[0]
		# wing_torque = 0

		action_data_pack = {"Time" : t,
							"Action num" : action_num,
							"Wing torques" : [wing_torque]
							}

		return action_data_pack

	def get_action_from_combo(self, combo_data):
		# During failsafes, the action is changed to 0!!!
		# The AI must know this when storing the action in the buffer!!
		action = np.array([combo_data["action"]["Wing torques"][0]])
		return action

	def get_next_state_from_combo(self, combo_data):
		# Separates the state from the combo 
		# into a usable 1D array
		next_state = []
		for state_name, state in combo_data["next state"].items():
			if state_name != "Time" and state_name != "Real time":
				for state_value in state:
					next_state.append(state_value)

		return next_state

	def get_reward(self, combo_data):
		# reward = self.rewards.reward_1(combo_data)
		reward, done = self.rewards.reward_2(combo_data, self.target)
		return reward, done

	def get_reward_from_combo(self, combo_data):
		# Separates the reward from the combo
		# into a single value
		reward = combo_data["reward"]["Reward"]
		return reward

	def reset(self):
		# Reset the wings and stroke plane to 0 degrees
		# Set state to resting state
		state = np.zeros(self.state_dim) # <-- Should include gravity :)
		return state

	def eval_policy(self, eval_episodes=10):
		# print("EVALUATING POLICY...")
		avg_reward = 0.
		for _ in range(eval_episodes):
			state, done = self.reset(), False
			while not done:
				action = self.policy.select_action(np.array(state))
				# print("hello")

				state, reward, done = self.step(action, _)
				
				# print(state, reward, done)

				avg_reward += reward

		avg_reward /= eval_episodes
		return avg_reward


