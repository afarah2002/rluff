import os
import random
import numpy as np
import argparse 
import time
import sys

sys.path.append("groundstation/ai_utils")
import utils
import TD3
import DDPG
import OurDDPG
sys.path.append("../..")


class AITechniques(object):

	def __init__(self, action_state_combo_queue, 
					   action_queue,
					   action_dim,
					   state_dim,
					   technique,
					   rewards, 
					   pi_client,
					   pc_server):
		self.action_state_combo_queue = action_state_combo_queue
		self.action_queue = action_queue
		self.rewards = rewards
		self.pi_client = pi_client
		self.pc_server = pc_server

		#-----------------class wide arguments-----------------#
		self.parser = argparse.ArgumentParser()
		self.parser.add_argument("--policy", default="TD3") # Policy name (TD3, DDPG or OurDDPG)
		self.parser.add_argument("--start_timesteps", default=1e3, type=int)# Time steps initial random policy is used
		self.parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
		self.parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
		self.parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
		self.parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
		self.parser.add_argument("--discount", default=0.99)                 # Discount factor
		self.parser.add_argument("--tau", default=0.005)                     # Target network update rate
		self.parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
		self.parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
		self.parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates

		self.args = self.parser.parse_args()
		self.file_name = f"{self.args.policy}"

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
				# action = np.random.rand(self.action_dim).clip(-self.max_action, max_action)
				action = np.array([random.uniform(-1., 1.) for i in range(self.action_dim)])
			else:
				# print("NOT RANDOM")
				action = (
					self.policy.select_action(np.array(state))
					+ np.random.normal(0, self.max_action * self.args.expl_noise, size=self.action_dim)
				).clip(-self.max_action, self.max_action)

			# Perform action
			action_num += 1
			next_state, reward, done = self.step(action, 
												 action_num, 
												 t, 
												 episode_num, 
												 state, 
												 episode_reward)
			# Store data in replay buffer
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
				state, done = self.reset(), False
				episode_reward = 0
				episode_timesteps = 0
				episode_num += 1
				action_num = 0

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
		next_state = self.get_next_state_from_combo(combo_data)
		# done = self.get_ep_status_from_combo(combo_data)
		# Calculate reward from combo, get episode state
		reward, done = self.get_reward(combo_data)
		# Return the next state and the reward
		print(combo_data_pack)
		print("\n")

		# reward = self.get_reward_from_combo(combo_data)
		# Add reward and episode reward to combo pack
		combo_data_pack["reward"] = {"Time" : t,
									 "Reward" : reward}

		combo_data_pack["episode reward"] = {"Time" : episode_num,
									 		 "Episode reward" : [episode_reward + reward[0]]}

		self.action_state_combo_queue.put(combo_data_pack)
		return next_state, reward, done

	def convert_NN_action_to_Pi_action(self, action, state):
		# Converts the values between 0 and 1 from the NN
		# output to values that can actually be sent to 
		# the Pi hardware/motors
		action_copy = np.array(action).copy()
		action[0] = 2.5*action_copy[0]
		return action

	def convert_action_to_action_pack(self, action, action_num, t):
		# Takes an action and converts it into the dictionary
		# that is passed into the action queue
		wing_torque = action[0]

		action_data_pack = {"Time" : t,
							"Action num" : action_num,
							"Wing torques" : [wing_torque]
							}

		return action_data_pack

	def get_next_state_from_combo(self, combo_data):
		# Separates the state from the combo 
		# into a usable 1D array
		next_state = []
		for state_name, state in combo_data["next state"].items():
			if state_name != "Time":
				for state_value in state:
					next_state.append(state_value)

		return next_state

	def get_reward(self, combo_data):
		# reward = self.rewards.reward_1(combo_data)
		reward, done = self.rewards.reward_2(combo_data)
		return reward, done

	def get_reward_from_combo(self, combo_data):
		# Separates the reward from the combo
		# into a single value
		reward = combo_data["reward"]["Reward"]
		return reward

	def get_ep_status_from_combo(self, combo_data):
		# Separates the episodes status from the combo
		# Returns done as True or False
		done = combo_data["done"]["done"]
		return done

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


