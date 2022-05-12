import gym
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import time

from flapper_env.resources.bird import Bird
from flapper_env.resources.goal import Goal
client = p.connect(p.GUI)
# client = p.connect(p.DIRECT)
class FlapperEnv(gym.Env):
	metadata = {'render.modes':['human']}

	def __init__(self):

		self.action_space = gym.spaces.Box(
			low=np.array([-1.3, -0.7854, -0.7854]), # [stroke plane angle, wing angle]
			high=np.array([1.3, 0.7854, 0.7854]))

		# How much data should we include?
		self.full_observation = True # Use full observation?
		if self.full_observation:
			# use self.bird.get_full_observation()
			self.observation_shape = 24
		else:
			# use regular self.bird.get_observation()
			self.observation_shape = 14

		self.observation_space = gym.spaces.Box(
			low=-1000*np.ones(self.observation_shape),
			high=1000*np.ones(self.observation_shape))

		self.np_random, _ = gym.utils.seeding.np_random()
		
		# self.client = p.connect(p.DIRECT)
		self.client = client
		p.setTimeStep(1/100, self.client)

		self.traj_reset = True
		self.bird = None
		self.goal = None
		self.done = False
		self.prev_dist_to_goal = None
		self.rendered_img = None
		self.render_rot_matrix = None
		self.max_ep_steps = 1000
		self.ep_steps = 0
		self.reset()

	def step(self, action):
		# print(action)
		self.bird.apply_action(action)
		p.stepSimulation()

		# What the bird is actually able to observe
		if self.full_observation:
			bird_ob, pos, ori, kill_bool = self.bird.get_full_observation()
		else:
			bird_ob, pos, ori, kill_bool = self.bird.get_observation() 

		dist_to_goal = np.linalg.norm(np.array([pos[0] - self.goal[0],
												pos[1] - self.goal[1]]))
		goal_reward = max(self.prev_dist_to_goal - dist_to_goal, 0)
		self.prev_dist_to_goal = dist_to_goal

		# Done by running out of bounds
		if (pos[0] >= 100 or pos[0] <= -100 or 
			pos[1] >= 100 or pos[1] <= -100 or
			pos[2] >= 1.5 or pos[2] <= 0.5):
			self.done = True

		# Done by reaching goal
		if dist_to_goal < 1:
			self.done = True
			goal_reward = 50

		# Done by breaking physical limits of wings/motors
		if kill_bool: 
			self.done = True

		# self.bird.log_data(data_classes, t)
		# Reward for flying fast forward, 
		# 			using less energy, 
		#			staying in vertical bounds, 
		#			reaching goal

		# reward = self.bird.reward() + goal_reward
		reward = self.bird.reward()
		# print(f"                                     Reward: {reward}")

		# ob = np.concatenate((bird_ob, self.goal))
		ob = bird_ob

		self.ep_steps += 1
		# Limits episode length
		if self.ep_steps == self.max_ep_steps:
			self.done = True
			self.ep_steps = 0

		return ob, reward, self.done, dict()

	def reset(self):

		p.resetSimulation(self.client)
		p.setGravity(0,0,0)

		# Reload bird
		self.bird = Bird(self.client, self.traj_reset)


		# Set goal to random target
		x = (self.np_random.uniform(5, 9) if self.np_random.randint(2) else
			 self.np_random.uniform(-5, -9))
		y = (self.np_random.uniform(5, 9) if self.np_random.randint(2) else
			 self.np_random.uniform(-5, -9))
		self.goal = np.array([x,y])
		self.prev_dist_to_goal = np.linalg.norm(self.goal)
		Goal(self.client,self.goal)

		self.done = False
		if self.full_observation:
			bird_ob, pos, ori, kill_bool = self.bird.get_full_observation()
		else:
			bird_ob, pos, ori, kill_bool = self.bird.get_observation()

		# ob = np.concatenate((bird_ob, self.goal))
		ob = bird_ob	

		return ob

	def render(self, mode='human'):

		if self.rendered_img is None:
			self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))

		# Base info
		bird_id, client_id = self.bird.get_ids()
		bird_ob, pos, ori, kill_bool = self.bird.get_observation()

		proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,nearVal=0.01, farVal=100)

		# Rotate camera direction
		rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
		camera_vec = np.matmul(rot_mat, [1, 0, 0])
		up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
		view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

		# Display image
		frame = p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
		frame = np.reshape(frame, (100, 100, 4))
		self.rendered_img.set_data(frame)
		plt.draw()
		plt.pause(.00001)

	def close(self):
		p.disconnect(self.client)

	def seed(self, seed=None):
		self.np_random, seed = gym.utils.seeding.np_random(seed)
		return [seed]