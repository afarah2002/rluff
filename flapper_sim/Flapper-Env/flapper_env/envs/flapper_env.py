import gym
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import time

from flapper_env.resources.bird import Bird
from flapper_env.resources.goal import Goal

class FlapperEnv(gym.Env):
	metadata = {'render.modes':['human']}

	def __init__(self, params):
		self.params = params
		#----------------------------------#
		# Continuous space boxes
		self.action_space = gym.spaces.Box(
			low=np.array([-1.3, -0.7854, -0.7854]), # [stroke plane angle, wing angle]
			high=np.array([1.3, 0.7854, 0.7854]))

		# How much data should we include?
		if self.params["env"]["full obs"]:
			# use self.bird.get_full_observation()
			self.observation_shape = 24
		else:
			# use regular self.bird.get_observation()
			self.observation_shape = 14
		self.observation_space = gym.spaces.Box(
			low=-1000*np.ones(self.observation_shape),
			high=1000*np.ones(self.observation_shape))
		self.np_random, _ = gym.utils.seeding.np_random()
		#----------------------------------#

		# Which physics client to use?
		if self.params["env"]["gui"]:
			self.client = p.connect(p.GUI)
		else:
			self.client = p.connect(p.DIRECT)

		p.setTimeStep(1/100, self.client)

		self.action_interval = 1 # every _th timestep, it acts
		self.n_step = 0
		self.bird = None
		self.done = False
		self.rendered_img = None
		self.render_rot_matrix = None
		self.max_ep_steps = self.params["env"]["max ep steps"]
		self.ep_steps = 0
		self.reset()

	def step(self, action):
		# print(action)
		self.n_step += 1
		if self.n_step % self.action_interval == 0:
			self.bird.apply_action(action)
		else:
			pass
		# p.stepSimulation()

		# What the bird is actually able to observe
		if self.params["env"]["full obs"]:
			bird_ob, pos, ori, kill_bool = self.bird.get_full_observation()
		else:
			bird_ob, pos, ori, kill_bool = self.bird.get_observation() 

		# Done by running out of bounds
		if (pos[0] >= 100 or pos[0] <= -100 or 
			pos[1] >= 100 or pos[1] <= -100 or
			pos[2] >= 1.5 or pos[2] <= 0.5):
			self.done = True

		# Done by breaking physical limits of wings/motors
		if kill_bool: 
			self.done = True

		# self.bird.log_data(data_classes, t)
		# Reward for flying fast forward, 
		# 			using less energy, 
		#			staying in vertical bounds, 
		#			reaching goal

		reward = self.bird.reward()
		ob = bird_ob

		self.ep_steps += 1
		# Limits episode length
		if self.ep_steps == self.max_ep_steps:
			self.done = True
			self.ep_steps = 0

		return ob, reward, self.done, dict()

	def reset(self):

		p.resetSimulation(self.client)
		gravity = self.params["env"]["gravity"]

		p.setGravity(gravity[0],gravity[1],gravity[2])

		# Reload bird
		self.bird = Bird(self.client, self.params)
		
		self.done = False
		if self.params["env"]["full obs"]:
			bird_ob, pos, ori, kill_bool = self.bird.get_full_observation()
		else:
			bird_ob, pos, ori, kill_bool = self.bird.get_observation()

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