import gym
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import time

from flapper_env.resources.bird import Bird

class FlapperEnv(gym.Env):
	metadata = {'render.modes':['human']}

	def __init__(self):
		self.action_space = gym.spaces.Box(
			low=np.array([-1.3, -200]), # [stroke plane angle, wing torque]
			high=np.array([1.3, 200]))
		self.observation_space = gym.spaces.Box(
			low=np.array([-100, -100, 0, -100, -100, -100, -100, -100, -100]), # [base pos (xyz), base vel (xyz) ]  
			high=np.array([100, 100, 100, 100, 100, 100, 100, 100, 100]))
		self.np_random, _ = gym.utils.seeding.np_random()
		# self.client = p.connect(p.DIRECT)
		self.client = p.connect(p.GUI)
		p.setTimeStep(1/1000, self.client)

		self.bird = None
		self.done = False
		self.rendered_img = None
		self.reset()

	def step(self, action, c):
		self.bird.apply_action(action, c)
		p.stepSimulation()
		# time.sleep(.5)	
		bird_ob, lim_broken_bool = self.bird.get_observation() # pos, vel, ang vel of base
		reward = 0

		if lim_broken_bool:
			self.done = True

		return bird_ob, reward, self.done, dict()

	def reset(self):
		p.resetSimulation(self.client)
		# p.setGravity(0,0,-1.97)
		p.setGravity(0,0,0)
		# Reload bird
		self.bird = Bird(self.client)
		self.done = False
		bird_ob = self.bird.get_observation()

		return np.array(bird_ob)

	def render(self):
		pass

	def close(self):
		p.disconnect(self.client)

	def seed(self, seed=None):
		self.np_random, seed = gym.utils.seeding.np_random(seed)
		return [seed]