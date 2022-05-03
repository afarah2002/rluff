import gym
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import time

from flapper_env.resources.bird import Bird
client = p.connect(p.GUI)
class FlapperEnv(gym.Env):
	metadata = {'render.modes':['human']}

	def __init__(self, GUI=False):
		self.action_space = gym.spaces.Box(
			low=np.array([-1.3, -0.7854, -0.7854]), # [stroke plane angle, wing angle]
			high=np.array([1.3, 0.7854, 0.7854]))
		self.observation_space = gym.spaces.Box(
			low=np.array([-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]), # [base pos (xyz), base vel (xyz) ]  
			high=np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]))
		self.np_random, _ = gym.utils.seeding.np_random()
		self.client = client
		p.setTimeStep(1/100, self.client)

		self.bird = None
		self.done = False
		self.rendered_img = None
		self.reset()

	def step(self, action):
		self.bird.apply_action(action)
		p.stepSimulation()
		# time.sleep(.5)
		bird_ob, kill_bool = self.bird.get_observation()
		# self.bird.log_data(data_classes, t)
		reward = self.bird.reward()
		print(f"                                     Reward: {reward}")

		self.done = False
		if kill_bool:
			self.done = True

		return bird_ob, reward, self.done, dict()

	def reset(self):
		p.resetSimulation(self.client)
		# p.setGravity(0,0,-1.97)
		p.setGravity(0,0,0)
		# Reload bird
		self.bird = Bird(self.client)
		self.done = False
		bird_ob, kill_bool = self.bird.get_observation()
		print(bird_ob)

		return np.array(bird_ob)

	def render(self):
		pass

	def close(self):
		p.disconnect(self.client)

	def seed(self, seed=None):
		self.np_random, seed = gym.utils.seeding.np_random(seed)
		return [seed]