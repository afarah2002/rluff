import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import threading
import random
import time
import os

class MyDataClass(object):

	def __init__(self, num_actions):
		self.num_actions = num_actions
		self.XData = [0]
		self.actions_data = [np.zeros(self.num_actions)]

		self.episode_numbers = [0]
		self.episodic_reward_data = [0]


class MyPlotClass(object):

	def __init__(self, dataClass):

		self._dataClass = dataClass

		self.fig, self.subplot_axes = plt.subplots(2,1,squeeze=False)
		self.fig.subplots_adjust(wspace=0.4)
		self.fig.tight_layout()

		#-----------------actions plot-----------------#
		self.actions_plot = self.subplot_axes[0][0]
		self.actions_plot.set_xlabel("Time")
		self.actions_plot.set_ylabel("Joint torques")
		self.actions_plot.set_title("Joint torques vs Time")
		self.actions_plot.grid()

		self.actions_lines = []
		for i in range(self._dataClass.num_actions):
			new_line = self.actions_plot.plot([],[],lw=2)[0]
			self.actions_lines.append(new_line)

		#-----------------rewards plot-----------------#
		self.rewards_plot = self.subplot_axes[1][0]
		self.rewards_plot.grid()
		self.rewards_plot.set_xlabel("Episode number")
		self.rewards_plot.set_ylabel("Episodic reward")
		self.rewards_plot.set_title("Episodic reward vs Episodic number")
		self.rewards_line = self.rewards_plot.plot([],[],lw=2)[0]
		
		#-----------------call animation-----------------#
		self.ani = FuncAnimation(self.fig, 
								 self.run,
								 interval = 10, 
								 repeat=True, blit=False)

	def run(self, i):  
		# print("plotting data")
		actions_transposed = np.array(self._dataClass.actions_data.copy()).T
		# print(self._dataClass.XData)
		# print(actions_transposed)
		for lnum,line in enumerate(self.actions_lines):
			line.set_data(self._dataClass.XData, actions_transposed[lnum])

		self.actions_plot.axes.relim()
		self.actions_plot.axes.set_ylim([-1.,1])
		self.actions_plot.axes.autoscale_view()

		self.rewards_line.set_data(self._dataClass.episode_numbers, self._dataClass.episodic_reward_data)
		
		self.rewards_plot.axes.relim()
		self.rewards_plot.axes.autoscale_view()

class SaveData(object):

	def __init__(self, action_file_loc, state_file_loc):
		self.action_file_loc = action_file_loc # txt filename string
		self.state_file_loc = state_file_loc 
		os.system("touch " + self.action_file_loc + " " + self.state_file_loc)
		open(self.action_file_loc).close()
		open(self.state_file_loc).close()


	def save_action(self, action_list):
		# print("Saving action data.....")
		print(action_list)
		with open(self.action_file_loc, 'a') as f:
			for item in action_list:
				f.write(str(item) + " ")
			f.write("\n")

	def save_state(self, state_list):
		print("Saving state data.....")
		with open(self.state_file_loc, 'a') as f:
			for item in state_list:
				f.write(str(item) + " ")
			f.write("\n")


