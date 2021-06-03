import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import threading
import random
import time

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

		# self.hLine, = plt.plot(0, 0)
		self.fig = plt.figure()
		# self.fig2 = plt.figure()
		self.fig.tight_layout()
		# self.fig.subplots_adjust(hspace=.5)

		#-----------------actions plot-----------------#
		self.actions_plot = self.fig.add_subplot(111)
		self.actions_plot.grid()


		self.actions_lines = []

		for i in range(self._dataClass.num_actions):
			new_line = self.actions_plot.plot([],[],lw=2)[0]
			self.actions_lines.append(new_line)

		#-----------------rewards plot-----------------#
		# self.rewards_plot = self.fig.add_subplot(211)
		# self.rewards_plot.grid()

		# self.rewards_line = self.rewards_plot.plot([],[],lw=2)[0]
		#----------------------------------------------#

		self.ani = FuncAnimation(plt.gcf(), 
								 self.run,
								 interval = 10, 
								 repeat=True, blit=False)

	def run(self, i):  
		# print("plotting data")
		# self.hLine.set_data(self._dataClass.XData, self._dataClass.actions_data)

		actions_transposed = np.array(self._dataClass.actions_data.copy()).T
		# print(self._dataClass.XData)
		# print(actions_transposed)
		# print(actions_transposed)
		for lnum,line in enumerate(self.actions_lines):
			# print(type(line))
			line.set_data(self._dataClass.XData, actions_transposed[lnum])

		self.actions_plot.axes.relim()
		self.actions_plot.axes.autoscale_view()

		# self.rewards_line.set_data(self._dataClass.episode_numbers, self._dataClass.episodic_reward_data)

		# self.rewards_plot.axes.relim()
		# self.rewards_plot.axes.autoscale_view()

# class MyDataFetchClass(threading.Thread):

# 	def __init__(self, dataClass):

# 		threading.Thread.__init__(self)

# 		self._dataClass = dataClass
# 		self._period = 0.3
# 		self._nextCall = time.time()


# 	def run(self):

# 		while True:
# 			print("updating data")
# 			# add data to data class
# 			self._dataClass.XData.append(self._dataClass.XData[-1] + .1)
# 			self._dataClass.actions_data.append(np.random.rand(self._dataClass.num_actions))
# 			# sleep until next execution
# 			self._nextCall = self._nextCall + self._period;
# 			time.sleep(self._nextCall - time.time())

# num_actions = 2


# data = MyDataClass(num_actions)
# plotter = MyPlotClass(data)
# fetcher = MyDataFetchClass(data)

# fetcher.start() # <---------- this is just the RL runner
# plt.show()
# #fetcher.join()