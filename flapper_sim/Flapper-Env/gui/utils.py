import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.figure as mplfig
from tkinter import ttk
import tkinter as tk

class GUIDataClass(object):

	'''
	For a particular figure, this where
	x and y(s) data are stored
	'''

	def __init__(self, tab_name, DC_name, num_lines):
		self.tab_name = tab_name
		self.data_class_name = DC_name
		self.num_lines = num_lines
		self.XData = np.zeros(1)
		# self.YData = np.array([np.zeros(self.num_lines)])
		self.YData = np.zeros((1,self.num_lines))

class NewMPLFigure(object):

	'''
	New MPL 2D figure object, capable of 
	displaying multiple ys for a single x
	'''
	def __init__(self, data_class):
		self.data_class = data_class
		self.tab_name = data_class.tab_name
		self.fig_name = data_class.data_class_name
		self.figure = mplfig.Figure(figsize=(6,2), dpi=200)
		self.axs = self.figure.add_subplot(111)
		self.axs.grid()
		self.lines = [self.axs.plot([],[],lw=2)[0] for i in range(data_class.num_lines)]
		self.axs.set_title(self.fig_name)
		self.axs.set_xlabel("Timestep")

class NewMPL3DFigure(object):

	'''
	Used for displaying 3D vectors real time
	'''
	def __init__(self, data_class):
		self.data_class = data_class
		self.tab_name = data_class.tab_name
		self.fig_name = data_class.data_class_name
		self.figure = mplfig.Figure(figsize=(6,4), dpi=200)
		self.axs = self.figure.add_subplot(111, projection='3d')
		self.axs.grid()
		self.axs.set_title(self.fig_name)
		self.axs.set_xlabel("Local x")
		self.axs.set_ylabel("Local y")
		self.axs.set_zlabel("Local z")

class NewTkFigure(tk.Frame):

	'''
	New Tk figure where a new MPL figure
	will be displayed in the GUI
	'''

	def __init__(self, parent, controller, f):
		tk.Frame.__init__(self, parent)
		self.controller = controller
		canvas = FigureCanvasTkAgg(f, self)
		canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
		canvas.draw()

class MPLAnimation:

	def animate(i, fig):
		tab_name = fig.tab_name
		fig_name = fig.fig_name
		line_set = fig.lines 
		data_class = fig.data_class
		plot = fig.axs

		max_y = 0
		min_y = 0

		# combo_data = combo_queue.get()
		# if combo_data:
			# new_x = combo_data[tab_name]["Time"]
			# new_y = combo_data[tab_name][fig_name]

			# data_class.XData.append(new_x)
			# data_class.YData.append(new_y)

		XData_copy = data_class.XData.copy()
		YData_copy = data_class.YData.copy()

		# XData_unique = set(XData_copy)

		# if fig_name == "Episode reward":
		# 	for x in XData_unique:
		# 		unique, counts = np.unique(XData_copy)
		# 		ref_dict = dict(zip(unique,counts))
		# 		if ref_dict[x] > 1:
		# 			index = min(min(np.where(XData_copy==x)))
		# 			YData_copy = np.delete(YData_copy,index)
		# 	XData_copy = np.array(XData_unique)

		if np.size(XData_copy,0) >= 100 and fig_name != "Episode reward":
			XData_copy = np.delete(XData_copy,np.s_[0:-100],0)
			YData_copy = np.delete(YData_copy,np.s_[0:-100],0)
			
		# print(data_class.XData)
		YData_transposed = YData_copy.copy().T
		for lnum, line in enumerate(line_set):
			line.set_data(XData_copy, YData_transposed[lnum])

		plot.axes.relim()
		plot.axes.autoscale_view()

		if np.max(YData_copy) > max_y:
			max_y = np.max(YData_copy)

		if np.min(YData_copy) < min_y:
			min_y = np.min(YData_copy)

		plot.set_ylim([min_y, max_y])
		# plot.set_ylim([-1, 1])
		plot.set_xlim([XData_copy[0], XData_copy[-1]])

		fig.figure.canvas.draw_idle()

class MPL3DAnimation:
	'''
	Animates vectors observed at each node on the wing
		- Wing velocity
		- Local flow
		- Lift
		- Drag
	'''
	def animate(i, fig):
		# Draw wing nodes
		pass
