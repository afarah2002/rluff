import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
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
		self.XData = [0]
		self.YData = [np.zeros(self.num_lines)]

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

		if len(XData_copy) >= 50 and fig_name != "Episode reward":
			del XData_copy[0:-50]
			del YData_copy[0:-50]

		# print(data_class.XData)
		YData_transposed = np.array(YData_copy.copy()).T
		for lnum, line in enumerate(line_set):
			line.set_data(XData_copy, YData_transposed[lnum])

		plot.axes.relim()
		plot.axes.autoscale_view()

		if np.max(YData_copy) > max_y:
			max_y = np.max(YData_copy)

		if np.min(YData_copy) < min_y:
			min_y = np.min(YData_copy)

		plot.set_ylim([min_y, max_y])

		fig.figure.canvas.draw_idle()
