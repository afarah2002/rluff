import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import matplotlib.figure as mplfig
from tkinter import ttk
import tkinter as tk

import groundstation.gui_module.utils as gui_utils

class GUI(tk.Tk):

	def __init__(self, gui_figs, *args, **kwargs):
		tk.Tk.__init__(self, *args, **kwargs)
		self.wm_title('Gui')
		container = tk.Frame(self)
		container.pack(side='top', fill='both', expand=True)

		tab_parent = ttk.Notebook(self)
		action_tab = ttk.Frame(tab_parent)
		next_state_tab = ttk.Frame(tab_parent)
		reward_tab = ttk.Frame(tab_parent)
		tabs = {"action" : action_tab,
				"next state" : next_state_tab,
				"reward" : reward_tab}

		tab_name = ""
		row_i = 0
		for gui_fig_type in gui_figs:
			row_i = 0
			for f in gui_fig_type:
				tab = tabs[f.tab_name]
				tab_parent.add(tab, text=f.tab_name)
				bfig = gui_utils.NewTkFigure(tab, controller=self, f=f.figure)
				bfig.grid(row=row_i, column=0, sticky="nsew")
				container.grid_rowconfigure(row_i, weight=1)
				row_i += 1

		# for i, f in zip(list(range(len(gui_figs))), gui_figs):
		# 	tab = tabs[f.tab_name]
		# 	tab_parent.add(tab, text=f.tab_name)
		# 	bfig = gui_utils.NewTkFigure(tab, controller=self, f=f.figure)
		# 	if tab_name != f.tab_name:
		# 		tab_name = f.tab_name
		# 		row_i = 0
		# 	row_i += 1
		# 	bfig.grid(row=i, column=0, sticky="nsew")
		# 	# give the rows equal weight so they are allotted equal size
		# 	container.grid_rowconfigure(i, weight=1)

		# you need to give at least one row and one column a positive weight 
		# https://stackoverflow.com/a/36507919/190597 (Bryan Oakley)
		# container.grid_columnconfigure(0, weight=1)
		tab_parent.pack(fill='both')


