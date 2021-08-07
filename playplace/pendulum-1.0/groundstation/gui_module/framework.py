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
		ep_reward_tab = ttk.Frame(tab_parent)
		
		tabs = {"action" : action_tab,
				"next state" : next_state_tab,
				"reward" : reward_tab,
				"episode reward" : ep_reward_tab
				}


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
		tab_parent.pack(fill='both')