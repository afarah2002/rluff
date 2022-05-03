import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import matplotlib.figure as mplfig
from tkinter import ttk
import tkinter as tk

import gui.utils as gui_utils

class GUI(tk.Tk):

	def __init__(self, gui_figs, *args, **kwargs):
		tk.Tk.__init__(self, *args, **kwargs)
		self.wm_title('Gui')
		container = tk.Frame(self)
		container.pack(side='top', fill='both', expand=True)

		tab_parent = ttk.Notebook(self)
		tab1 = ttk.Frame(tab_parent)
		tab2 = ttk.Frame(tab_parent)
		tab3 = ttk.Frame(tab_parent)
		tab4 = ttk.Frame(tab_parent)
		
		tabs = {"tab1" : tab1,
				"tab2" : tab2,
				"tab3" : tab3,
				"tab4" : tab4
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