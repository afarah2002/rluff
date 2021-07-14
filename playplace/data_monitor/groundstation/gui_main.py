#-----------------imports-----------------#
import numpy as np
import threading
import queue
import time
import random
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import rcParams
import matplotlib.pyplot as plt
rcParams.update({'figure.autolayout': True})

import sys
sys.path.insert(0,'../..')
import myTD3.myplottingutils as plottingutils

form = tk.Tk()
form.title("Tkinter Database Form")
form.geometry("500x200")

tab_parent = ttk.Notebook(form)
tab1 = ttk.Frame(tab_parent)
tab2 = ttk.Frame(tab_parent)

tab_parent.add(tab1, text="All records")
tab_parent.add(tab2, text="Add New Record")

tab_parent.pack(expand=1, fill='both')

form.mainloop()

# if __name__ == '__main__':






