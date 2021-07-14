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

def data_gen(data_storage):
    x_data = 0
    while True:
        # time.sleep(10e-3)
        x_data += .01
        new_data = [random.random() for _ in range(6)]

        # print(time, new_data)
        # print(data_storage.XData[-1])
        # print(len(data_storage.XData))

        data_storage.XData.append(x_data)
        data_storage.actions_data.append(new_data)

        if len(data_storage.XData) >= 50:
            del data_storage.XData[0]
            del data_storage.actions_data[0]
        root.update()


data_storage = plottingutils.MyDataClass(6)
plotter = plottingutils.MyPlotClass(data_storage)
fig = plotter.get_figure_object()

root = tk.Tk()
label = tk.Label(root, text='animation').grid(column=0, row=0)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(column=0, row=1)

# t_gui = threading.Thread(target=tk.mainloop())
t_data_gen = threading.Thread(target=data_gen,
                              args=(data_storage, ))

t_data_gen.start()
# t_gui.start()
# plt.show()
tk.mainloop()
# while True:
tk.update()

# plt.show()
# tk.mainloop()
# form = tk.Tk()
# form.title("Tkinter Database Form")
# form.geometry("500x200")

# tab_parent = ttk.Notebook(form)
# tab1 = ttk.Frame(tab_parent)
# tab2 = ttk.Frame(tab_parent)

# tab_parent.add(tab1, text="All records")
# tab_parent.add(tab2, text="Add New Record")

# tab_parent.pack(expand=1, fill='both')

# form.mainloop()

# if __name__ == '__main__':






