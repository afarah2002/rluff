import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import matplotlib.figure as mplfig
import tkinter as tk
import threading
import queue
import time

from client_test import GUISocketClient

def animate(i, fig, q):
	fig_name = fig.fig_name
	line_set = fig.lines
	data_class = fig.data_class
	plot = fig.axs

	try:
		new_data = q.get()
		print(new_data)
		new_x = new_data[fig_name][0]
		new_y = new_data[fig_name][1]

		data_class.XData.append(new_x)
		data_class.YData.append(new_y)

		# print(data_class.XData)

		if len(data_class.XData) > 50:
			del data_class.XData[0]
			del data_class.YData[0]

		YData_transposed = np.array(data_class.YData.copy()).T
		for lnum, line in enumerate(line_set):
			line.set_data(data_class.XData, YData_transposed[lnum])

		plot.axes.relim()
		# plot.axes.set_ylim([-1,1])
		plot.axes.autoscale_view()
		fig.figure.canvas.draw_idle()
	except:
		pass


class GUIDataClass(object):

	def __init__(self, DC_name, num_lines):
		self.data_class_name = DC_name
		self.num_lines = num_lines
		self.XData = [0]
		self.YData = [np.zeros(self.num_lines)]

class NewMPLFigure(object):

	def __init__(self, fig_name, data_class):
		self.fig_name = fig_name
		self.data_class = data_class
		self.figure = mplfig.Figure(figsize=(6,2), dpi=100)
		self.axs = self.figure.add_subplot(111)
		self.lines = [self.axs.plot([],[],lw=2)[0] for i in range(data_class.num_lines)]

class NewTkFigure(tk.Frame):
	def __init__(self, parent, controller, f):
		tk.Frame.__init__(self, parent)
		self.controller = controller
		canvas = FigureCanvasTkAgg(f, self)
		canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
		canvas.draw()


class App(tk.Tk):

	def __init__(self, *args, **kwargs):
		tk.Tk.__init__(self, *args, **kwargs)
		self.wm_title('Beam calculator')
		container = tk.Frame(self)
		container.pack(side='top', fill='both', expand=True)
		for i, f in enumerate(figs_list):
			bfig = NewTkFigure(container, controller=self, f=f.figure)
			bfig.grid(row=i, column=0, sticky="nsew")
			# give the rows equal weight so they are allotted equal size
			container.grid_rowconfigure(i, weight=1)
		# you need to give at least one row and one column a positive weight 
		# https://stackoverflow.com/a/36507919/190597 (Bryan Oakley)
		container.grid_columnconfigure(0, weight=1)


def recv_sock_data(queue, gui_client):
	while True:
		recv_data = gui_client.receive_data_pack()
		# with queue.mutex:
		queue.queue.clear()
		queue.put(recv_data)

HOST = "192.168.1.95"
try:
	gui_client = GUISocketClient(HOST)
except ConnectionRefusedError:
	print("No server found")
	exit()

q = queue.Queue()
_sentinel = object()

fig_names_list = ["joint torques", 
				  "stroke plane"]

data_class_dict = {"joint torques": GUIDataClass("joint torques", 6),
				   "stroke plane" : GUIDataClass("stroke plane", 1)}

figs_list = [NewMPLFigure("joint torques", data_class_dict["joint torques"]),
			 NewMPLFigure("stroke plane", data_class_dict["stroke plane"])]

lines_sets = [fig.lines for fig in figs_list]
socket_thread = threading.Thread(target=recv_sock_data, args=(q, gui_client))	

def main():	

	app = App()
	app.geometry('1280x720')
	anis = [animation.FuncAnimation(fig.figure, animate, interval=50, fargs=[fig, q]) 
			for fig in figs_list]
	socket_thread.start()
	app.mainloop()

if __name__ == '__main__':
	main()
