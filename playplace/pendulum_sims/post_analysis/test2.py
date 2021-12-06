import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams.update({'font.size': 14})
mpl.rc('text', usetex=True)

def read_raw_data(test_num, # Name of dir in test_data/
				  fig_name, # Name of subdir in test_data/test_num
				  axis):	# X or Y
	# file_name = f"test_data/delay_analysis/{test_num}/{fig_name}/{axis.capitalize()}Data.txt" 
	file_name = f"test_data/{test_num}/{fig_name}/{axis.capitalize()}Data.txt" 

	with open(file_name, "rb") as file:
		data = pickle.load(file)

	return data

def custom_plotter(input_dir):
	times = list(read_raw_data(input_dir, "Wing torques", "X")[458200:459000])
	torques = list(read_raw_data(input_dir, "Wing torques", "Y")[458200:459000,0])
	raw_angles = list(read_raw_data(input_dir, "Wing angles", "Y")[458200:459000,0])
	ang_vels = list(read_raw_data(input_dir,"Angular velocity", "Y")[458200:459000,0])

	torques = torques/np.max(torques)
	raw_angles = raw_angles/np.max(raw_angles)
	ang_vels = ang_vels/np.max(ang_vels)
	difference = np.subtract(torques,ang_vels)

	plt.grid()
	# plt.plot([times, times],[torques, raw_angles])
	plt.plot(times,torques)
	plt.plot(times,ang_vels)
	plt.plot(times,difference)
	plt.show()

if __name__ == '__main__':
	target_vel = 90.0
	data_dir = f"037_{target_vel}"
	custom_plotter(data_dir)