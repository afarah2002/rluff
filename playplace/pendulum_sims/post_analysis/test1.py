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
	file_name = f"test_data/{test_num}/{fig_name}/{axis.capitalize()}Data.txt" 

	with open(file_name, "rb") as file:
		data = pickle.load(file)

	return data

def plot_data(data_dir):
	# fig_names = ["Wing angles", "Wing torques"]
	lines = {"Wing angles" : ["Raw angles", "Max angles"],
			 "Wing torques" : ["Prescribed torques", "Measured torques"],
			 "Angular velocity" : ["Raw ang vel", "Avg ang vel"]}

	fig, ax = plt.subplots()
	ax.grid()
	for fig_name, labels in lines.items():
		x = list(read_raw_data(data_dir, fig_name, "X"))
		y = list(read_raw_data(data_dir, fig_name, "Y"))
		y = y/np.max(y)

		# for y_arr, label in zip(y, labels):
		ax.plot(x, y, label=labels)

	plt.legend(loc="center left")
	plt.show()

def plot_reward(data_dir):
	fig, ax = plt.subplots()
	ax.grid()
	x = list(read_raw_data(data_dir, "Episode reward", "X"))
	y = list(read_raw_data(data_dir, "Episode reward", "Y"))
	print(len(x))
	ax.plot(x,y)

def torque_analysis(data_dir, bounds):
	x = list(read_raw_data(data_dir, "Wing torques", "X"))
	y = list(read_raw_data(data_dir, "Wing torques", "Y"))

	avg_torque = np.mean(np.absolute(y[bounds[0]:bounds[1]]), axis=0)[1]
	# print("Average torque", avg_torque)



def compare_with_theory(data_dir, bounds):
	fig_names = ["Wing angles", "Angular velocity"]
	return_data_pack = []
	for fig_name in fig_names:
		x = list(read_raw_data(data_dir, fig_name, "X"))
		y = list(read_raw_data(data_dir, fig_name, "Y"))
		# print(y)

		theory_vel = float(data_dir.split("_")[1])
		real_length = 0.37 # m
		length = 0.3
		theory_ang_amp = 1.5708*(length/9.8)**(0.5)*theory_vel

		theory_dict = {"Angular velocity" : theory_vel,
					   "Wing angles" : theory_ang_amp}

		avg_val = np.mean(y[bounds[0]:bounds[1]], axis=0)[1]

		# print(fig_name + " theory:", theory_dict[fig_name])
		# print(fig_name + " achieved:", avg_val)

		return_data_pack.extend([theory_dict[fig_name], avg_val])


	torque_analysis(data_dir, bounds)
	return return_data_pack



def sim_results_183_eps():
	# 107_127
	# data_dir = "120_100"
	# data_dir = "014_90"
	target_ang_vels = []
	achieved_ang_vels = []
	theory_angles = []
	achieved_angles = []

	fig, (ax1, ax2) = plt.subplots(1, 2)

	for target_vel in np.arange(40,140,10.):

		data_dir = f"031_{target_vel}"
		# fig_name = "Episode reward"
		# fig_name = "Episode reward"
		bounds = [181000,182000]

		data_pack = compare_with_theory(data_dir, bounds)
		theory_angles.append(data_pack[0])
		achieved_angles.append(data_pack[1])
		target_ang_vels.append(data_pack[2])
		achieved_ang_vels.append(data_pack[3])

		plot_reward(data_dir)
		# plot_data(data_dir)
		# plt.show()

	ax1.grid()
	ax1.set_xlabel("Angular velocity (deg/s)")
	ax1.set_ylabel("Angle (deg)")
	ax1.plot(target_ang_vels, theory_angles, label="Theory", color='b')
	ax1.scatter(target_ang_vels, achieved_angles, label="Achieved ang vs target vel", color='g')
	ax1.scatter(achieved_ang_vels, achieved_angles, label="Achieved ang vs achieved ang vel", color='r')
	ax1.legend()

	ax2.grid()
	ax2.set_xlabel("Target angular velocity (deg/s)")
	ax2.set_ylabel("Achieved angular velocity (deg/s)")
	ax2.scatter(target_ang_vels, achieved_ang_vels, label="Achieved ang vel vs target ang vel", color='b')
	ax2.legend()

	plt.show()


if __name__ == '__main__':
	sim_results_183_eps()