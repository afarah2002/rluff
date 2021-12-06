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


# def plot_data(data_dir):
# 	# fig_names = ["Wing angles", "Wing torques"]
# 	lines = {"Wing angles" : "Raw angles",
# 			 "Wing torques" : "Prescribed torques",
# 			 "Angular velocity" : ["Raw ang vel", "Avg ang vel"]}

# 	fig, ax = plt.subplots()
# 	ax.grid()
# 	for fig_name, labels in lines.items():
# 		if fig_name == "Angular velocity":
# 			x = list(np.array(read_raw_data(data_dir, fig_name, "X")))
# 			y = list(np.array(read_raw_data(data_dir, fig_name, "Y")))
# 			y = y/np.max(y)
# 			linewidth=5.0
# 		else:
# 			x = list(np.array(read_raw_data(data_dir, fig_name, "X")))
# 			y = list(np.array(read_raw_data(data_dir, fig_name, "Y"))[:,0])
# 			y = y/np.max(y)
# 			linewidth=2.0

# 		# for y_arr, label in zip(y, labels):
# 		ax.plot(x, y, label=labels,linewidth=linewidth)

# 	ax.tick_params(axis='both', which='major', labelsize=40)
# 	ax.set_ylabel("Normalized reading", fontsize=40)
# 	ax.set_xlabel("Timestep", fontsize=40)

# 	plt.legend(bbox_to_anchor=(0.5, 1), loc='upper center', ncol=4, fontsize=30)
# 	plt.show()

def plot_reward(data_dir):
	fig, ax = plt.subplots()
	ax.grid()
	x = list(read_raw_data(data_dir, "Episode reward", "X"))
	y = list(read_raw_data(data_dir, "Episode reward", "Y"))
	print(len(x))
	ax.plot(x,y)
	ax.set_xlabel("Episode")
	ax.set_ylabel("Cumulative reward (maximum = 0)")
	ax.set_title("Episodic reward")
def torque_analysis(data_dir, bounds):
	x = list(read_raw_data(data_dir, "Wing torques", "X"))
	y = list(read_raw_data(data_dir, "Wing torques", "Y"))

	avg_torque = np.mean(np.absolute(y[bounds[0]:bounds[1]]), axis=0)
	# print("Average torque", avg_torque)



def compare_with_theory(data_dir, bounds):
	fig_names = ["Wing angles", "Angular velocity"]
	return_data_pack = []

	theory_vel = float(data_dir.split("_")[1])
	real_length = 0.37 # m
	length = 0.3
	theory_ang_amp = 1.5708*(length/9.8)**(0.5)*theory_vel # THIS IS AN AMPLITUDE

	theory_dict = {"Angular velocity" : theory_vel,
				   "Wing angles" : theory_ang_amp}

	achieved_results = {"Angular velocity" : None,
						"Wing angles" : None}

	for fig_name in fig_names:
		x = list(read_raw_data(data_dir, fig_name, "X"))
		y = list(read_raw_data(data_dir, fig_name, "Y"))
		print(len(y))

		if fig_name == "Wing angles":
			achieved_result = np.max(y[bounds[0]:])
		if fig_name == "Angular velocity":
			achieved_result = np.mean(y[bounds[0]:bounds[1]], axis=0)[1] # Ang vel average

		achieved_results[fig_name] = achieved_result

		print(fig_name + " theory:", theory_dict[fig_name])
		print(fig_name + " achieved:", achieved_result)


	torque_analysis(data_dir, bounds)
	return [theory_dict, achieved_results]


def reward_analysis(data_dir):
	episodes = list(read_raw_data(data_dir, "Episode reward", "X"))
	reward_data = list(read_raw_data(data_dir, "Episode reward", "Y"))
	
	ang_vels = list(np.array(read_raw_data(data_dir, "Angular velocity", "Y")).T[1,:])
	torques = list(np.array(read_raw_data(data_dir, "Wing torques", "Y")).T[1,:])

	# print(torques.shape)

	x = torques
	y = ang_vels

	target_x = 0
	target_y = 60



	plt.plot(x,y)
	# plt.scatter(target_x, target_y,'-ro')
	plt.show()
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

		data_dir = f"039_{target_vel}"
		# fig_name = "Episode reward"
		# fig_name = "Episode reward"
		# bounds = [299000,300000]
		bounds = [499000,500000]

		data_pack = compare_with_theory(data_dir, bounds)
		theory_angles.append(data_pack[0]["Wing angles"])
		achieved_angles.append(data_pack[1]["Wing angles"])
		target_ang_vels.append(data_pack[0]["Angular velocity"])
		achieved_ang_vels.append(data_pack[1]["Angular velocity"])

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

def sim():
	# 107_127
	# data_dir = "120_100"
	# data_dir = "014_90"
	target_ang_vels = []
	achieved_ang_vels = []
	theory_angles = []
	achieved_angles = []

	# fig, (ax1, ax2) = plt.subplots(1, 2)

	# for target_vel in np.arange(40,140,10.):
	target_vel = 90.0
	data_dir = f"037_{target_vel}"
	# fig_name = "Episode reward"
	# fig_name = "Episode reward"
	# bounds = [299000,300000]
	bounds = [458000,459000]

	# data_pack = compare_with_theory(data_dir, bounds)
	# theory_angles.append(data_pack[0])
	# achieved_angles.append(data_pack[1])
	# target_ang_vels.append(data_pack[2])
	# achieved_ang_vels.append(data_pack[3])

	plot_reward(data_dir)
	plot_data(data_dir)
	# reward_analysis(data_dir)
	plt.show()

	# ax1.grid()
	# ax1.set_xlabel("Angular velocity (deg/s)")
	# ax1.set_ylabel("Angle (deg)")
	# ax1.plot(target_ang_vels, theory_angles, label="Theory", color='b')
	# ax1.scatter(target_ang_vels, achieved_angles, label="Achieved ang vs target vel", color='g')
	# ax1.scatter(achieved_ang_vels, achieved_angles, label="Achieved ang vs achieved ang vel", color='r')
	# ax1.legend()

	# ax2.grid()
	# ax2.set_xlabel("Target angular velocity (deg/s)")
	# ax2.set_ylabel("Achieved angular velocity (deg/s)")
	# ax2.scatter(target_ang_vels, achieved_ang_vels, label="Achieved ang vel vs target ang vel", color='b')
	# ax2.legend()

	# plt.show()


if __name__ == '__main__':
	# sim_results_183_eps()
	sim()