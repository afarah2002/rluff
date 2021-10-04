import pickle
import matplotlib.pyplot as plt
import numpy as np

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
		# y = y/np.max(y)

		# for y_arr, label in zip(y, labels):
		ax.plot(x, y, label=labels)

	plt.legend(loc="center left")
	plt.show()

def plot_reward(data_dir):
	fig, ax = plt.subplots()
	ax.grid()
	x = list(read_raw_data(data_dir, "Episode reward", "X"))
	y = list(read_raw_data(data_dir, "Episode reward", "Y"))
	ax.plot(x,y)

def torque_analysis(data_dir, bounds):
	x = list(read_raw_data(data_dir, "Wing torques", "X"))
	y = list(read_raw_data(data_dir, "Wing torques", "Y"))

	avg_torque = np.mean(np.absolute(y[bounds[0]:bounds[1]]), axis=0)[1]
	print("Average torque", avg_torque)



def compare_with_theory(data_dir, bounds):
	fig_names = ["Wing angles", "Angular velocity"]
	for fig_name in fig_names:
		x = list(read_raw_data(data_dir, fig_name, "X"))
		y = list(read_raw_data(data_dir, fig_name, "Y"))
		# print(y)

		theory_vel = int(data_dir.split("_")[1])
		theory_ang_amp = 1.5708*(0.37/9.8)**(0.5)*theory_vel

		theory_dict = {"Angular velocity" : theory_vel,
					   "Wing angles" : theory_ang_amp}

		avg_val = np.mean(y[bounds[0]:bounds[1]], axis=0)[1]

		print(fig_name + " theory:", theory_dict[fig_name])
		print(fig_name + " achieved:", avg_val)

	torque_analysis(data_dir, bounds)



def main():
	# 107_127
	# data_dir = "120_100"
	data_dir = "127_90"
	# fig_name = "Episode reward"
	fig_name = "Episode reward"
	bounds = [300*100,390*100]

	compare_with_theory(data_dir, bounds)
	plot_reward(data_dir)
	plot_data(data_dir)


if __name__ == '__main__':
	main()