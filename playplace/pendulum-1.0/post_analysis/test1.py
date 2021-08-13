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

def main():
	data_dir = "041_200"
	fig_name = "Episode reward"

	x = read_raw_data(data_dir, fig_name, "X")
	y = read_raw_data(data_dir, fig_name, "Y")

	fig, ax = plt.subplots()
	ax.grid()
	ax.plot(x, y)
	plt.show()

if __name__ == '__main__':
	main()