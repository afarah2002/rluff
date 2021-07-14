import numpy as np
import matplotlib.pyplot as plt
from symfit import parameters, variables, sin, cos, Fit

class DataProcessor:

	def read_data(filename):
		data_array = []
		with open(filename, 'r') as f:
			line_count = 0
			for line in f:
				line_count += 1
				if line_count % 2 == 0:
					line_list = [float(i) for i in list(line.split())]
					data_array.append(line_list)

		return np.array(data_array)

	def action_plotter(action_data_array):
		t = action_data_array[0]
		actions = action_data_array[1:]

		fig, ax = plt.subplots()
		for joint in actions:
		# joint = actions[5]
			ax.plot(t[-100:-1], joint[-100:-1])


def fourier_series(x, f, n=0):
    """
    Returns a symbolic fourier series of order `n`.

    :param n: Order of the fourier series.
    :param x: Independent variable
    :param f: Frequency of the fourier series
    """
    # Make the parameter objects for all the terms
    a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
    # Construct the series
    series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
                     for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    return series

if __name__ == '__main__':
	action_file = "action_data/raw_TD3/test_2.txt"
	action_data_array = DataProcessor.read_data(action_file).T
	# print(action_data_array)
	# DataProcessor.action_plotter(action_data_array)
	# plt.show()
	x, y = variables('x, y')
	w, = parameters('w')
	model_dict = {y: fourier_series(x, f=w, n=5)}
	print(model_dict)

	# xdata = action_data_array[0][-100:-1]
	xdata = np.linspace(-np.pi, np.pi, 200)
	ydata = action_data_array[2][-201:-1]
	# Make step function data
	# xdata = np.linspace(-np.pi, np.pi)
	# ydata = np.zeros_like(xdata)
	# ydata[xdata > 0] = 1
	# ydata[xdata > 0] = 1
	# Define a Fit object for this model and data
	fit = Fit(model_dict, x=xdata, y=ydata)
	fit_result = fit.execute()
	print(fit_result)

	# Plot the result
	plt.plot(xdata, ydata)
	plt.plot(xdata, fit.model(x=xdata, **fit_result.params).y, ls=':')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show()