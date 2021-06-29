import numpy as np
import torch
import gym
import argparse
import os
import pybullet as p
import pybullet_envs
import threading 
import queue 
import matplotlib.pyplot as plt
import matplotlib
import time
import sys

import myoptions
import myplottingutils

RPI_BOOL = False
PLT_BOOL = True
if RPI_BOOL:
	sys.path.append("../servo_troubleshooting")
	from servo_module import Servo
	servo_1 = Servo(17, "stroke plane servo")
else:
	servo_1 = None

p.connect(p.DIRECT)

# matplotlib.use('wxAgg')

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment

if __name__ == '__main__':

	#-----------------pick algorithm here-----------------#
	ALGORITHM = "raw_TD3"
	# ALGORITHM = "brute_force"

	#-----------------set up queue stuff -----------------#
	q = queue.Queue()
	_sentinel = object()

	#-----------------alg and plot init-----------------#
	options_object = myoptions.Options(RENDER_BOOL=True)
	data_storage = myplottingutils.MyDataClass(options_object.action_dim)
	plotter = myplottingutils.MyPlotClass(data_storage)

	#-----------------saver init-----------------#
	filename_dict = {"raw_TD3": "raw_TD3/test_2.txt",
					 "brute_force": "brute_force/test_2.txt"}
	data_saver = myplottingutils.SaveData("action_data/" + filename_dict[ALGORITHM], 
										  "state_data/" + filename_dict[ALGORITHM])

	#-----------------alg thread init-----------------#
	t_raw_TD3 = threading.Thread(target=options_object.raw_TD3, args=(q,))
	t_brute_force = threading.Thread(target=options_object.brute_force, args=(q,))
	algorithm_dict = {"raw_TD3": t_raw_TD3, 
					  "brute_force": t_brute_force}
	t_alg = algorithm_dict[ALGORITHM]

	#-----------------consumer thread init-----------------#
	t_consumer = threading.Thread(target=options_object.consumer, 
								  args=(q, _sentinel, data_storage, RPI_BOOL, data_saver),
								  kwargs={'servo':servo_1})

	#-----------------start threads-----------------#
	t_alg.start()
	t_consumer.start()

	#-----------------show plot? (if not on pi)-----------------#	
	if PLT_BOOL:
		plt.show()
	