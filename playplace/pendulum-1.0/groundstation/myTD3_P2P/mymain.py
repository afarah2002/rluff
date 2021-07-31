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

RPI_BOOL = True
PLT_BOOL = False
if RPI_BOOL:
	#-----------------SERVO-----------------#
	# sys.path.append("../servo_troubleshooting")
	# from servo_module import Servo
	# servo_1 = Servo(17, "stroke plane servo")

	#-----------------STEPPER MOTOR-----------------#
	sys.path.append("../stepper_troubleshooting")
	from stepper_module import StepperMotor
	GPIO_pins = [17,27,22,23]
	stroke_plane_motor = StepperMotor(GPIO_pins, CONT_BOOL=False)

else:
	servo_1 = None
	stroke_plane_motor = None

p.connect(p.DIRECT)

# matplotlib.use('wxAgg')

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment

if __name__ == '__main__':

	#-----------------pick algorithm here-----------------#
	# ALGORITHM = "raw_TD3"
	ALGORITHM = "brute_force"

	#-----------------set up queue stuff -----------------#
	q = queue.Queue()
	_sentinel = object()

	#-----------------alg and plot init-----------------#
	options_object = myoptions.Options(RENDER_BOOL=False)
	data_storage = myplottingutils.MyDataClass(options_object.action_dim)
	# plotter = myplottingutils.MyPlotClass(data_storage)

	#-----------------saver init-----------------#
	# filename_dict = {"raw_TD3": "raw_TD3/test_3.txt",
	# 				 "brute_force": "brute_force/test_3.txt"}
	# data_saver = myplottingutils.SaveData("action_data/" + filename_dict[ALGORITHM], 
	# 									  "state_data/" + filename_dict[ALGORITHM])

	#-----------------alg thread init-----------------#
	t_raw_TD3 = threading.Thread(target=options_object.raw_TD3, args=(q,))
	t_brute_force = threading.Thread(target=options_object.brute_force, args=(q,))
	algorithm_dict = {"raw_TD3": t_raw_TD3, 
					  "brute_force": t_brute_force}
	t_alg = algorithm_dict[ALGORITHM]

	sys.path.append("../data_monitor/pi")
	from server_test import GUISocketServer
	HOST = '192.168.1.95'
	# PORT = 50007
	gui_server = GUISocketServer(HOST)
	#-----------------consumer thread init-----------------#
	t_consumer = threading.Thread(target=options_object.consumer, 
								  args=(q, _sentinel, data_storage, RPI_BOOL, None, gui_server),
								  kwargs={'motor':stroke_plane_motor})

	#-----------------start threads-----------------#
	t_alg.start()
	t_consumer.start()

	#-----------------show plot? (if not on pi)-----------------#	
	if PLT_BOOL:
		plt.show()
	