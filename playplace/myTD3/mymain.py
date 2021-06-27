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

sys.path.append("../servo_troubleshooting")
from servo_module import Servo

p.connect(p.DIRECT)

# matplotlib.use('wxAgg')

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment

if __name__ == '__main__':
	q = queue.Queue()
	_sentinel = object()
	servo_1 = Servo(17, "stroke plane servo")

	options_object = myoptions.Options()

	data_storage = myplottingutils.MyDataClass(options_object.action_dim)
	plotter = myplottingutils.MyPlotClass(data_storage)

	t1 = threading.Thread(target=options_object.raw_TD3, args=(q,))
	# t1 = threading.Thread(target=options_object.brute_force, args=(q,))
	t2 = threading.Thread(target=options_object.consumer, 
						  args=(q, _sentinel, data_storage, servo_1))
	t1.start()
	# time.sleep(1.e-3)
	t2.start()
	# plt.show()
	