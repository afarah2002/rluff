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

import myoptions
import myplottingutils

p.connect(p.DIRECT)

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment

if __name__ == '__main__':
	q = queue.Queue()
	_sentinel = object()

	options_object = myoptions.Options()

	data_storage = myplottingutils.MyDataClass(options_object.action_dim)
	plotter = myplottingutils.MyPlotClass(data_storage)

	t1 = threading.Thread(target=options_object.raw_TD3, args=(q,))
	# t1 = threading.Thread(target=options_object.brute_force, args=(q,))
	t2 = threading.Thread(target=options_object.consumer, args=(q, _sentinel, data_storage))
	t1.start()
	t2.start()
	plt.show()
	