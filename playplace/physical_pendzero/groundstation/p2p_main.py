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

from server_test import SocketServer
sys.path.append("./myTD3_P2P")
import myTD3_P2P.myoptions as myoptions

p.connect(p.DIRECT)

def main():

	#-----------------set up queue-----------------#
	data_output_queue = queue.Queue()
	#-----------------set up NN options obj-----------------#
	options_object = myoptions.Options()
	#-----------------set up server (pi)-----------------#
	HOST = '192.168.1.192'
	pc_server = SocketServer(HOST)

	#-----------------algorithm selection-----------------#
	# ALGORITHM = "raw_TD3"
	ALGORITHM = "brute_force"
	t_raw_TD3 = threading.Thread(target=options_object.raw_TD3, args=(data_output_queue,))
	t_brute_force = threading.Thread(target=options_object.brute_force, args=(data_output_queue,))
	algorithm_dict = {"raw_TD3": t_raw_TD3, 
					  "brute_force": t_brute_force}
	t_alg = algorithm_dict[ALGORITHM]

	t_queue = threading.Thread(target=options_object.gs_server_queue,
							   args=(data_output_queue, pc_server))

	t_alg.start()
	t_queue.start()


if __name__ == '__main__':
	main()