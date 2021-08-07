import time
import numpy as np
import random

import odrive
from odrive.enums import *
from odrive.utils import *

import pi.interact.act as act
import pi.interact.observe as observe

class Threads:

	def recv_actions_main(client, action_queue):
		# The PC is the client
		while 1:
			action_data_pack = client.receive_data_pack()
			print("Received:", action_data_pack)
			action_queue.put(action_data_pack)

	def act_n_obs_main(action_queue, 
					   action_state_combo_queue,
					   pc_client,
					   pi_server):

		client = pc_client
		server = pi_server

		# Action setup
		# stroke_plane_motor = act.StrokePlane([17,27,22,23])
		
		motors = [0] # one motor
		# motors = [0,1] # both motors
		wing_torque_motor = act.WingTorque(motors)

		# State setup
		odrive = wing_torque_motor.odrive
		wing_observer = observe.Wings(odrive, motors)

		while 1:
			# action_data = action_queue.get()
			dump_errors(odrive.odrv0, True)
			action_data = client.receive_data_pack()
			if action_data:
				# Turn motors here
				# print("Acting out", action_data)
				# Observe next state

				# There is a time lag for the motors to move, 
				# Do not run the motors in a separate thread

				# Actions
				wing_torque = float(action_data["Wing torques"][0])
				action_num = int(action_data["Action num"])

				# Act
				wing_torque_motor.turn(wing_torque)

				# Measure torque AFTER acting
				motor_currents = wing_observer.mot_cur()
				action_data["Observed torques"] = motor_currents

				# States
				time_step_name = "Time"
				time_step = action_data[time_step_name]

				state_1_name = "Wing angles"
				state_1 = wing_observer.ang_pos()

				state_2_name = "Angular velocity"
				state_2 = wing_observer.ang_vel()

				state_data = {time_step_name : time_step,
							  state_1_name : state_1,
							  state_2_name : state_2}

				rand_done = np.random.rand(1)
				if rand_done > 0.95: # 5% chance (for now)
					done = True
				else:
					done = False

				done_data = {time_step_name : time_step,
							 "done" : done}

				combo_data_pack = {"action" : action_data, 
								   "next state" : state_data,
								   "done" : done_data
								   }

				print(combo_data_pack)
				print("\n")


				action_state_combo_queue.put(combo_data_pack)
				# pi_server.send_data_pack(combo_data_pack)

			else:
				pass


	def send_combos_main(server, action_state_combo_queue):
		# The pi is the server
		while 1:
			combo_data_pack = action_state_combo_queue.get()
			if combo_data_pack:
				server.send_data_pack(combo_data_pack)
			else:
				pass

