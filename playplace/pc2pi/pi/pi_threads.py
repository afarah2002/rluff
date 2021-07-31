import time
import numpy as np
import random

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
		stroke_plane_motor = act.StrokePlane([17,27,22,23])
		
		motors = [1] # one motor
		# motors = [0,1] # both motors
		wing_torque_motor = act.WingTorque(motors)

		# State setup
		odrive = wing_torque_motor.odrive
		wing_observer = observe.Wings(odrive, motors)

		while 1:
			# action_data = action_queue.get()
			action_data = client.receive_data_pack()
			if action_data:
				# Turn motors here
				# print("Acting out", action_data)
				# Observe next state

				# There is a time lag for the motors to move, 
				# Do not run the motors in a separate thread

				# Actions
				wing_torque = float(action_data["Wing torques"][0])
				stroke_plane_d_theta = float(action_data["Stroke plane angle"][0])
				stroke_plane_speed_pct = float(action_data["Stroke plane speed"][0])

				# Act
				stroke_plane_motor.turn(stroke_plane_d_theta,
										stroke_plane_speed_pct)

				wing_torque_motor.turn(wing_torque)

				# States
				time_step_name = "Time"
				time_step = action_data[time_step_name]

				state_1_name = "IMU Readings"
				state_1 = list(np.random.rand(6))

				state_2_name = "Wing angles"
				# state_2 = list(np.random.rand(1))
				state_2 = wing_observer.ang_pos()

				state_data = {time_step_name : time_step,
							  state_1_name : state_1,
							  state_2_name : state_2}

				reward_data = {time_step_name : time_step,
							   "Reward" : [50 - 100*float(np.random.rand(1))]}

				# done_data = {time_step_name : time_step,
				# 			 "done" : random.choice([True, False])}

				done_data = {time_step_name : time_step,
							 "done" : False}


				combo_data_pack = {"action" : action_data, 
								   "next state" : state_data,
								   "reward" : reward_data,
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

