import time
import numpy as np
import random

import odrive
from odrive.enums import *
from odrive.utils import *

import pi.interact.act as act
import pi.interact.observe as observe

motors = [0] # one motor
# motors = [0,1] # both motors
wing_torque_motor = act.WingTorque(motors)
odrive_mod = wing_torque_motor.odrive

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

		# State setup
		wing_observer = observe.Wings(odrive_mod, motors)
		FAILSAFE_BOOL = False
		resume_timestep = 0

		# Use if action is d_tau
		wing_torque = 0

		while 1:
			# action_data = action_queue.get()
			dump_errors(odrive_mod.odrv0, True)
			action_data = client.receive_data_pack()
			if action_data:
				# Turn motors here
				# print("Acting out", action_data)
				# Observe next state

				# There is a time lag for the motors to move, 
				# Do not run the motors in a separate thread

				time_step_name = "Time"
				time_step = action_data[time_step_name]

				# Actions
				wing_torque = float(action_data["Wing torques"][0])
				# action_data["Wing torques"].append(wing_torque)

				# if abs(wing_torque) > 0.05:
				# 	wing_torque = 0.05*wing_torque/abs(wing_torque) # Set to max
				# 	action_data["Wing torques"][0] = 0 # Set dT to 0

				# --- SPEED FAILSAFE - KEEPS BREAKING THE PENDULUM!!!
				current_speed = abs(wing_observer.ang_vel()[0])
				if current_speed == 0:
					current_speed = 0.00001
				current_angle = abs(wing_observer.ang_pos()[0])
				vel_upper_bound = 1000 # deg/s
				ang_upper_bound = 90
				wait_period = 300 # timesteps

				# if current_speed >= vel_upper_bound:
				if current_angle >= ang_upper_bound:
					print("\n\n\n\n FAILSAFE REACHED \n\n\n\n")
					FAILSAFE_BOOL = True
					resume_timestep = time_step + wait_period

				if FAILSAFE_BOOL:
					print("Slowing down.........")
					# Stop the motor from adding more torque
					# wing_torque = -0.01*wing_observer.ang_vel()[0]/current_speed
					wing_torque = 0 
					# Update action data w/ 0 torque
					action_data["Wing torques"] = [0] #<--- NO! DON'T REWARD THE AI FOR THIS!!!
					for m in motors:
						odrive_mod.turn_pos(m, 0)

				# if time_step > resume_timestep:
				if current_speed < 10 and current_angle < 10:
					FAILSAFE_BOOL = False				
				# --- --- --- --- --- --- --- --- --- --- --- --- --- 

				# Act
				print(f"Wing torque: {wing_torque}")
				wing_torque_motor.turn(wing_torque)

				# Measure torque AFTER acting
				measured_torque = wing_observer.mot_trq()
				action_data["Wing torques"].append(measured_torque[0])
				action_data["Observed torques"] = measured_torque
				# States

				state_1_name = "Wing angles"
				state_1 = wing_observer.ang_pos()

				state_2_name = "Angular velocity"
				state_2 = [0 for m in motors]

				state_3_name = "Real time"
				state_3 = [time.time()]

				state_data = {time_step_name : time_step,
							  state_1_name : state_1,
							  state_2_name : state_2,
							  state_3_name : state_3}

				combo_data_pack = {"action" : action_data, 
								   "next state" : state_data,
								   }

				print(f"Timestep: {time_step}")
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


