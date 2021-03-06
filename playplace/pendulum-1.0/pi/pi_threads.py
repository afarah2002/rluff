import time
import numpy as np
import array
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
wing_observer = observe.Wings(odrive_mod, motors)

class ConstantDataMonitor(object):

	def __init__(self):

		# Master data lists	1
		self.action = 0 # prescribed torque
		self.timestep = 0

		self.real_time = np.zeros(1)
		self.timesteps = array.array('i')
		# self.wing_torques = np.zeros((1,2)) # Real and observed
		self.wing_angles = np.zeros(1) 

	def update_action(self, timestep, action):
		# Torque from most recent action is constantly 
		# applied, this must be shown in data monitor
		self.action = action # Timestep, prescribed torque, observed torque
		self.timestep = timestep
		print(self.timestep)

	def constant_monitor_thread_main(self):
		while 1:
			time.sleep(0.01) # Resolution used in sims

			real_time = time.time()
			obs_trq = wing_observer.mot_trq()
			wing_angles = wing_observer.ang_pos()

			self.real_time = np.append(self.real_time, real_time)
			self.timesteps = np.append(self.timesteps, int(self.timestep))
			# self.wing_torques = np.append(self.wing_torques, [self.action])
			self.wing_angles = np.append(self.wing_angles, wing_angles)

	def retrieve_subset(self):
		print("Getting subsets")
		# wing_torques_subset = []

		indices = [i for i,value in enumerate(self.timesteps) if value == int(self.timestep)]
		min_index = min(indices)
		max_index = max(indices)

		real_time_subset = list(self.real_time[max_index-20:max_index])
		# print(real_time_subset)
		# wing_torques_subset.append(list(self.wing_torques[min_index:max_index]))
		wing_angles_subset = list(self.wing_angles[max_index-20:max_index])

		subset_data_pack = {"Real time" : real_time_subset,
							"Wing angles" : wing_angles_subset,
							# "Wing torques" : wing_torques_subset
							}

		return subset_data_pack
		

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
					   pi_server,
					   CDM
					   ):

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
				subsets = CDM.retrieve_subset()
				CDM.update_action(time_step, wing_torque)
				# print(wing_torque)
				# action_data["Wing torques"].append(wing_torque)

				# if abs(wing_torque) > 0.05:
				# 	wing_torque = 0.05*wing_torque/abs(wing_torque) # Set to max
				# 	action_data["Wing torques"][0] = 0 # Set dT to 0

				action_num = int(action_data["Action num"])

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
					# wing_torque = 0 
					# Update action data w/ 0 torque
					wing_torque = -(current_angle/90)*0.05
					action_data["Wing torques"] = [wing_torque] #<--- NO! DON'T REWARD THE AI FOR THIS!!!
					# for m in motors:
					# 	odrive_mod.turn_pos(m, 0)

				# if time_step > resume_timestep:
				if current_speed < 10 and current_angle < 10:
					FAILSAFE_BOOL = False				
				# --- --- --- --- --- --- --- --- --- --- --- --- --- 

				# Act
				wing_torque_motor.turn(wing_torque)
				# CDM.update_action(int(time_step), wing_torque)

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
								   "subsets" : subsets}

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


