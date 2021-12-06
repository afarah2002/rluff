# from __future__ import print_function

# import odrive
# from odrive.enums import *
# from odrive.utils import *
# import time
# import math
# import numpy as np

# class ODrive(object):

# 	def __init__(self, motors):
# 		self.odrv0 = odrive.find_any()
# 		self.axes = [self.odrv0.axis0,
# 					 self.odrv0.axis1]
# 		for m in motors:
# 			self.calibrate(m)

# 	def tune(self):
# 		self.odrv0.save_configuration()

# 	def calibrate(self, axis_num):
# 		print("Calibrating...")
# 		axis = self.axes[axis_num]
# 		axis.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
# 		while axis.current_state != AXIS_STATE_IDLE:
# 			time.sleep(0.01)
# 		self.turn_pos(axis_num, axis.encoder.pos_estimate)

# 	def turn_pos(self, axis_num, pos):
# 		axis = self.axes[axis_num]
# 		axis.controller.config.control_mode = CONTROL_MODE_POSITION_CONTROL
# 		axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
# 		axis.controller.input_pos = pos

# 	def turn_vel(self, axis_num, vel):
# 		axis = self.axes[axis_num]
# 		axis.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
# 		axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
# 		axis.controller.input_vel = vel

# 	def turn_trq(self, axis_num, trq):
# 		axis = self.axes[axis_num]
# 		axis.controller.config.control_mode = CONTROL_MODE_TORQUE_CONTROL
# 		axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
# 		axis.controller.input_torque = trq

# 	def read_shadow_pos(self, axis_num):
# 		axis = self.axes[axis_num]
# 		raw_shadow = axis.encoder.shadow_count
# 		angle = (raw_shadow/8192.) * 360
# 		return angle

# 	# def read_rpm

from __future__ import print_function

import odrive
from odrive.enums import *
from odrive.utils import *
import time
import math
from datetime import datetime
import numpy as np

class ODrive(object):

	def __init__(self, motors):
		self.odrv0 = odrive.find_any()
		self.axes = [self.odrv0.axis0,
					 self.odrv0.axis1]
		self.zeros = [0,0]
		for m in motors:
			# self.tune(m)
			calib_bool = input("Calibrate? Enter y/n")
			if calib_bool == "y":
				self.calibrate(m)
			else:
				pass
			self.find_zero(m)

	def tune(self, axis_num):
		'''
		Good combo 1: 2.5, 0.02, 0.
		Good combo 2: 2.5, 0.02, 0.5*(1/.5)*axis.controller.config.vel_gain

		'''
		axis = self.axes[axis_num]
		axis.controller.config.pos_gain = 2.5
		axis.controller.config.vel_gain = 0.02
		axis.controller.config.vel_integrator_gain = 0.5*(1/.5)*axis.controller.config.vel_gain
		self.odrv0.save_configuration()

	def calibrate(self, axis_num):
		print("Calibrating...")
		axis = self.axes[axis_num]
		axis.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
		while axis.current_state != AXIS_STATE_IDLE:
			time.sleep(0.01)
		# axis.controller.config.input_mode = INPUT_MODE_PASSTHROUGH
		# self.turn_pos(axis_num, axis.encoder.pos_estimate)

	def turn_pos(self, axis_num, pos):
		axis = self.axes[axis_num]
		axis.controller.config.control_mode = CONTROL_MODE_POSITION_CONTROL
		axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
		axis.controller.input_pos = pos

	def turn_vel(self, axis_num, vel):
		axis = self.axes[axis_num]
		axis.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
		axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
		axis.controller.input_vel = vel

	def turn_trq(self, axis_num, trq):
		axis = self.axes[axis_num]
		# axis.controller.input_mode = INPUT_MODE_TORQUE_RAMP
		axis.controller.config.control_mode = CONTROL_MODE_TORQUE_CONTROL
		axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
		axis.controller.input_torque = trq

	def read_shadow_pos(self, axis_num):
		axis = self.axes[axis_num]
		return axis.encoder.shadow_count

	def read_angle(self, axis_num):
		raw_shadow_count = self.read_shadow_pos(axis_num)
		angle = (raw_shadow_count/8192.)*360 - self.zeros[axis_num]
		return angle

	def read_ang_vel(self, axis_num):
		axis = self.axes[axis_num]
		raw_ang_vel = axis.encoder.vel_estimate # turn/s
		ang_vel = raw_ang_vel*360 # deg/s
		
		return ang_vel

	def read_measured_current(self, axis_num):
		axis = self.axes[axis_num]
		Iq_measured = axis.motor.current_control.Iq_measured # A
		return Iq_measured

	def read_setpoint_current(self, axis_num):
		axis = self.axes[axis_num]
		Iq_setpoint = axis.motor.current_control.Iq_setpoint # A
		return Iq_setpoint


	def read_trq(self, axis_num):
		trq_measured = self.read_measured_current(axis_num)*8.27/400. # Nm
		return trq_measured


	def find_zero(self, axis_num):
		print("Move Axis" + str(axis_num) + "to zero position")
		while True:
			enter = input("Press enter after zero is found")
			if enter == "":
				break
		self.zeros[axis_num] = self.read_angle(axis_num)
		print("This is the new zero: ", self.zeros[axis_num])
