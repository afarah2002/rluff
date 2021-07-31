from __future__ import print_function

import odrive
from odrive.enums import *
from odrive.utils import *
import time
import math
import numpy as np

class ODrive(object):

	def __init__(self, motors):
		self.odrv0 = odrive.find_any()
		self.axes = [self.odrv0.axis0,
					 self.odrv0.axis1]
		for m in motors:
			self.calibrate(m)

	def tune(self):
		self.odrv0.save_configuration()

	def calibrate(self, axis_num):
		print("Calibrating...")
		axis = self.axes[axis_num]
		axis.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
		while axis.current_state != AXIS_STATE_IDLE:
			time.sleep(0.01)
		self.turn_pos(axis_num, axis.encoder.pos_estimate)

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
		axis.controller.config.control_mode = CONTROL_MODE_TORQUE_CONTROL
		axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
		axis.controller.input_torque = trq

	def read_shadow_pos(self, axis_num):
		axis = self.axes[axis_num]
		raw_shadow = axis.encoder.shadow_count
		angle = (raw_shadow/8192.) * 360
		return angle

	# def read_rpm