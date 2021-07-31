from __future__ import print_function

import odrive
from odrive.enums import *
from odrive.utils import *
import time
import math
import numpy as np

class ODrive(object):

	def __init__(self):
		self.odrv0 = odrive.find_any()
		self.axes = [self.odrv0.axis0,
					 self.odrv0.axis1]
		# self.tune(1)
		self.calibrate(1)

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
		return axis.encoder.shadow_count

	def read_angle(self, axis_num, zero):
		raw_shadow_count = self.read_shadow_pos(axis_num)
		angle = (raw_shadow_count/8192.)*360 - zero
		return angle

	def find_zero(self, axis_num):
		print("Move Axis" + str(axis_num) + "to zero position")
		while True:
			enter = input("Press enter after zero is found")
			if enter == "":
				break
		zero = self.read_angle(axis_num, 0)
		print("This is the new zero: ", zero)
		return zero

def main():
	od = ODrive()
	motor = 1
	trq_val = 2.5
	zero = od.find_zero(motor)
	while True:
		pos = od.read_angle(motor, zero)
		if pos > 90:
			trq = -trq_val
		if pos < -90:
			trq = trq_val
		try:
			od.turn_trq(motor, trq)
			print(trq, pos)
		except UnboundLocalError:
			od.turn_trq(motor, trq_val)
		time.sleep(.01)

if __name__ == '__main__':
	main()


