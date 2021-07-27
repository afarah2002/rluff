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
		self.calibrate(1)

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
		return axis.encoder.shadow_count

def main():
	od = ODrive()
	motor = 1
	trq_val = 2.5
	while True:
		pos = od.read_shadow_pos(motor)
		if pos > 4096:
			trq = -trq_val
		if pos < 0:
			trq = trq_val
		try:
			od.turn_trq(motor, trq)
			print(trq, pos)
		except UnboundLocalError:
			od.turn_trq(motor, trq_val)
		time.sleep(.01)

if __name__ == '__main__':
	main()


