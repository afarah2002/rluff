from __future__ import print_function

import odrive
from odrive.enums import *
from odrive.utils import *
import time
import math
import numpy as np
from timeit import default_timer as timer

class ODrive(object):

	def __init__(self):
		self.odrv0 = odrive.find_any()
		self.axes = [self.odrv0.axis0,
					 self.odrv0.axis1]
		# self.tune(0)
		self.calibrate(0)
		dump_errors(self.odrv0, True)
		self.zero = 0
		self.find_zero(0)

	def tune(self, axis_num):
		'''
		Good combo 1: 2.5, 0.02, 0.
		Good combo 2: 2.5, 0.02, 0.5*(1/.5)*axis.controller.config.vel_gain

		'''
		axis = self.axes[axis_num]
		#-----------------defaults-----------------#
		# axis.controller.config.pos_gain = 20.0
		# axis.controller.config.vel_gain = 0.16
		# axis.controller.config.vel_integrator_gain = 0.32
		#------------------------------------------#
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

	def read_angle(self, axis_num):
		raw_shadow_count = self.read_shadow_pos(axis_num)
		angle = (raw_shadow_count/8192.)*360 - self.zero
		return angle

	def read_ang_vel(self, axis_num):
		axis = self.axes[axis_num]
		raw_ang_vel = axis.encoder.vel_estimate # turn/s
		ang_vel = raw_ang_vel*360 # deg/s
		return ang_vel

	def read_current(self, axis_num):
		axis = self.axes[axis_num]
		Iq_measured = axis.motor.current_control.Iq_measured # A
		return Iq_measured

	def read_trq(self, axis_num):
		trq_measured = self.read_current(axis_num)*8.27 # Nm
		return trq_measured

	def find_zero(self, axis_num):
		print("Move Axis" + str(axis_num) + "to zero position")
		while True:
			enter = input("Press enter after zero is found")
			if enter == "":
				break
		self.zero = self.read_angle(axis_num)
		print("This is the new zero: ", self.zero)


def main():
	od = ODrive()
	motor = 0
	trq_val = 2.5
	while True:
		# dump_errors(od.odrv0, True)
		pos = od.read_angle(motor)
		ang_vel = od.read_ang_vel(motor)
		Iq_current = od.read_current(motor)
		trq_measured = od.read_trq(motor)

		if pos > 90:
			trq = -trq_val
		if pos < -90:
			trq = trq_val

		try:
			od.turn_trq(motor, trq)
			print(trq, trq_measured)
		except UnboundLocalError:
			od.turn_trq(motor, trq_val)

		time.sleep(.01)

def test_pos():
	od = ODrive()
	motor = 0
	trq_val = 2.5
	pos = 0
	while True:
		dump_errors(od.odrv0, True)
		read_pos = od.read_angle(motor)
		ang_vel = od.read_ang_vel(motor)

		pos += 1

		try:
			od.turn_pos(motor, read_pos)
			print(pos, ang_vel)
		except UnboundLocalError:
			od.turn_pos(motor, pos)

		time.sleep(.01)

def test_angvel():
	od = ODrive()
	motor = 0
	trq_val = .1
	while True:
		dump_errors(od.odrv0, True)
		pos = od.read_angle(motor)
		ang_vel = od.read_ang_vel(motor)
		od.turn_trq(motor, trq_val)
		print(ang_vel)


if __name__ == '__main__':
	main()
	# test_angvel()
	# test_pos()


