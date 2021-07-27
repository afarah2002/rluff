from __future__ import print_function

import odrive
from odrive.enums import *
from odrive.utils import *
import time
import math
import numpy as np


def calibrate(odrv0):
	# calibrate
	print("Calibrating...")
	odrv0.axis1.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
	while odrv0.axis1.current_state != AXIS_STATE_IDLE:
		time.sleep(0.01)

def set_gains(odrv0):
	odrv0.axis1.controller.config.pos_gain = 20.
	odrv0.axis1.controller.config.vel_gain = 0.025
	odrv0.axis1.controller.config.vel_integrator_gain = 0
	odrv0.save_configuration()

def turn_basic(odrv0, val):
	odrv0.axis1.controller.config.control_mode = CONTROL_MODE_POSITION_CONTROL
	odrv0.axis1.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
	odrv0.axis1.controller.input_pos = val
	# while odrv0.axis1.current_state != AXIS_STATE_IDLE:
	# 	time.sleep(1)

def turn_basic_vel(odrv0, val):
	odrv0.axis1.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
	odrv0.axis1.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
	odrv0.axis1.controller.input_pos = val
	time.sleep(1)

def turn_basic_trq(odrv0, val):
	odrv0.axis1.controller.config.control_mode = CONTROL_MODE_TORQUE_CONTROL
	odrv0.axis1.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
	odrv0.axis1.controller.input_torque = val

def combo(odrv0, pos, trq):
	odrv0.axis1.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
	odrv0.axis1.controller.config.control_mode = CONTROL_MODE_POSITION_CONTROL
	odrv0.axis1.controller.input_pos = pos
	odrv0.axis1.controller.config.control_mode = CONTROL_MODE_TORQUE_CONTROL
	odrv0.axis1.controller.input_torque = val

def test1_pos():
	#-----------------THIS WORKS-----------------#
	odrv0 = odrive.find_any()
	# set_gains(odrv0)
	calibrate(odrv0)
	t_pos = np.sin(np.arange(0, 100*np.pi, 0.01))
	for t in t_pos:
		turn_basic(odrv0, t)
		# time.sleep(.01)
		print("Pos: ", t, odrv0.axis1.controller.pos_setpoint)

def test2():
	odrv0 = odrive.find_any()
	# set_gains(odrv0)
	calibrate(odrv0)
	while True:
		turns = float(input("Num of turns"))
		turn_basic(odrv0, turns)
		print("Pos: ", turns, odrv0.axis1.controller.pos_setpoint)
		# vel = float(input("Vel: "))
		# turn_basic_vel(odrv0, vel)
		# print(vel)

def test3_torque():
	odrv0 = odrive.find_any()
	# set_gains(odrv0)
	calibrate(odrv0)
	trqs = 2.0*np.sin(np.arange(0, 100*np.pi, 0.1))
	for t in trqs:
		turn_basic_trq(odrv0, t)
		print("Trq: ", t, odrv0.axis1.motor.current_control.Iq_measured)
		# time.sleep(0.01)
	# while True:

def test4_torque():
	odrv0 = odrive.find_any()
	# set_gains(odrv0)
	calibrate(odrv0)
	while True:
		trq = float(input("Trq: "))
		turn_basic_trq(odrv0, trq)
		print("Trq: ", trq, odrv0.axis1.motor.current_control.Iq_measured)

def test5_torque_restrict():
	odrv0 = odrive.find_any()
	# set_gains(odrv0)
	calibrate(odrv0)
	turn_basic_trq(odrv0, odrv0.axis1.encoder.pos_estimate)
	# trqs = 2.0*np.sin(np.arange(0, 100*np.pi, 0.1))
	# for t in trqs:
	# 	turn_basic_trq(odrv0, t)
	# 	print("Trq: ", t, odrv0.axis1.motor.current_control.Iq_measured)
	trq_val = 2.5
	while True:
		# pos = odrv0.axis1.controller.pos_setpoint
		pos = odrv0.axis1.encoder.shadow_count
		# if abs(pos-bound) < 1000:
		dump_errors(odrv0, True)
		if pos > 4096:
			trq = -trq_val
		if pos < 0:
			trq = trq_val
		try:
			print(pos, trq, odrv0.axis1.motor.current_control.Iq_measured)
			turn_basic_trq(odrv0, trq)
		except UnboundLocalError:
			print(pos, trq_val, odrv0.axis1.motor.current_control.Iq_measured)
			turn_basic_trq(odrv0, trq_val)
		time.sleep(.01)

if __name__ == '__main__':
	# test1_pos()
	# test4_torque()
	# test5_combo()
	test5_torque_restrict()