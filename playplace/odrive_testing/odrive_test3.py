from __future__ import print_function

import odrive
from odrive.enums import *
from odrive.utils import *
import time
import math
import numpy as np

odrv0 = odrive.find_any()
# odrv0.config.dc_max_negative_current = -0.1

axes = [odrv0.axis1]
# axes = [odrv0.axis0, odrv0.axis1]

# for axis in axes:
# 	axis.motor.config.motor_type = MOTOR_TYPE_HIGH_CURRENT
# 	# axis.motor.config.motor_type = MOTOR_TYPE_GIMBAL
# 	# axis.motor.config.current_lim = 24 # means voltage for gimbal motors
# 	axis.controller.config.vel_limit = 20
# 	axis.motor.config.calibration_current = 4 # means voltage for gimbal motors
# 	axis.motor.config.pole_pairs = 12
# 	axis.encoder.config.cpr = 8192

# odrv0.save_configuration()
# odrv0.reboot()

for axis in axes:
	axis.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
	print("calibrating axis")
	while axis.current_state != AXIS_STATE_IDLE:
		time.sleep(0.01)

#-----------------NO ERROR AT THIS POINT-----------------#

for axis in axes:
	axis.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
	print("velocity control mode on")


# print(odrv0.config.dc_max_negative_current, "\n", 
# 	  odrv0.config.max_regen_current)
odrv0.axis1.controller.config.pos_gain = 20.0
odrv0.axis1.controller.config.vel_gain = 0.16
odrv0.axis1.controller.config.vel_integrator_gain = 0

print("\n\n")
dump_errors(odrv0, True)
print("\n\n")


for axis in axes:
	axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
	print("Close loop on")

print("\n\n")
dump_errors(odrv0, True)
print("\n\n")

# print(odrv0.axis0.controller.config.vel_integrator_gain)
# while True:
# 	odrv0.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
# 	odrv0.axis0.controller.input_vel = 10
# 	print(odrv0.axis0.current_state)
# time.sleep(1)

# print("\n\n")
# dump_errors(odrv0, True)
# print("\n\n")

# t0 = time.monotonic()
# while True:
#     setpoint = 8192 * math.sin((time.monotonic() - t0)*2)
#     # setpoint = input("Position: ")
#     odrv0.axis1.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
#     odrv0.axis1.controller.input_pos = setpoint
#     # while odrv0.axis0.current_state != AXIS_STATE_IDLE:
#     # 	time.sleep(0.01)
#     dump_errors(odrv0, True)
#     time.sleep(0.01)
    # obs_sp = odrv0.axis0.controller.pos_setpoint 
    # print(int(setpoint), "   ", obs_sp)


# ts = np.arange(0,20*np.pi,1000)
# ts = np.arange(25,50,1)

# for t in ts:
# 	# setpoint = float(np.sin(t))
# 	setpoint = t
# 	print("goto " + str(int(setpoint)))
# 	odrv0.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
# 	odrv0.axis0.controller.input_vel = setpoint
# 	time.sleep(1)


# for axis in axes:
# 	axis.controller.config.control_mode = CONTROL_MODE_POSITION_CONTROL

# print(odrv0.axis0.controller.config.control_mode, "\n",
# 	  odrv0.axis0.controller.config.input_mode, "\n", 
# 	  odrv0.axis1.controller.config.control_mode, "\n",
# 	  odrv0.axis1.controller.config.input_mode, "\n")

# pos = int(input("position: "))
# odrv0.axis0.controller.input_pos = pos

# print("\n\n")
# dump_errors(odrv0, True)
# print("\n\n")

# t0 = time.monotonic()
# while True:
# 	setpoint = 10000.0 * math.sin((time.monotonic() - t0)*2)
# 	# setpoint = input("Position: ")
# 	print("goto " + str(int(setpoint)))
# 	odrv0.axis0.controller.input_pos = setpoint
# 	odrv0.axis1.controller.input_pos = setpoint
# 	dump_errors(odrv0, True)
# 	time.sleep(0.1)

# while True:
# 	for axis in axes:
# 		input_pos = int(input("position: "))
# 		axis.controller.input_pos = input_pos
# 		print(axis.controller.pos_setpoint)
# 		while axis.current_state != AXIS_STATE_IDLE:
# 			time.sleep(0.01)
# 		dump_errors(odrv0, True)
