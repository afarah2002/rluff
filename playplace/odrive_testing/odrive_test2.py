import odrive
from odrive.enums import *
from odrive.utils import *
import time
import math

# Find a connected ODrive (this will block until you connect one)
print("finding an odrive...")
odrv0 = odrive.find_any()
# odrv0.erase_configuration()

print("\n\n")
dump_errors(odrv0, True)
print("\n\n")

odrv0.config.dc_max_negative_current = -0.01
odrv0.config.enable_brake_resistor = False



axes = [odrv0.axis0, odrv0.axis1]

for axis in axes:
	axis.motor.config.motor_type = MOTOR_TYPE_HIGH_CURRENT
	# axis.motor.config.motor_type = MOTOR_TYPE_GIMBAL
	axis.motor.config.current_lim = 10 # means voltage for gimbal motors
	axis.controller.config.vel_limit = 4000
	axis.motor.config.calibration_current = 5 # means voltage for gimbal motors
	axis.motor.config.pole_pairs = 12
	axis.encoder.config.cpr = 8192

	# axis.controller.config.pos_gain = 0
	# axis.controller.config.vel_gain = 0
	# axis.controller.config.vel_integrator_gain = 0

# odrv0.save_configuration()
# odrv0.reboot()

# time.sleep(1)
for axis in axes:
	print(axis.motor)

for axis in axes:
	axis.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
	print("calibrating axis")
	while axis.current_state != AXIS_STATE_IDLE:
		time.sleep(0.01)
	# axis.requested_state = AXIS_STATE_ENCODER_OFFSET_CALIBRATION
	# print("calibrating encoder")
	# while axis.current_state != AXIS_STATE_IDLE:
	# 	time.sleep(0.01)


print("\n\n")
dump_errors(odrv0, True)
print("\n\n")

# for axis in axes:
# 	# axis.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
# 	# print("velocity control mode on")
	# axis.controller.config.control_mode = INPUT_MODE_POS_FILTER

for axis in axes:
	axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
	print("Close loop on")


while True:
	for axis in axes:
		input_pos = int(input("position: "))
		axis.controller.input_pos = input_pos
		print(axis.controller.pos_setpoint)
		while axis.current_state != AXIS_STATE_IDLE:
			time.sleep(0.01)
		dump_errors(odrv0, True)
