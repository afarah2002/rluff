# from __future__ import print_function

# import odrive
# from odrive.enums import *
# import time
# import math

# print ("finding an odrive...")
# odrv0 = odrive.find_any()
# print("found", odrv0)

# def config(odrv0):
# 	odrv0.config.dc_max_negative_current = -0.01
# 	axes = [odrv0.axis0,
# 			odrv0.axis1]

# 	for axis in axes:
# 		axis.motor.config.current_lim = 10
# 		axis.controller.config.vel_limit = 10
# 		axis.motor.config.calibration_current = 5
# 		axis.motor.config.pole_pairs = 12
# 		axis.motor.config.motor_type = MOTOR_TYPE_GIMBAL
# 		axis.encoder.config.cpr = 8192

# 	odrv0.save_configuration()

# 	for axis in axes:
# 		axis.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE

# 	odrive.dump_errors(odrv0)


# if __name__ == '__main__':
# 	config(odrv0)

"""
Example usage of the ODrive python library to monitor and control ODrive devices
"""

from __future__ import print_function

import odrive
from odrive.enums import *
from odrive.utils import *
import time
import math

# Find a connected ODrive (this will block until you connect one)
print("finding an odrive...")
my_drive = odrive.find_any()

# Find an ODrive that is connected on the serial port /dev/ttyUSB0
#my_drive = odrive.find_any("serial:/dev/ttyUSB0")

# Calibrate motor and wait for it to finish
print("starting calibration...")
my_drive.axis0.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
while my_drive.axis0.current_state != AXIS_STATE_IDLE:
    time.sleep(0.1)
# my_drive.axis1.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
# while my_drive.axis1.current_state != AXIS_STATE_IDLE:
#     time.sleep(0.1)



my_drive.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
# my_drive.axis1.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL

dump_errors(my_drive)
# time.sleep(10)

# To read a value, simply read the property
print("Bus voltage is " + str(my_drive.vbus_voltage) + "V")

# Or to change a value, just assign to the property
my_drive.axis0.controller.input_pos = 8000
print("Position setpoint is " + str(my_drive.axis0.controller.input_pos))

# And this is how function calls are done:
for i in [1,2,3,4]:
    print('voltage on GPIO{} is {} Volt'.format(i, my_drive.get_adc_voltage(i)))

# A sine wave to test
t0 = time.monotonic()
while True:
    setpoint = 10000.0 * math.sin((time.monotonic() - t0)*2)
    # setpoint = input("Position: ")
    print("goto " + str(int(setpoint)))
    my_drive.axis0.controller.input_pos = setpoint
    dump_errors(my_drive, True)
    time.sleep(0.01)

# Some more things you can try:

# Write to a read-only property:
my_drive.vbus_voltage = 11.0  # fails with `AttributeError: can't set attribute`

# Assign an incompatible value:
my_drive.motor0.input_pos = "I like trains"  # fails with `ValueError: could not convert string to float`