import time 
import RPi.GPIO as GPIO
import numpy as np

from RpiMotorLib import RpiMotorLib

init_delay = 0.05

class StepperMotor(object):
	
	drive_conversions = {"half": 0.5,
						 "full": 1.0}

	def __init__(self, GPIO_pins, drive="half", CONT_BOOL=False, init_delay=0.000001):
		self.motor = RpiMotorLib.BYJMotor("stepper_motor", "28BYJ")
		self.GPIO_pins = GPIO_pins
		self.drive = drive
		# initial position is 0 (make sure to calibrate manually IRL!!)
		self.angles_list = [0] 
		self.CONTINUOUS_BOOL = CONT_BOOL 
		self.VERB_BOOL = False
		self.init_delay = init_delay

	def turn_to_angle(self, speed_pct, angle=None, d_angle=None):
		# Bring angle back between 0 and 360 if you don't want continuous rotation
		if type(angle) == float:
			main_angle = angle
		if type(d_angle) == float:
			main_angle = d_angle 

		if self.CONTINUOUS_BOOL:
			pass
		else: 
			while main_angle > 360:
				main_angle -= 360

		if type(angle) == float:
			self.angles_list.append(main_angle)
			angle_diff = self.angles_list[-1] - self.angles_list[-2]
		if type(d_angle) == float:
			angle_diff = main_angle

		# calculate the number of steps from the angle
		steps = abs(angle_diff)*512./360.

		# Delay (half vs full)
		min_delay = {True:0.0009, 
					 False:0.0013}

		delay = 0.0014*self.drive_conversions[self.drive]/(speed_pct/100)
		# print(min_delay[self.VERB_BOOL])
		# time.sleep(10)

		# If angle diff less than 0, turn clockwise
		if angle_diff >= 0:
			CCW_BOOL = True
		if angle_diff < 0:
			CCW_BOOL = False

		# Turn the motor
		self.motor.motor_run(self.GPIO_pins, 
							  delay,
							  steps, 
							  CCW_BOOL, 
							  self.VERB_BOOL, 
							  self.drive, 
							  self.init_delay)

if __name__ == '__main__':
	GPIO_pins = [17,27,22,23]
	stroke_plane_servo = StepperMotor(GPIO_pins, drive="half", CONT_BOOL=False)
	# speed_pct = 100
	# angle = 90
	# d_angle = 90
	# while True:
		# speed_pct = float(input("Speed percentage: "))
		# angle = float(input("Angle: "))
		# stroke_plane_servo.turn_to_angle(speed_pct, angle=angle)
		# d_angle = float(input("d_angle: "))
		# stroke_plane_servo.turn_to_angle(speed_pct, d_angle=d_angle)

	# try:
	# 	t_list = np.linspace(0,2*np.pi,100)
	# 	for t in t_list:
	# 		angle = float(180*np.sin(t))
	# 		print(angle)
	# 		speed_pct = float(40*np.cos(t) + 50.)
	# 		# speed_pct = 100

	# 		stroke_plane_servo.turn_to_angle(speed_pct, angle=angle)

	try:
		while True:
			stroke_plane_servo.turn_to_angle(100, d_angle=90.)

	except KeyboardInterrupt:
		exit()


	GPIO.cleanup()
