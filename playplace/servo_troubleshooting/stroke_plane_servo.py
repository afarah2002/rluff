import RPi.GPIO as GPIO
import time
import numpy as np
import random 

class Servo(object):

	def __init__(self, pin_num, servo_name, pwm_frq=50, pwm_start=2.5):
		self.pin_num = pin_num
		self.servo_name = servo_name
		self.pwm_frq = pwm_frq
		self.pwm_start = pwm_start
		self.prev_angle = random.uniform(0., 260.)
		print(self.prev_angle)
		GPIO.setmode(GPIO.BCM)
		GPIO.setup(self.pin_num, GPIO.OUT)
		self.pin = GPIO.PWM(self.pin_num, self.pwm_frq)
		self.pin.start(pwm_start)
		self.reset()
		
	def reset(self):
		# Sets servo to the zero position
		self.turn_to_angle(0)
		time.sleep(1.)

	def turn_to_angle(self, angle):
		# Given angle in deg, turn servo to that angle
		print("Angle of " + self.servo_name + ": " + str(angle))
		duty_cycle_from_angle = self.pwm_start + (angle/27)
		self.pin.ChangeDutyCycle(duty_cycle_from_angle)

		# time.sleep(np.abs(self.prev_angle - angle)/27)
		# time.sleep(0.01)
		self.prev_angle = angle


	def close(self):
		self.pin.stop()
		GPIO.cleanup()


if __name__ == '__main__':
	stroke_plane_servo = Servo(17, "Stroke plane servo")
	
	for angle in list(np.arange(0, 280, 10)):
		stroke_plane_servo.turn_to_angle(angle)

	# while True:
	# 	try: 
	# 		angle = float(input("Servo angle: "))
	# 		stroke_plane_servo.turn_to_angle(angle)
	# 	except ValueError:
	# 		print("Invalid input! NaN")


	stroke_plane_servo.close()