import RPi.GPIO as GPIO
import time 
import numpy as np
import random

class Servo(object):
	'''
	reset: resets the servo to 0
	turn_to_angle: turns to a given angle at the maximum speed
		(make sure to set a long enough delay)
	turn_with_speed: turns to a given angle at a given speed 
		(percentage of the maximum speed)
	'''

	def __init__(self, pin_num, servo_name, pwm_frq=50, pwm_start=2.5):
		self.pin_num = pin_num
		self.servo_name = servo_name
		self.pwm_frq = pwm_frq
		self.pwm_start = pwm_start

		self.angles_list = [0.]
		self.max_speed = 100.

		GPIO.setmode(GPIO.BCM)
		GPIO.setup(self.pin_num, GPIO.OUT)
		self.pin = GPIO.PWM(self.pin_num, self.pwm_frq)
		self.pin.start(pwm_start)
		self.reset()

	def reset(self):
		angle = 0.
		delay = 2.
		self.turn_to_angle(angle, delay)

	def turn_to_angle(self, angle, delay):
		# Given angle in deg, turn servo to that angle
		# print("Angle of " + self.servo_name + ": " + str(angle), "\n --------------------")
		duty_cycle_from_angle = self.pwm_start + (angle/27)
		self.pin.ChangeDutyCycle(duty_cycle_from_angle)
		time.sleep(delay)

	def turn_with_speed(self, angle, speed):
		speed *= 3. # scaling to make percentages work :P
		self.angles_list.append(angle)
		D_theta = np.abs(self.angles_list[-1] - self.angles_list[-2])
		bounded_list = np.linspace(self.angles_list[-2], 
								   self.angles_list[-1],
								   int(D_theta*self.max_speed/speed))

		d_theta = 0.1
		for i in range(len(bounded_list)):
			if i > 0:
				d_theta = np.abs(bounded_list[i] - bounded_list[i-1])
			delay = d_theta/speed
			self.turn_to_angle(bounded_list[i], delay)


	def close(self):
		self.pin.stop()
		GPIO.cleanup()

# if __name__ == '__main__':
# 	servo_1 = Servo(17, "stroke plane servo")
# 	while True:
# 		angle = float(input("Angle: "))
# 		speed = float(input("Speed: "))
# 		servo_1.turn_with_speed(angle, speed)