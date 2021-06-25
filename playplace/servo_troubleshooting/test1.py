import RPi.GPIO as GPIO
import time
import numpy as np

servoPIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPIN, GPIO.OUT)

p = GPIO.PWM(servoPIN, 50) # GPIO 17 for PWM with 50Hz
p.start(2.5) # Initialization

def turn(pin, duty_cycle):
	pin.ChangeDutyCycle(duty_cycle)
	time.sleep(0.5)
	# print(duty_cycle)

def turn_to_angle(pin, angle):
	print(angle)
	duty_cycle_from_angle = 2.5 + (angle/27)
	turn(pin, duty_cycle_from_angle)

def reset(pin):
	turn_to_angle(pin, 0)
	

try:
	reset(p)
	angles_list = list(np.arange(0,280,10))
	for angle in angles_list:
		turn_to_angle(p, angle)

except KeyboardInterrupt:
	print("stopped")

finally:
	p.stop()
	GPIO.cleanup()
