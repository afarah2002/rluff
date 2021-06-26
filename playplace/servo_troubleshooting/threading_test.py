import numpy as np
import threading 
import queue
import time
import matplotlib.pyplot as plt
import random

from stroke_plane_servo import Servo

# precision_factor = int(input("Precision: "))

def producer(out_q, max_speed):
	while True:
		uniform_list = list(np.array(100*[list(np.linspace(0, 270, 60)) + \
										  list(np.flip(np.linspace(10, 260, 60)))]).flatten())
		print(len(uniform_list), "\n\n\n")
		# sine_list = list(130 - 130*np.cos(np.linspace(0,4*np.pi,60)))
		sine_list = list(130 - 130*np.cos(np.arange(0, 4*np.pi, 4*np.pi/60)))

		LIST = sine_list
		difference = 1
		for i in range(len(LIST)):
			angle = LIST[i]
			if i > 0:
				difference = np.abs(angle - LIST[i-1])
				print("difference = ", difference)
			# print("Sent angle: ", angle)
			delay = difference/max_speed
			out_q.put((angle, delay))
			# must include pause in the producer so it doesn't spam the consumer		
			time.sleep(delay) 

def rand_producer(out_q, speed):
	angles_list = [135]
	while True:
		angle = random.uniform(0., 260.)
		angles_list.append(angle)

		difference = np.abs(angles_list[-1] - angles_list[-2])

		delay = difference/(speed)

		print("Angle: ", angle, "   ", " Delay: ", delay)

		out_q.put((angle, delay))
		time.sleep(delay)

def speed_conscious_random_producer(out_q, speed):
	''' 
	Given a speed and a random angle, generates a 
	list between the random angle and the current
	angle so that the speed from beginning to end 
	is the specified speed	
	'''
	max_speed = 150
	angles_list = [135]
	while True:
		angle = random.uniform(0., 260.)
		angles_list.append(angle)
		print("Angle: ", angle)

		D_theta = np.abs(angles_list[-1] - angles_list[-2])
		bounded_list = np.linspace(angles_list[-2], 
								   angles_list[-1],
								   int(D_theta*max_speed/speed))

		d_theta = 0.1
		for i in range(len(bounded_list)):
			if i > 0:
				d_theta = np.abs(bounded_list[i] - bounded_list[i-1])
			delay = d_theta/speed
			
			out_q.put((bounded_list[i], delay))
			time.sleep(delay)



def consumer(in_q, servo, _sentinel):
	while True:
		for data in iter(in_q.get, _sentinel):
			angle = data[0]
			delay = data[1]
			# print("Received angle:", angle)
			try:
				servo.turn_to_angle(angle, delay)
			except KeyboardInterrupt:
				print("stopped")

if __name__ == '__main__':
	q = queue.Queue()
	_sentinel = object()

	# max_speed = 150
	speed = float(input("Speed: "))

	servo_1 = Servo(17, "Stroke plane servo")

	t1 = threading.Thread(target=speed_conscious_random_producer, args=(q, speed))
	t2 = threading.Thread(target=consumer, args=(q, servo_1, _sentinel))

	t1.start()
	t2.start()