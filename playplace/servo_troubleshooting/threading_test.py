import numpy as np
import threading 
import queue
import time
import matplotlib.pyplot as plt

from stroke_plane_servo import Servo

def producer(out_q):
	while True:
		# for angle in list(np.arange(0, 270, 10)) + list(np.flip(np.arange(10, 260, 10))):
		for angle in 135 + 135*np.sin(np.linspace(0,2*np.pi,100)):
			print("Sent angle: ", angle)
			out_q.put(angle)
			# must include pause in the producer so it doesn't spam the consumer		
			# time.sleep(0.01) 


def consumer(in_q, servo, _sentinel):
	while True:
		for data in iter(in_q.get, _sentinel):
			angle = data
			servo.turn_to_angle(angle)
			print("Received angle:", angle)


if __name__ == '__main__':
	q = queue.Queue()
	_sentinel = object()

	servo_1 = Servo(17, "Stroke plane servo")

	t1 = threading.Thread(target=producer, args=(q,))
	t2 = threading.Thread(target=consumer, args=(q, servo_1, _sentinel))

	t1.start()
	t2.start()