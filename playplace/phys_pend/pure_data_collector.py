import pi.interact.observe as observe
import pi.interact.act as act
import time
from timeit import default_timer as timer
import numpy as np


def main():
	motors = [0]
	wing_torque_motor = act.WingTorque(motors)
	odrive_mod = wing_torque_motor.odrive
	wing_observer = observe.Wings(odrive_mod, motors)

	print("Swing it up...")
	time.sleep(2)
	print("Go!!!")
	time.sleep(0.25)

	t_total = 50
	dt = 0.0001
	Nt = 50000+1
	t = np.zeros([Nt,1])
	y = np.zeros([Nt,2]) # current angle, current angular velocity
	t0 = timer()

	for i in range(Nt-1):
		current_angle = wing_observer.ang_pos()[0]
		current_time = timer() - t0 
		t[i+1] = current_time
		y[i+1,0] = current_angle
		time.sleep(dt)
		print(i, " ", current_time, f"Angle: {current_angle}")

	np.save("t.npy",t)
	np.save("y.npy",y)

if __name__ == '__main__':
	main()