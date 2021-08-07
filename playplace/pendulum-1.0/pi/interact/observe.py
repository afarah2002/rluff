class Wings(object):

	'''
	Observes the angular position 
	and velocities of the wings 
	Must use the same ODrive obj 
	as the wing action
	'''

	def __init__(self, odrive, motors):
		self.motors = motors
		self.odrive = odrive

	def ang_pos(self):
		'''
		Observes the angle (in deg)
		of the pendulum
		'''
		ang_pos_pack = []
		for m in self.motors:
			ang_pos = self.odrive.read_angle(m)
			ang_pos_pack.append(ang_pos)
		return ang_pos_pack

	def ang_vel(self):
		'''
		Observes the angular velocity (in deg/s)
		of the pendulum
		'''
		ang_vel_pack = []
		for m in self.motors:
			ang_vel = self.odrive.read_ang_vel(m)
			ang_vel_pack.append(ang_vel)
		return ang_vel_pack

	def mot_cur(self):
		'''
		Measures the current through motors
		Displays currents in action section of GUI
		'''
		mot_cur_pack = []
		for m in self.motors:
			Iq_measured = self.odrive.read_current(m)
			mot_cur_pack.append(Iq_measured)
		return mot_cur_pack

	def mot_trq(self):
		'''
		Measures the torque through applied by motor
		Displays measured torques in action section of GUI
		'''
		mot_cur_pack = []
		for m in self.motors:
			Iq_measured = self.odrive.read_trq(m)
			mot_cur_pack.append(Iq_measured)
		return mot_cur_pack

class IMU(object):

	'''
	Return different readings from the IMU
	''' 

	def __init__(self):
		pass

	def XYZ_9pack(self):
		'''
		Pos, vel, acc for XYZ
		'''
		pass

	def RPY_6pack(self):
		'''
		Pos, vel for roll, pitch, yaw 
		'''
		pass
