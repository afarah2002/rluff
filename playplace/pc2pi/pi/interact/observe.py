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
		For now, observes the angle 
		based on the shadow count of
		the motors
		'''
		ang_pos_pack = []
		for m in self.motors:
			ang_pos = self.odrive.read_shadow_pos(m)
			ang_pos_pack.append(ang_pos)
		return ang_pos_pack

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
