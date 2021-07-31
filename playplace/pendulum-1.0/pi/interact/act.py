from pi.interact.stepper_module.stepper_module import StepperMotor
from pi.interact.odrive_module.odrive_module import ODrive

class StrokePlane(object):

	'''
	Controls a stepper motor (NEMA 8)
	Declared with a list of 4 pins 
	Takes in values of d_theta and speed_pct
	'''

	def __init__(self, pins):
		self.stepper_motor = StepperMotor(pins,
										  drive="half",
										  CONT_BOOL=False)
		
	def turn(self, d_theta, speed_pct):
		self.stepper_motor.turn_to_angle(speed_pct, d_angle=d_theta)


class WingTorque(object):

	'''
	Controls the two T-motors 
	Takes in a value for the torques
	*Note! Values are the same for both motors*
	'''

	def __init__(self, motors):
		self.motors = motors
		self.odrive = ODrive(self.motors)

	def turn(self, torque):
		for m in self.motors:
			self.odrive.turn_trq(m, torque)