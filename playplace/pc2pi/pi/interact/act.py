from pi.interact.stepper_module.stepper_module import StepperMotor

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

