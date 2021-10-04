import math
import numpy as np

class BASIC_KINEM(object):

	def __init__(self, mass, length, damping_factor):
		self.mass = mass
		self.length = length
		self.damping_factor = damping_factor

	def acceleration(self, initial_angle, torque):
		acceleration = -(9.8/self.length) * math.sin(initial_angle*np.pi/180) + torque/(self.mass*self.length**2)
		return acceleration

	def velocity(self, initial_velocity, acceleration, dt):
		velocity = acceleration*dt*180/np.pi + initial_velocity
		damped_velocity = self.damping_factor*velocity
		return damped_velocity

	def angle(self, initial_angle, velocity, dt):
		angle = velocity*dt + initial_angle
		return angle

