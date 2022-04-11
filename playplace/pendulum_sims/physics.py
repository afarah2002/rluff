import math
import numpy as np

class BASIC_KINEM(object):

	def __init__(self, mass, length, damping_factor):
		self.mass = mass
		self.length = length
		self.damping_factor = damping_factor
		# output to values that can actually be sent to 

	def acceleration(self, initial_angle, torque):
		acceleration = -(9.8/self.length) * math.sin(initial_angle) + torque/(self.mass*self.length**2)
		return acceleration

	def velocity(self, initial_velocity, acceleration, dt):
		velocity = acceleration*dt + initial_velocity
		damped_velocity = self.damping_factor*velocity
		return damped_velocity

	def angle(self, initial_angle, velocity, dt):
		angle = velocity*dt + initial_angle
		return angle


class BASIC_KINEM_FLEX(object):

	def __init__(self, mass, length, damping_factor):
		self.mass = mass
		self.length = length
		self.damping_factor = damping_factor
		# output to values that can actually be sent to 

	def acceleration(self, initial_angle, torque, length):
		acceleration = -(9.8/length) * math.sin(initial_angle) + torque/(self.mass*length**2)
		return acceleration

	def velocity(self, initial_velocity, acceleration, dt):
		velocity = acceleration*dt + initial_velocity
		damped_velocity = self.damping_factor*velocity
		return damped_velocity

	def angle(self, initial_angle, velocity, dt):
		angle = velocity*dt + initial_angle
		return angle


import numpy as np

#-----------------RK4 code, thanks Milo!-----------------#

class RK4(object):

	def __init__(self, m, b, r, dt):

		self.m = m # bob mass
		self.b = b # damping constant
		self.r = r # radius
		self.dt = dt

	def getdydt(self,t,y,tau):
		
		# y = [current theta, current dtheta/dt] = [angle, angular velocity]

		g = -9.8
		I = self.m*self.r*self.r
		
		dy1dt = y[1];
		dy2dt = (self.r*g*self.m*y[0] - self.b*y[1] + tau)/I;
		dydt = np.array([dy1dt,dy2dt])
		return dydt

	def forward(self,t,y,tau=None):
		# Makes torque an option
		if tau:
			tau = tau
		else:
			tau = 0.0

		K1 = self.dt*self.getdydt(t,y,tau)
		K2 = self.dt*self.getdydt(t+0.5*self.dt, y+0.5*K1,tau)
		K3 = self.dt*self.getdydt(t+0.5*self.dt, y+0.5*K2,tau)
		K4 = self.dt*self.getdydt(t+self.dt,y+K3,tau)
		y = y+(K1 + 2.0*K2 + 2.0*K3 + K4)/6.0
		t = t+self.dt
		return t, y 

