import math
import numpy as np
import pygame


width, height = 800, 400 # w/h of window
white = (255,255,255)
black = (0,0,0)
gray = (150, 150, 150)
Dark_red = (150, 0, 0)

class Ball(object):
 
	def __init__(self, XY, radius, length):  # Set ball coordenates and radius
		self.x = XY[0]
		self.y = XY[1]
		self.radius = radius
		self.length = length
 
	def draw(self, bg):  # Draw circle and line based on XY coordinates
		pygame.draw.lines(bg, black, False, [(width/2, 50), (self.x, self.y)], 2)
		pygame.draw.circle(bg, black, (self.x, self.y), self.radius)
		pygame.draw.circle(bg, Dark_red, (self.x, self.y), self.radius - 2)
 


class Graphics(object):

	def __init__(self, pendulum):
		self.pendulum = pendulum
		self.background = pygame.display.set_mode((width, height))
	
	def grid(self):  # Draw a grid behind the pendulum
		for x in range(50, width, 50):
			pygame.draw.lines(self.background, gray, False, [(x, 0), (x, height)])
			for y in range(50, height, 50):
				pygame.draw.lines(self.background, gray, False, [(0, y), (width, y)])
		pygame.draw.circle(self.background, black, (int(width/2), 50), 5)

	def get_path(self, angle): # with angle and length calculate x and y position
		self.pendulum.x = round(width/2 + 500*self.pendulum.length * math.sin(angle))
		self.pendulum.y = round(50 + 500*self.pendulum.length * math.cos(angle))
 
	def redraw(self): # Clean up the screen and start a new grid and new frame of pendulum with new coordinates
		self.background.fill(white)
		self.grid()
		self.pendulum.draw(self.background)
		pygame.display.update()

class Graphics_Flex(object):

	def __init__(self, pendulums):
		self.pendulums = pendulums
		self.background = pygame.display.set_mode((width, height))
	
	def grid(self):  # Draw a grid behind the pendulum
		for x in range(50, width, 50):
			pygame.draw.lines(self.background, gray, False, [(x, 0), (x, height)])
			for y in range(50, height, 50):
				pygame.draw.lines(self.background, gray, False, [(0, y), (width, y)])
		pygame.draw.circle(self.background, black, (int(width/2), 50), 5)

	def get_path(self, angle): # with angle and length calculate x and y position
		for pendulum in self.pendulums:
			pendulum.x = round(width/2 + 500*pendulum.length * math.sin(angle))
			pendulum.y = round(50 + 500*pendulum.length * math.cos(angle))
 
	def redraw(self): # Clean up the screen and start a new grid and new frame of pendulum with new coordinates
		self.background.fill(white)
		self.grid()
		for pendulum in self.pendulums:
			pendulum.draw(self.background)
		pygame.display.update()