# Wesley Fernandes
# python simple pendulum with pygame
 
import pygame
import math
import random
import numpy
 
#VARIABLES
width, height = 800, 400   # set the width and height of the window
						   # (you can increase or decrease if you want to, just remind to keep even numbers)
Out = False                # if True,out of while loop, and close pygame
acceleration = False       # when true it allow us to find the acceleration and damping for the pendulum
length = 0                 # the length between the ball and the support
angle = 0                  # the angle that you begin when click in window
vel = 0                    # velocity that angle is increased and damped
Aacc = 0                   # acceleration
 
#COLORS
white = (255,255,255)
black = (0,0,0)
gray = (150, 150, 150)
Dark_red = (150, 0, 0)
 
#BEFORE START
pygame.init()
background = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
 
class Ball(object):
 
	def __init__(self, XY, radius):  # Set ball coordenates and radius
		self.x = XY[0]
		self.y = XY[1]
		self.radius = radius
 
	def draw(self, bg):  # Draw circle and line based on XY coordinates
		pygame.draw.lines(bg, black, False, [(width/2, 50), (self.x, self.y)], 2)
		pygame.draw.circle(bg, black, (self.x, self.y), self.radius)
		pygame.draw.circle(bg, Dark_red, (self.x, self.y), self.radius - 2)
 
 
def grid():  # Draw a grid behind the pendulum
	for x in range(50, width, 50):
		pygame.draw.lines(background, gray, False, [(x, 0), (x, height)])
		for y in range(50, height, 50):
			pygame.draw.lines(background, gray, False, [(0, y), (width, y)])
	pygame.draw.circle(background, black, (int(width/2), 50), 5)
 
def angle_Length():  # Send back the length and angle at the first click on screen
	length = math.sqrt(math.pow(pendulum.x - width/2, 2) + math.pow(pendulum.y - 50, 2))
	angle = math.asin((pendulum.x - width/2)/ length)
	return (angle, length)
 
def get_path(first_angle, length): # with angle and length calculate x and y position
	pendulum.x = round(width/2 + length * math.sin(angle))
	pendulum.y = round(50 + length * math.cos(angle))
 
def redraw(): # Clean up the screen and start a new grid and new frame of pendulum with new coordinates
	background.fill(white)
	grid()
	pendulum.draw(background)
	pygame.display.update()
 
 
pendulum = ball((int(width / 2),-100), 5) # I start the class with some random coordinates

timestep = 0 
while not Out:
	timestep +- 0.01
	clock.tick(120)             #Set how many frames are draw per second
								#If changed, maybe, could be a good idea change some values at acceleration
 
	for event in pygame.event.get():                     #
		if event.type == pygame.QUIT:                    #
			Out = True                                   #
		if event.type == pygame.MOUSEBUTTONDOWN:         #  Read if you want go out
			pendulum = ball(pygame.mouse.get_pos(), 15)  #             or
			angle, length = angle_Length()               #   click the mouse button
			acceleration = True                          #
 
	if acceleration:   # Increase acceleration and damping in the pendulum moviment
		length = 200
		mass = 1
		Aacc = -0.005 * math.sin(angle)
		torque = math.sin(timestep)
		Aacc += 100000*torque/(mass*length**2)
		vel += Aacc
		# vel *= 0.99  # damping factor
		angle += vel
		print(angle)
		get_path(angle, length)
 
	redraw()
 
pygame.quit()
