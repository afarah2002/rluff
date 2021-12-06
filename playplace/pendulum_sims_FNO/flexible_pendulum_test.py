import pygame
import numpy as np
import time

import physics
import graphics

mass = 0.1
length = 0.3
damping_factor = 0.99

physics_engine = physics.BASIC_KINEM(mass, length, damping_factor)

radii_res = 100
radii = np.linspace(0,length,radii_res)

width, height = 800,400
Out = False

pygame.init()
pendulums = [graphics.Ball((int(width/2),-100),10,r) for r in radii]
graphics_engine = graphics.Graphics_Flex(pendulums)
clock = pygame.time.Clock()

timesteps = np.linspace(0,200,2000)

init_acc = 0
init_vel = 0
init_ang = 0

dt = 0.1


# while not Out:
for t in timesteps:
	clock.tick(20)
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			Out = True

	torque = np.sin(t)

	acceleration = physics_engine.acceleration(init_ang, torque, radius)
	velocity = physics_engine.velocity(init_vel, acceleration, dt)
	angle = physics_engine.angle(init_ang, velocity, dt)

	graphics_engine.get_path(angle)
	graphics_engine.redraw()

	init_acc = acceleration
	init_vel = velocity
	init_ang = angle



