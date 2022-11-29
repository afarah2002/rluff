#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 12:32:19 2022

@author: tyler
"""
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

log_dir = 'Lift2Drag/'


df = pd.read_csv(log_dir + 'alphas.csv')
raw_alpha = df.to_numpy()

df = pd.read_csv(log_dir+'as.csv')
raw_air_speed = df.to_numpy()

df = pd.read_csv(log_dir+'force.csv')
raw_loc_force = df.to_numpy()


total_gliders = 1
curr_glider = 1

alpha_raw1 = raw_alpha[curr_glider-1::total_gliders,1]
alpha_raw2 = raw_alpha[curr_glider-1::total_gliders,2]
alpha_raw3 = raw_alpha[curr_glider-1::total_gliders,3]
alpha_raw4 = raw_alpha[curr_glider-1::total_gliders,4]

as_raw = raw_air_speed[curr_glider-1::total_gliders,1]
force_x_raw = raw_loc_force[curr_glider-1::total_gliders,1]
force_y_raw = raw_loc_force[curr_glider-1::total_gliders,2]
force_z_raw = raw_loc_force[curr_glider-1::total_gliders,3]


plt.figure();
ax = plt.axes()
plt.plot(alpha_raw1*180/np.pi, '-.')
plt.plot(alpha_raw2*180/np.pi, '--')
plt.plot(alpha_raw3*180/np.pi, '-.')
plt.plot(alpha_raw4*180/np.pi, '--')
ax.legend(['W1','W2','W3','W4'])

plt.figure();
ax = plt.axes()
plt.plot(force_x_raw, '-.')
plt.plot(force_y_raw, '--')
plt.plot(force_z_raw, '-.')
ax.legend(['W1','W2','W3','W4'])

#%%


# ang = np.reshape(raw_alpha[1:,1],(-1,1,1))
ang = alpha_raw1[1:]
# ang = alpha_raw1[2:]

c = np.cos(-ang)
s = np.sin(-ang)
R = torch.tensor([[c, -s],[s, c]])
R = torch.permute(R,(2,0,1))

force_x = torch.reshape(torch.tensor(force_x_raw[1:]), (-1,1,1))
force_z = torch.reshape(torch.tensor(force_z_raw[1:]), (-1,1,1))
force = torch.cat((force_x, force_z), 1)

force_rot = torch.matmul(R, force)

Drag = force_rot[:, 0, 0]
Lift = force_rot[:, 1, 0]
LoD = Lift / Drag


plt.figure();
ax = plt.axes()
plt.scatter(ang*180/np.pi, Drag,s=1)
plt.scatter(ang*180/np.pi, Lift,s=1)
plt.grid(True)
ax.legend(['Drag','Lift'])
plt.xlim([-10, 10])
plt.ylim([-1000, 1000])


#%%


plt.figure();
ax = plt.axes()
plt.scatter(Lift, Drag, s=5)
plt.grid(True)
plt.title('Lift vs Drag')
plt.xlabel('Lift (N)')
plt.ylabel('Drag (N)')
plt.ylim([-2, 2])

plt.figure();
ax = plt.axes()
plt.scatter(ang*180/np.pi, LoD, s=5)
plt.grid(True)
plt.title('Lift/Drag')
plt.ylabel('LoD')
plt.xlabel('Alpha')
# ax.legend(['Tmp'])
plt.xlim([-10, 10])
# plt.ylim([-50, 50])


# plt.figure();
# ax = plt.axes()
# plt.scatter(ang*180/np.pi, Lift, s=5)
# plt.scatter(ang*180/np.pi, Drag, s=5)
# plt.grid(True)
# # ax.legend(['Tmp'])
# plt.xlim([-2, 2])
# plt.ylim([-100, 100])
# ax.legend(['Lift','Drag'])

#%%
eps = 0.000001
dist = torch.arange(0,1000)
target_radius = 100
rew = 1/(dist/target_radius)+eps
rew = torch.where(rew>1, 1, rew)
plt.figure()
plt.plot(dist, rew)
plt.grid(True)
plt.show()

#%% Print All Variables
v = dir()
for d in v:
    print(type(d))

