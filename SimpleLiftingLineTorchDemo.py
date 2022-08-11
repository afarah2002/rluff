import torch
import math

device = torch.device("cuda:0")
torch.set_printoptions(precision=10)


# import numpy as np
import matplotlib.pyplot as plt
import time

#%% SETUP:
    
M = 2;                        # number of wings of certian type
N = 50;                         # number of "stations" per wing
eps = 0.0001;                     # small offset
Cla = 6.5;                      # lift coeff slope [Cl/rad]
rho = 1.0;                      # fluid density
c =  torch.reshape(torch.linspace(1.0,2.0,N),[N,1]).to(device); 
s = 5.0;                        # semi span
y = torch.reshape(torch.linspace(eps-s,s-eps,N),[N,1]).to(device); 
                                # "station" locs
theta = torch.arccos(y/s);         # "station" locs in theta coord system

# temp vec for later:
vec1 = torch.reshape(torch.sin(theta)*c*Cla/8.0/s,[N,1])

# making alpha Vinf arrays:
# alpha = 5.0*(2.0*torch.rand(1,M) - 1.0)*3.14/180.0;
# Vinf = 4.0*(torch.rand(1,M))+1.0;


alpha = 5.0*(2.0*torch.linspace(1.0,2.0,M) - 1.0)*3.14/180.0;
Vinf = 4.0*(torch.linspace(1.0,2.0,M))+1.0;
alpha = alpha.to(device)
Vinf = Vinf.to(device)

alpha = alpha.reshape(1,M)
Vinf = Vinf.reshape(1,M)


# build system matrix:
n = torch.reshape(torch.linspace(1,N,N),[1,N]).to(device)
mat1 = (n*c*Cla/8.0/s +  torch.sin(theta))*torch.sin(n*theta)
mat2 = 4.0*s*torch.sin(n*theta)
#%% EACH UPDATE:

    
now = time.perf_counter(); 
    
vec2 = alpha 
# print(vec1, vec2)
# print(vec1.shape(), vec2.shape())

RHS = torch.matmul(vec1,vec2)

A = torch.linalg.solve(mat1,RHS)    
# each col in above will have the "A" coeffs for the mth wing
Gamma = torch.matmul(mat2,A)*Vinf*rho

exec_time = time.perf_counter() - now; 

print(A)

#%%
print(exec_time)
plt.plot(y.cpu().numpy(),Gamma.cpu().numpy())
plt.xlabel('y [m]')
plt.ylabel('Lift Dist [N/m]')

plt.show()
