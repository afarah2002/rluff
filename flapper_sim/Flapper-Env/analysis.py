import numpy as np 
import matplotlib.pyplot as plt

loaded_expert = np.load("/home/nasa01/Documents/UML/willis/rluff/flapper_sim/Flapper-Env/expert_data.npz")
expert_observations = loaded_expert['expert_observations']
expert_actions = loaded_expert['expert_actions']

timesteps = np.arange(0,len(expert_observations))

pos = expert_observations[:,0:3]
ori = expert_observations[:,3:7]
vel = expert_observations[:,7:10]
ang_vel = expert_observations[:,10:13]
joint_dyn = expert_observations[:,13:-3]

plt.plot(timesteps, joint_dyn[:,4])
plt.plot(timesteps, joint_dyn[:,0])
plt.plot(timesteps, vel[:,1])
plt.show()