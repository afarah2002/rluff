import numpy as np
import matplotlib.pyplot as plt


test_num = "018"
target = 100

raw_data_loc = f"test_data/{test_num}_{target}/raw_data/"


actions_buffer =  np.load(raw_data_loc + "actions_buffer.npy")
states_buffer =  np.load(raw_data_loc + "states_buffer.npy")
real_states_buffer =  np.load(raw_data_loc + "real_states_buffer.npy")
avg_theta_dot_buffer =  np.load(raw_data_loc + "avg_theta_dot_buffer.npy")
real_rewards_buffer = np.load(raw_data_loc + "real_rewards_buffer.npy")
std_buffer = np.load(raw_data_loc + "std_buffer.npy")
phase_space_amp_buffer = np.load(raw_data_loc + "phase_space_amp_buffer.npy")

t = len(real_rewards_buffer)

# plt.plot(range(t), real_rewards_buffer[:t], label="rewards")
# plt.plot(range(t), actions_buffer[:t], label="torques")
# plt.plot(range(t), avg_theta_dot_buffer[:t], label="avg theta dot")
# plt.plot(range(t), np.sqrt(phase_space_amp_buffer)[:t], lael="phase space amp")
# plt.plot(range(t), np.power(real_states_buffer[:t,0],2) + np.power(real_states_buffer[:t,1],2))
# plt.plot(range(t), std_buffer[:t], label="std")
plt.plot(range(t), states_buffer[:t,1], label="predicted")
plt.plot(range(t), real_states_buffer[:t,1], label="real")
plt.legend()

plt.show()
