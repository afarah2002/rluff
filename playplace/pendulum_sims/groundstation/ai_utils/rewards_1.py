import numpy as np
import math

class R_omega:

	def polyn_1(omega, omega_target, A=0.001, C=0):
		# Quadratic
		return -A*((np.abs(omega) - omega_target))**2 + C

	def log_1(omega, omega_target, C=50):
		# -|ln()|
		return -omega_target*np.log(np.abs(omega/omega_target)) + C

	def hyperb_1(omega, omega_target, A=100, B=3):
		# 1/((x-target)^2) - |x-target|
		return A/((omega - omega_target)**2) - B*np.abs(omega - omega_target)

	def abs_1(omega, omega_target, A=10):
		# -|x-target|
		return -A*np.abs(omega - omega_target)

class R_tau:

	def hyperb_1(tau, A=1e-4):
		# 1/(x^2)
		return A/(tau**2)

	def hyperb_2(tau, A=1e-4):
		# 1/|x|
		return A/np.abs(tau)

	def hyperb_3(tau, A=1e-5, B=1):
		# 1/(x^2) - |x|
		return A/(tau**2) - B*np.abs(tau)

	def polyn_1(tau, A=10, C=0):
		# Quadratic
		return -A*(tau)**2 + C

	def abs_1(tau, A=2e4):
		# -|x-target|
		return -A*np.abs(tau)


class R_theta:

	def polyn_1(theta, A=1e-5, C=0):
		# -A(x/B)^4 + C
		return -A*(theta)**4 + C

class RewardUtils:

	def weighted_average(data_array, alpha=1):
		# higher alpha means more recent data is weighted heavier
		num = np.size(data_array,0) # Num of timesteps since start
		t_weights = np.linspace(0, 1, num) # Evenly spaced steps for w = e^(alpha*t)
		weights = np.exp(alpha*t_weights) # Exponential weight generation
		weights /= np.max(weights) # Normalizes weights, doesn't affect value 

		weighted_average = np.sum(data_array*weights)/np.sum(weights) # Weighted avg formula

		return weighted_average


class Rewards_1(object):

	def __init__(self, gui_data_classes):
		self.gui_data_classes = gui_data_classes

	def reward_1(self, combo_data, target):

		wing_torque = combo_data["action"]["Wing torques"]
		action_num = combo_data["action"]["Action num"]
		next_ang_pos = combo_data["next state"]["Wing angles"]
		next_ang_vel = combo_data["next state"]["Angular velocity"]

		# Make copies of data class subsets
		prev_depth = 20 # number of recent elements used
		if np.size(self.gui_data_classes["Wing torques"].XData,0) < prev_depth:
			prev_depth = np.size(self.gui_data_classes["Wing torques"].XData,0)
		else:
			prev_depth = 20

		prev_wing_trq = np.array(self.gui_data_classes["Wing torques"].YData[-prev_depth:,0])
		prev_ang_pos = np.array(self.gui_data_classes["Wing angles"].YData[-prev_depth:,0])
		prev_ang_vel = np.array(self.gui_data_classes["Angular velocity"].YData[-prev_depth:,0])

		# Update copies of data class subsets
		np.append(prev_wing_trq, wing_torque)
		np.append(prev_ang_pos, next_ang_pos)
		np.append(prev_ang_vel, next_ang_vel)


		target_ang_vel = float(target) # deg/s

		avg_ang_pos = np.mean(np.absolute(prev_ang_pos))
		avg_abs_trq = np.mean(np.absolute(prev_wing_trq))
		avg_abs_ang_vel = np.mean(np.absolute(prev_ang_vel)) # avg of abs val of ang vel

		# Instantaneous - uses current/next state
		# ang_vel = next_ang_vel[0]
		wng_trq = wing_torque[0]
		ang_pos = next_ang_pos[0]
		# Average
		ang_vel = avg_abs_ang_vel
		# wng_trq = wing_torque[0]
		# ang_pos = avg_ang_pos

		if ang_vel == 0:
			ang_vel = 0.001
		if wng_trq == 0:
			wng_trq = 0.001
		if ang_pos == 0:
			ang_pos = 0.001

		# Calculate reward
		R_ang_vel = 0.1*R_omega.polyn_1(ang_vel, target_ang_vel)
		# R_ang_vel = 0.1*R_omega.log_1(ang_vel, target_ang_vel)
		# Did we hit the failsafe? Do not reward this! LOL
		if not np.all(prev_wing_trq==0):
			# R_torque = R_tau.hyperb_3(wng_trq)/100
			R_torque = R_tau.polyn_1(wng_trq)
		else:
			# R_torque = R_tau.hyperb_1(100)/100
			print("FAILSAFE")
			R_torque = -5000
		R_ang_pos = 5*R_theta.polyn_1(ang_pos)
		reward = [(R_ang_vel + R_torque + R_ang_pos)/200]

		# Episode ends after 1000 actions
		if action_num == 1000:
			done = True
		else:
			done = False

		return reward, done

	def reward_2(self, combo_data, target):

		# Uses weighted moving average to assess AI real time

		wing_torque = combo_data["action"]["Wing torques"]
		action_num = combo_data["action"]["Action num"]
		next_ang_pos = combo_data["next state"]["Wing angles"]
		next_ang_vel = combo_data["next state"]["Angular velocity"]

		# Make copies of data class subsets
		prev_depth = 200 # number of recent elements used
		if np.size(self.gui_data_classes["Wing torques"].XData,0) < prev_depth:
			prev_depth = np.size(self.gui_data_classes["Wing torques"].XData,0)
		else:
			prev_depth = 200

		prev_wing_trq = np.array(self.gui_data_classes["Wing torques"].YData[-prev_depth:,0])
		prev_ang_pos = np.array(self.gui_data_classes["Wing angles"].YData[-prev_depth:,0])
		prev_ang_vel = np.array(self.gui_data_classes["Angular velocity"].YData[-prev_depth:,0])

		# Update copies of data class subsets
		prev_wing_trq = np.append(prev_wing_trq, wing_torque[1])
		prev_ang_pos = np.append(prev_ang_pos, next_ang_pos)
		prev_ang_vel = np.append(prev_ang_vel, next_ang_vel)

		target_ang_vel = float(target) # deg/s

		# Weighted average of angular velocity
		weighted_ang_vel_avg = RewardUtils.weighted_average(np.absolute(prev_ang_vel), alpha=5)
		weighted_torque = RewardUtils.weighted_average(np.absolute(prev_wing_trq), alpha=5)

		# Correspondig angular amplitude
		ang_amplitude = max(np.absolute(prev_ang_pos[-20:]))
		print("Weighted angular velocity average: ", weighted_ang_vel_avg)
		print("Corresponding angular amplitude: ", ang_amplitude)

		# Instantaneous - uses current/next state
		# ang_vel = next_ang_vel[0]
		wng_trq = wing_torque[0]
		ang_pos = next_ang_pos[0]
		# Average
		ang_vel = weighted_ang_vel_avg
		# wng_trq = weighted_torque
		# ang_pos = avg_ang_pos

		# Display the weighted moving avg on the ang vel plot
		self.gui_data_classes["Angular velocity"].YData[-1,-1] = weighted_ang_vel_avg 
		self.gui_data_classes["Wing angles"].YData[-1,-1] = ang_amplitude

		if ang_vel == 0:
			ang_vel = 0.001
		if wng_trq == 0:
			wng_trq = 0.001
		if ang_pos == 0:
			ang_pos = 0.001

		# Episode ends after 100 actions
		if action_num == 100:
			done = True
		else:
			done = False
		# Calculate reward
		R_ang_vel = R_omega.polyn_1(ang_vel, target_ang_vel)
		R_torque = R_tau.polyn_1(wng_trq)
		# R_ang_pos = R_theta.polyn_1(ang_pos)
		reward = [R_ang_vel + R_torque]

		# Did we hit the failsafe? Do not reward this! LOL
		if not np.all(prev_wing_trq[-10:]==0):
			pass
		else:
			print("FAILSAFE")
			# Penalize and end episode
			reward = [-15 + R_ang_vel]
			# done = True


		return reward, done
		

