import numpy as np
import math

class Rewards(object):

	def __init__(self, gui_data_classes):
		self.gui_data_classes = gui_data_classes

	def reward_1(self, combo_data):
		wing_torque = combo_data["action"]["Wing torques"]
		action_num = combo_data["action"]["Action num"]
		next_ang_pos = combo_data["next state"]["Wing angles"]
		next_ang_vel = combo_data["next state"]["Angular velocity"]

		# Make copies of data class subsets
		prev_depth = 50 # number of recent elements used
		if len(wing_torque) < 50:
			prev_depth = len(wing_torque)

		prev_wing_trq = np.array(self.gui_data_classes["Wing torques"].YData[-prev_depth:])
		prev_ang_pos = np.array(self.gui_data_classes["Wing angles"].YData[-prev_depth:])
		prev_ang_vel = np.array(self.gui_data_classes["Angular velocity"].YData[-prev_depth:])

		# Update copies of data class subsets
		np.append(prev_wing_trq, wing_torque)
		np.append(prev_ang_pos, next_ang_pos)
		np.append(prev_ang_vel, next_ang_vel)

		# Calculate reward
		target_vel = 422 # deg/s
		avg_abs_trq = np.mean(np.absolute(prev_wing_trq[-20:]))
		avg_abs_ang_vel = np.mean(np.absolute(prev_ang_vel[-20:])) # avg of abs val of ang vel
		diff = abs(avg_abs_ang_vel - target_vel)

		if diff <= 5:
			real_time_reward = 100/diff - abs(wing_torque[0])*10
		else:
			real_time_reward = -action_num - abs(wing_torque[0])*10

		# Check for lollipop (long term avg vel)
		lol_avg_abs_ang_vel = np.mean(np.absolute(prev_ang_vel))
		lol_diff = abs(lol_avg_abs_ang_vel - target_vel)

		if lol_diff <= 5:
			lollipop = 100
			done = True
		else: 
			lollipop = 0
			done = False

		# # If there is no lollipop, there is a 5% chance that the ep will terminate
		# if not done:
		# 	rand_done = np.random.rand(1)
		# 	if rand_done > 0.95:
		# 		done = True
		# 	else:
		# 		done = False

		reward = [real_time_reward + lollipop]
		
		return reward, done

	def reward_2(self, combo_data, target):
		'''
		RATIONAL FUNC
		'''
		wing_torque = combo_data["action"]["Wing torques"]
		action_num = combo_data["action"]["Action num"]
		next_ang_pos = combo_data["next state"]["Wing angles"]
		next_ang_vel = combo_data["next state"]["Angular velocity"]

		# Make copies of data class subsets
		prev_depth = 100 # number of recent elements used
		if len(wing_torque) < 100:
			prev_depth = len(wing_torque)

		prev_wing_trq = np.array(self.gui_data_classes["Wing torques"].YData[-prev_depth:])
		prev_ang_pos = np.array(self.gui_data_classes["Wing angles"].YData[-prev_depth:])
		prev_ang_vel = np.array(self.gui_data_classes["Angular velocity"].YData[-prev_depth:])

		# Update copies of data class subsets
		np.append(prev_wing_trq, wing_torque)
		np.append(prev_ang_pos, next_ang_pos)
		np.append(prev_ang_vel, next_ang_vel)

		# Calculate reward
		target_vel = target # deg/s
		avg_abs_trq = np.mean(np.absolute(prev_wing_trq))
		avg_abs_ang_vel = np.mean(np.absolute(prev_ang_vel)) # avg of abs val of ang vel
		diff = abs(avg_abs_ang_vel - target_vel)

		reward = [100/diff - abs(wing_torque[0])]

		# Episode ends after 1000 actions
		if action_num == 1000:
			done = True
		else:
			done = False
		
		return reward, done

	def reward_3(self, combo_data, target):
		'''
		HYPERBOLA
		'''
		wing_torque = combo_data["action"]["Wing torques"]
		action_num = combo_data["action"]["Action num"]
		next_ang_pos = combo_data["next state"]["Wing angles"]
		next_ang_vel = combo_data["next state"]["Angular velocity"]

		# Make copies of data class subsets
		prev_depth = 100 # number of recent elements used
		if len(wing_torque) < 100:
			prev_depth = len(wing_torque)

		prev_wing_trq = np.array(self.gui_data_classes["Wing torques"].YData[-prev_depth:])
		prev_ang_pos = np.array(self.gui_data_classes["Wing angles"].YData[-prev_depth:])
		prev_ang_vel = np.array(self.gui_data_classes["Angular velocity"].YData[-prev_depth:])

		# Update copies of data class subsets
		np.append(prev_wing_trq, wing_torque)
		np.append(prev_ang_pos, next_ang_pos)
		np.append(prev_ang_vel, next_ang_vel)

		# Calculate reward
		target_vel = target # deg/s
		avg_abs_trq = np.mean(np.absolute(prev_wing_trq))
		avg_abs_ang_vel = np.mean(np.absolute(prev_ang_vel)) # avg of abs val of ang vel
		diff = abs(avg_abs_ang_vel - target_vel)

		reward = [100/diff - abs(wing_torque[0])*diff]

		# Episode ends after 1000 actions
		if action_num == 1000:
			done = True
		else:
			done = False
		
		return reward, done

	def reward_4(self, combo_data, target):
		'''
		HYPERBOLA
		Constraints on position mimic pain if it tries to break itself
		'''
		wing_torque = combo_data["action"]["Wing torques"]
		action_num = combo_data["action"]["Action num"]
		next_ang_pos = combo_data["next state"]["Wing angles"]
		next_ang_vel = combo_data["next state"]["Angular velocity"]

		# Make copies of data class subsets
		prev_depth = 500 # number of recent elements used
		if len(wing_torque) < prev_depth:
			prev_depth = len(wing_torque)

		prev_wing_trq = np.array(self.gui_data_classes["Wing torques"].YData[-prev_depth:])
		prev_ang_pos = np.array(self.gui_data_classes["Wing angles"].YData[-prev_depth:])
		prev_ang_vel = np.array(self.gui_data_classes["Angular velocity"].YData[-prev_depth:])

		# Update copies of data class subsets
		np.append(prev_wing_trq, wing_torque)
		np.append(prev_ang_pos, next_ang_pos)
		np.append(prev_ang_vel, next_ang_vel)

		# Calculate reward
		target_vel = target # deg/s
		pos_constraint = 50
		constraint_pain = 0
		if abs(next_ang_pos[0]) > pos_constraint:
			constraint_pain = 200

		avg_abs_trq = np.mean(np.absolute(prev_wing_trq))
		avg_abs_ang_vel = np.mean(np.absolute(prev_ang_vel)) # avg of abs val of ang vel
		diff = abs(avg_abs_ang_vel - target_vel)

		reward = [100/diff - abs(wing_torque[0])*diff - constraint_pain]

		# Episode ends after 1000 actions
		if action_num == 1000:
			done = True
		else:
			done = False
		
		return reward, done	

	def reward_5(self, combo_data, target):
		'''
		HYPERBOLA
		Constraints on position mimic pain if it tries to break itself
		'''
		wing_torque = combo_data["action"]["Wing torques"]
		action_num = combo_data["action"]["Action num"]
		next_ang_pos = combo_data["next state"]["Wing angles"]
		next_ang_vel = combo_data["next state"]["Angular velocity"]

		# Make copies of data class subsets
		prev_depth = 100 # number of recent elements used
		if len(wing_torque) < prev_depth:
			prev_depth = len(wing_torque)

		prev_wing_trq = np.array(self.gui_data_classes["Wing torques"].YData[-prev_depth:])
		prev_ang_pos = np.array(self.gui_data_classes["Wing angles"].YData[-prev_depth:])
		prev_ang_vel = np.array(self.gui_data_classes["Angular velocity"].YData[-prev_depth:])

		# Update copies of data class subsets
		np.append(prev_wing_trq, wing_torque)
		np.append(prev_ang_pos, next_ang_pos)
		np.append(prev_ang_vel, next_ang_vel)

		# Calculate reward
		target_vel = target # deg/s
		pos_constraint = 50
		constraint_pain = 0
		if abs(next_ang_pos[0]) > pos_constraint:
			constraint_pain = 10
		if abs(next_ang_pos[0]) > 85:
			constraint_pain = 10000

		avg_abs_trq = np.mean(np.absolute(prev_wing_trq))
		avg_abs_ang_vel = np.mean(np.absolute(prev_ang_vel)) # avg of abs val of ang vel
		diff = abs(avg_abs_ang_vel - target_vel)

		reward = [100/diff - abs(wing_torque[0]) - diff/10 - constraint_pain]
		# reward = [100/diff - abs(wing_torque[0]) - diff/10 - constraint_pain + 10]

		# Episode ends after 1000 actions
		if action_num == 1000:
			done = True
		else:
			done = False
		
		return reward, done

	def reward_6(self, combo_data, target):
		'''
		The faster it goes (bounded by angles), the more it is rewarded
		'''
		wing_torque = combo_data["action"]["Wing torques"]
		action_num = combo_data["action"]["Action num"]
		next_ang_pos = combo_data["next state"]["Wing angles"]
		next_ang_vel = combo_data["next state"]["Angular velocity"]

		# Make copies of data class subsets
		prev_depth = 100 # number of recent elements used
		if np.size(wing_torque,0) < prev_depth:
			prev_depth = len(wing_torque)

		prev_wing_trq = self.gui_data_classes["Wing torques"].YData[-prev_depth:,:]
		prev_ang_pos = self.gui_data_classes["Wing angles"].YData[-prev_depth:,:]
		prev_ang_vel = self.gui_data_classes["Angular velocity"].YData[-prev_depth:,:]

		# Update copies of data class subsets
		np.append(prev_wing_trq, wing_torque)
		np.append(prev_ang_pos, next_ang_pos)
		np.append(prev_ang_vel, next_ang_vel)

		# Calculate reward
		# target_vel = target # deg/s
		pos_constraint = 50
		constraint_pain = 0
		if abs(next_ang_pos[0]) > pos_constraint:
			constraint_pain = 1000

		avg_abs_trq = np.mean(np.absolute(prev_wing_trq))
		avg_abs_ang_vel = np.mean(np.absolute(prev_ang_vel)) # avg of abs val of ang vel
		# diff = abs(avg_abs_ang_vel - target_vel)

		# reward = [100/diff - abs(wing_torque[0]) - diff/10 - constraint_pain]
		# reward = [100/diff - abs(wing_torque[0]) - diff/10 - constraint_pain + 10]
		reward = [avg_abs_ang_vel/100 - 100*abs(wing_torque[0]) - constraint_pain]

		# Episode ends after 1000 actions
		if action_num == 1000:
			done = True
		else:
			done = False
		
		return reward, done

	def reward_7(self, combo_data, target):
		'''
		Instead of targeting a velocity, it targets a frequency 
		In the case of a natural pendulum, this would be the max
		angular velocity divided by the max angle
		'''
		wing_torque = combo_data["action"]["Wing torques"]
		action_num = combo_data["action"]["Action num"]
		next_ang_pos = combo_data["next state"]["Wing angles"]
		next_ang_vel = combo_data["next state"]["Angular velocity"]

		# Make copies of data class subsets
		prev_depth = 100 # number of recent elements used
		if len(wing_torque) < prev_depth:
			prev_depth = len(wing_torque)


		prev_wing_trq = np.array(self.gui_data_classes["Wing torques"].YData[-prev_depth:])
		prev_ang_pos = np.array(self.gui_data_classes["Wing angles"].YData[-prev_depth:])
		prev_ang_vel = np.array(self.gui_data_classes["Angular velocity"].YData[-prev_depth:])
		prev_time = np.array(self.gui_data_classes["Angular velocity"].YData[-prev_depth:])

		# Update copies of data class subsets
		np.append(prev_wing_trq, wing_torque)
		np.append(prev_ang_pos, next_ang_pos)
		np.append(prev_ang_vel, next_ang_vel)

		# Calculate reward
		# target_vel = target # deg/s
		# pos_constraint = 50
		# constraint_pain = 0
		# if abs(next_ang_pos[0]) > pos_constraint:
		# 	constraint_pain = 10

		avg_abs_trq = np.mean(np.absolute(prev_wing_trq))
		avg_abs_ang_vel = np.mean(np.absolute(prev_ang_vel)) # avg of abs val of ang vel

		# min_velocity_index = np.argmin(np.absolute(prev_ang_vel[-50:])) # Where the max angle occurs, in the last 50 actions
		max_ang_vel = np.amax(np.absolute(prev_ang_vel[-50:])) # Maximum velocity
		# t_min_velocity = prev_time[min_velocity_index] # Time where max angle occurs
		max_angle = np.amax(np.absolute(prev_ang_pos[-50:]))

		try:
			obs_ang_freq = float(max_ang_vel/max_angle)
		except RuntimeWarning:
			obs_ang_freq = 2
			
		target_freq = target

		diff = abs(obs_ang_freq - target_freq)
		print(obs_ang_freq)

		# reward = [100/diff - abs(wing_torque[0]) - diff/10 - constraint_pain]
		reward = [0.01/diff - abs(wing_torque[0]) - diff]

		# Episode ends after 1000 actions
		if action_num == 1000:
			done = True
		else:
			done = False
		
		return reward, done


	def reward_8(self, combo_data, target):

		wing_torque = combo_data["action"]["Wing torques"]
		action_num = combo_data["action"]["Action num"]
		next_ang_pos = combo_data["next state"]["Wing angles"]
		next_ang_vel = combo_data["next state"]["Angular velocity"]

		# Make copies of data class subsets
		prev_depth = 100 # number of recent elements used
		if len(wing_torque) < prev_depth:
			prev_depth = len(wing_torque)

		prev_wing_trq = np.array(self.gui_data_classes["Wing torques"].YData[-prev_depth:])
		prev_ang_pos = np.array(self.gui_data_classes["Wing angles"].YData[-prev_depth:])
		prev_ang_vel = np.array(self.gui_data_classes["Angular velocity"].YData[-prev_depth:])

		# Update copies of data class subsets
		np.append(prev_wing_trq, wing_torque)
		np.append(prev_ang_pos, next_ang_pos)
		np.append(prev_ang_vel, next_ang_vel)

		# Calculate reward

		target_ang_vel = float(target) # deg/s
		bound_ang = 50. # deg

		avg_ang_pos = np.mean(np.absolute(prev_ang_pos))
		avg_abs_trq = np.mean(np.absolute(prev_wing_trq))
		avg_abs_ang_vel = np.mean(np.absolute(prev_ang_vel)) # avg of abs val of ang vel

		ang_vel = next_ang_vel[0]
		ang_pos = next_ang_pos[0]

		# ang_vel = avg_abs_ang_vel
		# ang_pos = avg_ang_pos

		if ang_vel == 0:
			ang_vel = 0.001

		# Instantaneous - uses current/next state
		reward = [-20*abs(np.log(abs(ang_vel/target_ang_vel))) 
				  - abs(wing_torque[0]) + (-ang_pos**2 + bound_ang**2)/100 + 20]

		# reward = [-(ang_vel - target_ang_vel)**2/1000 - abs(wing_torque[0])
		# 			 + (-(ang_pos)**2 + bound_ang**2)/100]

		# Episode ends after 1000 actions
		if action_num == 1000:
			done = True
		else:
			done = False

		return reward, done
		
	def reward_9(self, combo_data, target):

		wing_torque = combo_data["action"]["Wing torques"]
		action_num = combo_data["action"]["Action num"]
		next_ang_pos = combo_data["next state"]["Wing angles"]
		next_ang_vel = combo_data["next state"]["Angular velocity"]
		real_time = combo_data["next state"]["Real time"]

		# Make copies of data class subsets
		prev_depth = 100 # number of recent elements used
		if len(wing_torque) < prev_depth:
			prev_depth = len(wing_torque)

		prev_wing_trq = np.array(self.gui_data_classes["Wing torques"].YData[-prev_depth:])
		prev_ang_pos = np.array(self.gui_data_classes["Wing angles"].YData[-prev_depth:])
		prev_ang_vel = np.array(self.gui_data_classes["Angular velocity"].YData[-prev_depth:])
		prev_real_time = np.array(self.gui_data_classes["Real time"].YData[-prev_depth:])

		# Update copies of data class subsets
		np.append(prev_wing_trq, wing_torque)
		np.append(prev_ang_pos, next_ang_pos)
		np.append(prev_ang_vel, next_ang_vel)
		np.append(prev_real_time, real_time)

		# Calculate reward

		# Calculate frequency from FFT

		if prev_depth > 100:
			y_fft = np.fft.fft(prev_ang_pos)
			y_fft = y_fft[:round(len(prev_real_time)/2)]
			y_fft = np.abs(y_fft) 
			y_fft = y_fft/max(y_fft) 
			freq_axis = np.linspace(0,400,len(y_fft))

			f_loc = np.argmax(y_fft)
			f_val = freq_axis[f_loc]

			print(f"Observed frequency f = {f_val}")
		else:
			f_val = 0.001


		target_freq = float(target) # deg/s
		bound_ang = 50. # deg

		avg_ang_pos = np.mean(np.absolute(prev_ang_pos))
		avg_abs_trq = np.mean(np.absolute(prev_wing_trq))
		avg_abs_ang_vel = np.mean(np.absolute(prev_ang_vel)) # avg of abs val of ang vel

		ang_vel = next_ang_vel[0]
		ang_pos = next_ang_pos[0]

		# ang_vel = avg_abs_ang_vel
		# ang_pos = avg_ang_pos

		if ang_vel == 0:
			ang_vel = 0.001

		# Instantaneous - uses current/next state
		# reward = [-20*abs(np.log(abs(ang_vel/target_ang_vel))) 
		# 		  - abs(wing_torque[0]) + (-ang_pos**2 + bound_ang**2)/100]

		reward = [-20*abs(np.log(abs(f_val/target_freq))) 
				  - abs(wing_torque[0]) + (-(ang_pos)**2 + bound_ang**2)/100]

		# Episode ends after 1000 actions
		if action_num == 1000:
			done = True
		else:
			done = False

		return reward, done
		
