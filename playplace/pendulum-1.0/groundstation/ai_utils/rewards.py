import numpy as np

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

	def reward_2(self, combo_data):
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
		target_vel = 422 # deg/s
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

