import time
import pickle
import numpy as np
import pathlib
import os

from groundstation.ai_utils.pendzero import PendZero

class Threads:

	def ai_main(test_num,
				target,
				N_timesteps,
				action_state_combo_queue, 
				action_queue, 
				data_classes, 
				pi_client,
				pc_server):

		# rewards = Rewards(data_classes) # Obj used to calculate rewards
		# action_dim = 1 # Pend torque
		# state_dim = 2 # Angle, ang vel
		PendZero(test_num,
				target,
				N_timesteps,
				action_state_combo_queue,
				action_queue,
				data_classes,
				pi_client,
				pc_server)

	def ai_main_test(action_queue):
		time_step = 0
		while 1:
			time_step_name = "Time"
			time_step += .1
			action_1_name = "Wing torques"
			action_1 = list([np.sin(time_step)])

			action_2_name = "Stroke plane angle"
			action_2 = list([5*np.sin(time_step)])

			action_3_name = "Stroke plane speed"
			action_3 = list(np.array(abs(np.random.rand(1))*100).clip(75,100))

			action_data_pack = {time_step_name : time_step,
								action_1_name : action_1,
								action_2_name : action_2,
								action_3_name : action_3} 

			action_queue.put(action_data_pack)
			time.sleep(.1)

	def send_actions_main(server, action_queue):
		# The PC is the server
		while 1:
			new_data = action_queue.get()
			# print("NEW DATA: ", new_data)
			if new_data:
				new_data_pack = new_data 
				server.send_data_pack(new_data_pack)
			else:
				pass

	def recv_combos_main(client, action_state_combo_queue):
		# The pi is the client
		while 1:
			combo_data_pack = client.receive_data_pack()
			# print("Received:", combo_data_pack)
			action_state_combo_queue.put(combo_data_pack)
			# print("Put data in combo")

	def save_data_main(data_classes, test_num, target, combo_queue):

		test_data_main_loc = f"test_data/{test_num}_{target}/"
		pathlib.Path(test_data_main_loc).mkdir(parents=True, exist_ok=True)
		
		print("saving")

		while True:
			combo_data = combo_queue.get()

			for data_dir, data_class in data_classes.items():
				tab_name = data_class.tab_name

				new_x = [combo_data[tab_name]["Time"]]

				if data_class.data_class_name == "Angular velocity" or data_class.data_class_name == "Wing angles":
					new_y = combo_data[tab_name][data_dir]*data_class.num_lines
				else:
					new_y = combo_data[tab_name][data_dir]

				# data_class.XData.append(new_x)
				# data_class.YData.append(new_y)

				if new_x not in data_class.XData:
					data_class.XData = np.append(data_class.XData, new_x)
					data_class.YData = np.append(data_class.YData, new_y).reshape(
									   len(data_class.XData),len(new_y))
				else:
					data_class.XData[-1] = new_x[0]
					data_class.YData[-1] = new_y

				data_loc = f"{test_data_main_loc}{data_dir}/"
				pathlib.Path(data_loc).mkdir(parents=True, exist_ok=True)
				
				x_file_loc = f"{data_loc}XData.npy"
				y_file_loc = f"{data_loc}YData.npy"
				
				np.save(x_file_loc, data_class.XData)
				np.save(y_file_loc, data_class.YData)

				# with open(x_file_loc, "wb") as xp:
				# 	pickle.dump(data_class.XData, xp)
				# with open(y_file_loc, "wb") as yp:
				# 	pickle.dump(data_class.YData, yp)



