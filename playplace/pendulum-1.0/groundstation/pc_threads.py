import time
import pickle
import numpy as np
import pathlib
import os

from groundstation.ai_utils.options import AITechniques
from groundstation.ai_utils.rewards import Rewards 

class Threads:

	def ai_main(test_num,
				target,
				action_state_combo_queue, 
				action_queue, 
				technique,
				data_classes, 
				pi_client,
				pc_server):

		rewards = Rewards(data_classes) # Obj used to calculate rewards
		action_dim = 1 # Pend torque
		state_dim = 2 # Angle, ang vel
		AI_infinte_res = AITechniques(test_num,
									  target,
									  action_state_combo_queue,
									  action_queue,
									  action_dim,
									  state_dim,
									  technique,
									  rewards,
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
		

		while True:
			combo_data = combo_queue.get()

			# for data_dir, data_class in data_classes.items():

			for data_dir, data_class in data_classes.items():
				tab_name = data_class.tab_name

				new_x = combo_data[tab_name]["Time"]
				new_y = combo_data[tab_name][data_dir]

				data_class.XData.append(new_x)
				data_class.YData.append(new_y)

				# print(data_class.YData)

				data_loc = f"{test_data_main_loc}{data_dir}/"
				pathlib.Path(data_loc).mkdir(parents=True, exist_ok=True)
				
				x_file_loc = f"{data_loc}XData.txt"
				y_file_loc = f"{data_loc}YData.txt"
				
				with open(x_file_loc, "wb") as xp:
					pickle.dump(data_class.XData, xp)
				with open(y_file_loc, "wb") as yp:
					pickle.dump(data_class.YData, yp)



