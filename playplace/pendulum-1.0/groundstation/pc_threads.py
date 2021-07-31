import time
import numpy as np

from groundstation.ai_utils.options import AITechniques

class Threads:

	def ai_main(action_state_combo_queue, 
				action_queue, 
				technique, 
				pi_client,
				pc_server):
		action_dim = 3
		state_dim = 7
		AI_infinte_res = AITechniques(action_state_combo_queue,
									  action_queue,
									  action_dim,
									  state_dim,
									  technique,
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

