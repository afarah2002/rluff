import time
import numpy as np

class Threads:

	def ai_main(action_queue):
		time_step = 0
		while 1:
			time_step_name = "Time"
			time_step += .01
			action_1_name = "Wing torques"
			action_1 = list([np.sin(time_step),
							 np.cos(time_step)])

			action_2_name = "Stroke plane angle"
			action_2 = list(np.random.rand(1))

			action_data_pack = {time_step_name : time_step,
								action_1_name : action_1,
								action_2_name : action_2} 

			action_queue.put(action_data_pack)
			time.sleep(.01)

	def send_actions_main(server, action_queue):
		# The PC is the server
		while 1:
			new_data = action_queue.get()
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