import time
import numpy as np

class Threads:

	def recv_actions_main(client, action_queue):
		# The PC is the client
		while 1:
			action_data_pack = client.receive_data_pack()
			print("Received:", action_data_pack)
			action_queue.put(action_data_pack)

	def act_n_obs_main(action_queue, action_state_combo_queue):
		while 1:
			action_data = action_queue.get()
			if action_data:
				# Turn motors here
				print("Acting out", action_data)
				# Observe next state
				time_step_name = "Time"
				time_step = action_data[time_step_name]

				state_1_name = "IMU Readings"
				state_1 = list(np.random.rand(6))

				state_2_name = "Wing angles"
				state_2 = list(np.random.rand(1))

				state_data = {time_step_name : time_step,
							  state_1_name : state_1,
							  state_2_name : state_2}

				combo_data_pack = {"action" : action_data, 
								   "next state" : state_data}

				action_state_combo_queue.put(combo_data_pack)
			else:
				pass


	def send_combos_main(server, action_state_combo_queue):
		# The pi is the server
		while 1:
			combo_data_pack = action_state_combo_queue.get()
			if combo_data_pack:
				server.send_data_pack(combo_data_pack)
			else:
				pass

