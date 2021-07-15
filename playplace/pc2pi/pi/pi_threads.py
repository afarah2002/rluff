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
			action = action_queue.get()
			if action:
				# Turn motors here
				print("Acting out", action[1])
				# Observe next state
				next_state = list(np.random.rand(1))
				combo_data = {"action" : action, "next state" : next_state}
				action_state_combo_queue.put(combo_data)
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

