import time

class Threads:

	def ai_main(action_queue):
		time_step = 0
		while 1:
			time_step += 1
			action_queue.put([time_step])
			time.sleep(.01)

	def send_actions_main(server, action_queue):
		# The PC is the server
		while 1:
			new_data = action_queue.get()
			if new_data:
				data_pack = ["Action", new_data] 
				server.send_data_pack(data_pack)
			else:
				pass

	def recv_combos_main(client):
		# The pi is the client
		while 1:
			combo_data_pack = client.receive_data_pack()
			print("Received:", combo_data_pack)