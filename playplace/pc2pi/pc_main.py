import queue
import threading
import time

import comms_module.communicate as communicate
import groundstation.pc_threads as pc_threads

def build_pi_client(pi_IP):
	while True:
		time.sleep(1.)
		try:
			pi_client = communicate.Client(pi_IP, port=50007)
			if pi_client:
				break
		except ConnectionRefusedError:
			print("No Pi yet...")
	print("Pi found")
	return pi_client

def build_pc_server(pc_IP):
	pc_server = communicate.Server(pc_IP, port=50007)
	return pc_server

def main():

	# Comms
	pi_IP = "192.168.1.95"
	pi_client= build_pi_client(pi_IP)
	pc_IP = "192.168.1.192"
	pc_server = build_pc_server(pc_IP)

	# Queues
	action_queue = queue.Queue()
	action_state_combo_queue = queue.Queue()

	# Threads
	ai_thread = threading.Thread(target=pc_threads.Threads.ai_main,
								 args=(action_queue,))
	send_actions_thread = threading.Thread(target=pc_threads.Threads.send_actions_main,
										   args=(pc_server, action_queue))
	recv_combos_thread = threading.Thread(target=pc_threads.Threads.recv_combos_main,
										  args=(pi_client,))
	print("Starting")
	ai_thread.start()
	send_actions_thread.start()
	recv_combos_thread.start()

	# GUI Mainloop


if __name__ == '__main__':
	main()