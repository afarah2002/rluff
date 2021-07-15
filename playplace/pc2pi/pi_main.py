import queue
import threading
import time

import comms_module.communicate as communicate
import pi.pi_threads as pi_threads

def build_pi_server(pi_IP):
	pi_server = communicate.Server(pi_IP, port=50007)
	return pi_server

def build_pc_client(pc_IP):
	while True:
		time.sleep(1.)
		try:
			pc_client = communicate.Client(pc_IP, port=50007)
			if pc_client:
				break
		except ConnectionRefusedError:
			print("No PC yet...")
	print("PC found")
	return pc_client

def main():

	# Comms
	pi_IP = "192.168.1.95"
	pi_server = communicate.Server(pi_IP, port=50007)
	pc_IP  = "192.168.1.192"
	pc_client = build_pc_client(pc_IP)

	# Queues
	action_queue = queue.Queue()
	action_state_combo_queue = queue.Queue()

	'''
	Sequence of events
	Thread 1:
	 - Receive actions from PC server 
	 - Post actions to action queue
	Thread 2:
	 - Sample action from action queue
	 - Act out action 
	 - Observe next_state 
	 - Post {"action" : {"action 1" : [timestep, [action 1]],
	 					 "action 2" : [timestep, [action 2]]},
	 		 "next state" : {"state 1" : [timestep, [state 1]],
	 		 				 "state 2" : [timestep, [state 2]]}
	 		}
	 	  to combo queue
	Thread 3:
	 - Send combos in combo queue to PC client
	'''	

	# Threads
	recv_actions_thread = threading.Thread(target=pi_threads.Threads.recv_actions_main,
										   args=(pc_client, action_queue))
	act_n_obs_thread = threading.Thread(target=pi_threads.Threads.act_n_obs_main,
										args=(action_queue, action_state_combo_queue))
	send_combos_thread = threading.Thread(target=pi_threads.Threads.send_combos_main,
										  args=(pi_server,action_state_combo_queue))

	print("Starting")
	recv_actions_thread.start()
	act_n_obs_thread.start()
	send_combos_thread.start()

if __name__ == '__main__':
	main()