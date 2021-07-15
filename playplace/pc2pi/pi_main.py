import comms_module.communicate as communicate

def main():

	# Comms
	pi_IP = "192.168.1.95"
	pi_server = communicate.Server(pi_IP)

	pc_IP  = "192.168.1.192"
	pc_server = communicate.Client(pc_IP)

	# Queues
	action_queue = None
	action_state_combo_queue = None

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

if __name__ == '__main__':
	main()