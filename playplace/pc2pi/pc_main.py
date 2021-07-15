import comms_module.communicate as communicate

def main():

	# Comms
	pc_IP = "192.168.1.192"
	pc_server = communicate.Server(pc_IP)

	pi_IP = "192.168.1.95"
	pi_client = communicate.Client(pi_IP)

	# Queues
	action_queue = None
	action_state_combo_queue = None

	# Threads
	ai_thread = None
	send_actions_thread = None
	recv_combos_thread = None

	# GUI Mainloop


if __name__ == '__main__':
	main()