import queue
import itertools
import matplotlib.animation as animation
import multiprocessing
import threading
import time

import comms_module.communicate as communicate
import groundstation.pc_threads as pc_threads
import groundstation.gui_module.utils as gui_utils
import groundstation.gui_module.framework as gui_framework


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
	# manager = multiprocessing.Manager()
	action_queue = multiprocessing.Queue()
	action_state_combo_queue = multiprocessing.Queue()

	# Threads
	ai_test_thread = threading.Thread(target=pc_threads.Threads.ai_main_test,
								 	  args=(action_queue,))
	ai_main_thread = threading.Thread(target=pc_threads.Threads.ai_main,
									  args=(action_state_combo_queue, 
									  		action_queue, 
									  		"infinte res",
									  		pi_client))
	send_actions_thread = threading.Thread(target=pc_threads.Threads.send_actions_main,
										   args=(pc_server, action_queue))
	recv_combos_thread = threading.Thread(target=pc_threads.Threads.recv_combos_main,
										  args=(pi_client, action_state_combo_queue))
	# print("Starting")
	# ai_thread.start()
	# send_actions_thread.start()
	# recv_combos_thread.start()

	
	gui_data_classes = {"Wing torques" : gui_utils.GUIDataClass("Wing torques", 1),
						"Stroke plane angle" : gui_utils.GUIDataClass("Stroke plane angle", 1),
						"Stroke plane speed" : gui_utils.GUIDataClass("Stroke plane speed", 1),
						"IMU Readings" : gui_utils.GUIDataClass("IMU Readings", 6),
						"Wing angles" : gui_utils.GUIDataClass("Wing angles", 1),
						"Reward" : gui_utils.GUIDataClass("Reward", 1)}
	
	# gui_figs = [gui_utils.NewMPLFigure("action", "Wing torques", gui_data_classes["Wing torques"]),
	# 			gui_utils.NewMPLFigure("action", "Stroke plane angle", gui_data_classes["Stroke plane angle"]),
	# 			gui_utils.NewMPLFigure("next state", "IMU Readings", gui_data_classes["IMU Readings"]),
	# 			gui_utils.NewMPLFigure("next state", "Wing angles", gui_data_classes["Wing angles"]),
	# 			gui_utils.NewMPLFigure("reward", "Reward", gui_data_classes["Reward"])]

	gui_action_figs = [gui_utils.NewMPLFigure("action", "Wing torques", gui_data_classes["Wing torques"]),
					   gui_utils.NewMPLFigure("action", "Stroke plane angle", gui_data_classes["Stroke plane angle"]),
					   gui_utils.NewMPLFigure("action", "Stroke plane speed", gui_data_classes["Stroke plane speed"])]
	gui_state_figs = [gui_utils.NewMPLFigure("next state", "IMU Readings", gui_data_classes["IMU Readings"]),
					  gui_utils.NewMPLFigure("next state", "Wing angles", gui_data_classes["Wing angles"])]
	gui_reward_figs = [gui_utils.NewMPLFigure("reward", "Reward", gui_data_classes["Reward"])]

	gui_figs = [gui_action_figs,
				gui_state_figs,
				gui_reward_figs]

	for gui_fig_type in gui_figs:
		lines_sets = [fig.lines for fig in gui_fig_type]

	gui_app = gui_framework.GUI(gui_figs)

	anis = [animation.FuncAnimation(fig.figure, 
									gui_utils.MPLAnimation.animate,
									interval=50,
									fargs=[fig, action_state_combo_queue])
									for fig in list(itertools.chain.from_iterable(gui_figs))]
	print("Starting")
	# ai_test_thread.start()
	ai_main_thread.start()
	send_actions_thread.start()
	# recv_combos_thread.start()

	# action_state_combo_queue.join()
	# action_queue.join()

	gui_app.mainloop()

	# GUI Mainloop


if __name__ == '__main__':
	main()