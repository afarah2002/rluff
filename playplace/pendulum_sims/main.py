import queue
import itertools
import matplotlib.animation as animation
import multiprocessing
import threading
import time

import physics
import groundstation.pc_threads as pc_threads
import groundstation.gui_module.utils as gui_utils
import groundstation.gui_module.framework as gui_framework

def main():

	mass = 0.2
	length = 0.3
	damping_factor = 0.99

	physics_engine = physics.BASIC_KINEM(mass, length, damping_factor)
	test_num = "001"
	target = 90 # ang vel, deg/s
	GUI = False

	# Queues
	action_queue = multiprocessing.Queue()
	action_state_combo_queue = multiprocessing.Queue()

	# Data class init 
	data_classes = {"Wing torques" : gui_utils.GUIDataClass("action", "Wing torques", 2),
					"Observed torques" : gui_utils.GUIDataClass("action", "Observed torques", 1),
					"Wing angles" : gui_utils.GUIDataClass("next state", "Wing angles", 2),
					"Angular velocity" : gui_utils.GUIDataClass("next state", "Angular velocity", 2),
					"Real time" : gui_utils.GUIDataClass("next state", "Real time", 1),
					"Reward" : gui_utils.GUIDataClass("reward", "Reward", 1),
					"Episode reward" : gui_utils.GUIDataClass("episode reward", "Episode reward", 1)}
	


	# Threads
	# ai_test_thread = threading.Thread(target=pc_threads.Threads.ai_main_test,
	# 							 	  args=(action_queue,))
	ai_main_thread = threading.Thread(target=pc_threads.Threads.ai_main,
									  args=(test_num,
									  		target,
									  		action_state_combo_queue, 
									  		action_queue, 
									  		"infinte res",
									  		data_classes,
									  		physics_engine))
	send_actions_thread = threading.Thread(target=pc_threads.Threads.send_actions_main,
										   args=(action_queue,))
	# recv_combos_thread = threading.Thread(target=pc_threads.Threads.recv_combos_main,
	# 									  args=(pi_client, action_state_combo_queue))
	pygame_thread = threading.Thread(target=pc_threads.Threads.pygame_main,
									 args=(physics_engine, action_state_combo_queue))
	save_data_thread = threading.Thread(target=pc_threads.Threads.save_data_main,
										args=(data_classes, test_num, target, action_state_combo_queue))

	# Init GUI and animation
	if GUI:

		gui_action_figs = [gui_utils.NewMPLFigure(data_classes["Wing torques"]),
						   gui_utils.NewMPLFigure(data_classes["Observed torques"])]
		gui_state_figs = [gui_utils.NewMPLFigure(data_classes["Wing angles"]),
						  gui_utils.NewMPLFigure(data_classes["Angular velocity"])]
		gui_reward_figs = [gui_utils.NewMPLFigure(data_classes["Reward"]),
						   gui_utils.NewMPLFigure(data_classes["Episode reward"])]

		gui_figs = [gui_action_figs,
					gui_state_figs,
					gui_reward_figs]

		for gui_fig_type in gui_figs:
			lines_sets = [fig.lines for fig in gui_fig_type]

		gui_app = gui_framework.GUI(gui_figs)

		anis = [animation.FuncAnimation(fig.figure, 
										gui_utils.MPLAnimation.animate,
										interval=10, # make this large enough so it doesn't lag!
										fargs=[fig])
										for fig in list(itertools.chain.from_iterable(gui_figs))]
	# Start threads
	print("Starting")
	# ai_test_thread.start()
	ai_main_thread.start()
	# send_actions_thread.start()
	pygame_thread.start()
	save_data_thread.start()
	# recv_combos_thread.start()

	# Start GUI
	if GUI:
		gui_app.mainloop()

if __name__ == '__main__':
	main()