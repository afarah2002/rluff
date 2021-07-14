import socket, pickle, json
import time
import numpy as np 


class GUISocketServer(object):

	def __init__(self, pi_IP_address, port=50007):
		self.HOST = pi_IP_address
		self.PORT = port
		self.gui_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.gui_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self.gui_socket.bind((self.HOST, self.PORT))
		self.gui_socket.listen(1)
		self.conn, self.addr = self.gui_socket.accept()

	def send_data_pack(self, data_pack):
		data_string = pickle.dumps(data_pack, -1)
		self.conn.send(data_string)

	def close(self):
		self.gui_socket.close()

if __name__ == '__main__':
	HOST = '192.168.1.95'
	# PORT = 50007
	gui_server = GUISocketServer(HOST)
	arr = {"joint torques" : [0, [1,2,3,4,5,6]], 
			"stroke plane"  : [0, [1]]}
	while 1:
		arr["joint torques"][0] += .1
		arr["joint torques"][1] = np.sin(arr["joint torques"][0] + np.random.rand(6))
		arr["stroke plane"][0] += .1
		arr["stroke plane"][1] = np.cos(arr["stroke plane"][0] + np.random.rand(1))

		print(arr)
		gui_server.send_data_pack(arr)
		time.sleep(0.02)
	gui_server.close()