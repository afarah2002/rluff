import socket, pickle

class GUISocketClient(object):

	def __init__(self, pi_IP_address, port=50007, recv_size=8192):
		self.HOST = pi_IP_address
		self.PORT = port
		self.recv_size = recv_size
		self.gui_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.gui_socket.connect((self.HOST, self.PORT))

	def receive_data_pack(self):
		data = self.gui_socket.recv_pyobj(self.recv_size)
		# data = self.gui_socket.recv(self.recv_size)
		# data_arr = pickle.loads(data)
		return data_arr



	def close(self):
		self.gui_socket.close()

# if __name__ == '__main__':
# 	HOST = '192.168.1.95'
# 	# PORT = 50007
# 	gui_client = GUISocketClient(HOST)
# 	while True:
# 		recv_data = gui_client.receive_data_pack()
# 		print('Received:', recv_data)