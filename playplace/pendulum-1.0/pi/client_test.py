import socket, pickle

class SocketClient(object):

	def __init__(self, client_IP_address, port=50007, recv_size=8192):
		self.HOST = client_IP_address
		self.PORT = port
		self.recv_size = recv_size
		self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.socket.connect((self.HOST, self.PORT))

	def receive_data_pack(self):
		try:
			data = self.socket.recv(self.recv_size)
			data_arr = pickle.loads(data)
		except (pickle.UnpicklingError, ValueError, KeyError, TypeError) as error:
			print(error)
			data_arr = None
		return data_arr



	def close(self):
		self.socket.close()

# if __name__ == '__main__':
# 	HOST = '192.168.1.95'
# 	# PORT = 50007
# 	gui_client = GUISocketClient(HOST)
# 	while True:
# 		recv_data = gui_client.receive_data_pack()
# 		print('Received:', recv_data)