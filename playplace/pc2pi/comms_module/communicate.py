import socket
import pickle
import numpy 

class Server(object):

	def __init__(self, server_IP_address, port=50007):
		self.HOST = server_IP_address
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

class Client(object):

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