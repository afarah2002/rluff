import time

from client_test import SocketClient

if __name__ == '__main__':
	HOST = '192.168.1.192'
	# PORT = 50007
	while True:
		time.sleep(1.)
		try:
			pi_client = SocketClient(HOST)
			if pi_client:
				break
		except ConnectionRefusedError:
			print("No connection yet...")
	print("Connection found")
	while True:
		recv_data = pi_client.receive_data_pack()
		print('Received:', recv_data)
	pi_client.close()