import tensorflow as tf
import numpy as np


#-----------------ACTOR NN-----------------#
class Actor:
	def __init__(self, state, state_dims, action_dims, action_bounds, scope):
		self.state = state
		self.state_dims = state_dims
		self.action_dims = action_dims
		self.scope = scope
		self.fc_layer_size = 256
		# state layer
		# layer 1 
		# layer 2
		# layer 3
		# action layer
		with tf.variable_scope(scope):
			self.input_layer = tf.layers.dense(self.state, 
											   self.fc_layer_size, 
											   activation='relu',
											   kernel_initializer=tf.contrib.layers.xavier_initializer(),
											   bias_initializer=tf.zeros_initializer())
			
			self.fc1 = tf.layers.dense(self.input_layer, 
									   self.fc_layer_size, 
									   activation='relu',
									   kernel_initializer=tf.contrib.layers.xavier_initializer(),
									   bias_initializer=tf.zeros_initializer())

			self.fc2 = tf.layers.dense(self.fc1, 
									   self.fc_layer_size, 
									   activation='relu',
									   kernel_initializer=tf.contrib.layers.xavier_initializer(),
									   bias_initializer=tf.zeros_initializer())

			self.fc3 = tf.layers.dense(self.fc2, 
									   self.fc_layer_size, 
									   activation='relu',
									   kernel_initializer=tf.contrib.layers.xavier_initializer(),
									   bias_initializer=tf.zeros_initializer())

			self.output = tf.layers.dense(self.fc3, 
										  self.fc_layer_size, 
										  activation='relu',
										  kernel_initializer=tf.contrib.layers.xavier_initializer(),
										  bias_initializer=tf.zeros_initializer())

#-----------------CRITIC NN-----------------#
class Critic:
	def __init__(self, state, action, state_dims, action_dims):
		pass