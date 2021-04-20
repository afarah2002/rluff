import gym
import time
import tensorflow.compat.v1 as tf
import random
import numpy as np
import math
import matplotlib.pyplot as plt

class Model:
	# This is where TensorFlow does its thing
	def __init__(self, num_states, num_actions, batch_size):
		'''
		Given in the context of the problem we are trying to solve
		(How the robot can observe and act in its environment)
		'''
		self._num_states = num_states # the number of states in the env
		self._num_actions = num_actions # the number of possible actions
		self._batch_size = batch_size 
		#-----------------placeholders-----------------#
		'''
		a placeholder is a pre-fromatted container 
		into which content can be placed
		'''
		self._states = None
		self._actions = None
		#-----------------output operations-----------------#
		self._logits = None
		self._optimizer = None
		self._var_init = None
		#-----------------set up the model-----------------#
		'''
		This runs the model class that follows the init func
		'''
		self._define_model()

	def _define_model(self):
		#-----------------TF placeholders-----------------#
		# holds the state data
		self._states = tf.placeholder(shape=[None,self._num_states], dtype=tf.float32)
		# holds the training data
		self._q_s_a = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32)
		# create a few fully connected layers
		fc1 = tf.layers.dense(self._states, 50, activation=tf.nn.relu)
			# this first layer takes in the states as input into 50 nodes
			# and uses the rectified linear unit activation to relate 
			# the input layer to fc1
		'''
		RELU returns 0 if it gets a negative input, but returns the positive 
		value x it gets, introducing non-linearity into the model
		Read more here:
		https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0 
			'''
		fc2 = tf.layers.dense(fc1, 50, activation=tf.nn.relu)
			# another layer, takes in the 50 fc1 outputs as input, 
			# does another rectified linear unit activation 
		self._logits = tf.layers.dense(fc2, self._num_actions)
		'''
					0 0 0 0 .... however many states there are
			fc1	   /|/|/|/|  
				  0000000000 ... 50 nodes
			fc2	  |/|/|/|/|/ 
				  0000000000 ... 50 more nodes
		logits	  |  |  |  /
					0 0 0    ... outputs to however many 
								 actions the agent can take

		'''
		loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
			# gets the MSE loss between the action values output by the NN
			# (logits) and the action values of the training data (Q)
		self._optimizer = tf.train.AdamOptimizer().minimize(loss)
			# uses a generic Adam optimizer to optimize (minimize)
			# the loss function
		self._var_init = tf.global_variables_initializer()
			# TF boiler plate variable

	# There need to be functions for prediction and training
	def predict_one(self, state, sess):
		# Calls the logits operation and returns the output of the NN
		# Called whenever the agent needs to choose an action
		return sess.run(self._logits, feed_dict={self._states:
													state.reshape(1, self._num_states)})

	def predict_batch(self, states, sess):
		# Given a whole batch of input states, predits a whole batch of 
		#	output states
		return sess.run(self._logits, feed_dict={self._states: states})

	def train_batch(self, sess, x_batch, y_batch):
		# Runs the batch trainer
		sess.run(self._optimizer, feed_dict={self._states: x_batch, 
											 self._q_s_a: y_batch})