import random

class Memory(object):
	# This is where the agent stores all of the results of the action that agent takes in the game
	# Used to batch train the NN
	def __init__(self, max_memory):		
		self._max_memory = max_memory # this is how many pieces of info the class can store
		# info takes the form of a tuple: (state, action, reward, next_state)
		# higher max means more memories, but too much can throw errors
		self._samples = [] # this list holds all of the memories

	def add_sample(self, sample):
		# adds the info tuple to the list of memories, cutting out the earliest memory
		# IF the memory is too full 
		self._samples.append(sample) 
		if len(self._samples) > self._max_memory:
			self._samples.pop(0)

	def sample(self, no_samples):
		# "no_samples" means "number of samples"
		# returns a random selection of samples in the samples list, 
		# but not more than the memory actually holds
		if no_samples > len(self._samples):
			return random.sample(self._samples, len(self._samples))
		else:
			return random.sample(self._samples, no_samples)