import numpy as np

class NumPyCreator:
	def __init__(self, values = []):
		pass

	def from_list(self, lst):
		#takes in a list and returns its corresponding NumPy array.
		return np.asarray(lst)

	def from_tuple(self, tpl):
		#takes in a tuple and returns its corresponding NumPy array.
		return np.asarray(tpl)

	def from_iterable(self, itr):
		#takes in an iterable and returns an array which contains all of its elements.
		return np.fromiter(itr, float)

	def from_shape(self, shape, value = 0):
		#returns an array filled with the same value.
		#The first argument is a tuple which specifies the shape of the array,
			#and the second argument specifies the value of all the elements.
		#This value must be 0 by default.
		return np.full(shape, value)

	def random(self, shape):
		#returns an array filled with random values.
		#It takes as an argument a tuple which specifies the shape of the array.
		return np.random.uniform(size=shape)

	def identity(self, n):
		#returns an array representing the identity matrix of size n.
		return np.identity(n, dtype = None)
