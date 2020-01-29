import numpy as np

class ColorFilter:
	def invert(self, array):
		#Takes a NumPy array of an image as an argument and returns an array
		# with inverted color.
		# Authorized function : None
		# Authorized operator: -
		res = np.zeros((len(array), len(array[0]), 3))
		res[:,:,:] = 1 - array[:,:,:]
		return res

	def to_color(self, array, n):
		elem = np.zeros((len(array), len(array[0]), 3))
		elem[:,:,n] = array[:,:,n]
		return elem

	def to_blue(self, array):
		#Takes a NumPy array of an image as an argument and returns an
		# array with a blue filter.
		# Authorized function : .zeros, .shape
		# Authorized operator: None
		return self.to_color(array, 2)

	def to_green(self, array):
		#Takes a NumPy array of an image as an argument and returns an
		# array with a green filter.
		# Authorized function : None
		# Authorized operator: *
		return self.to_color(array, 1)

	def to_red(self, array):
		#Takes a NumPy array of an image as an argument and returns an array
		# with a red filter.
		# Authorized function : green, blue
		# Authorized operator: -, +
		return self.to_color(array, 0)

	def celluloid(self, array):
		#Takes a NumPy array of an image as an argument, and returns an
		# array with a celluloid shade filter.
		# The celluloid filter must display at least four thresholds of shades. Be careful! You are not
		# asked to apply black contour on the object here (you will have to, but later...), you only have to
		# work on the shades of your images.
		# Authorized function: arange, linspace
		pass
