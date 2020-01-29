import numpy as np
# some_file.py
import sys
# sys.path.append('../Day00')
# insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1, "/Users/ldevelle/42/Bootcamp_Python/Machine_learning/Day00/sum.py")
from day00 import vec_gradient
from day00 import sum_, dot



# array([[19.5937..], [-2.8021..], [-25.1999..], [-47.5978..]])


class MyLinearRegression():
	"""
	Description:
	My personnal linear regression class to fit like a boss.
	"""
	def __init__(self, theta):
		"""
		Description:
			generator of the class, initialize self.
		Args:
			theta: has to be a list or a numpy array, it is a vector of
			dimension (number of features + 1, 1).
		Raises:
			This method should noot raise any Exception.
		"""
		if isinstance(theta, np.ndarray):
			self.theta = theta
		else :
			self.theta = np.array(theta)
		self.nb_features = self.theta.shape[0]
		pass

	def sum_(x, f = lambda x: x):
		"""Computes the sum of a non-empty numpy.ndarray onto wich a function is
			applied element-wise, using a for-loop.
		Args:
			x: has to be an numpy.ndarray, a vector.
			f: has to be a function, a function to apply element-wise to the
				vector.
		Returns:
			The sum as a float.
			None if x is an empty numpy.ndarray or if f is not a valid function.
		Raises:
			This function should not raise any Exception.
		"""
		if isinstance(x, np.ndarray):
			return None
		answer = 0.0
		for i in range(len(x)):
			answer += f(x[i])
		return (answer)

	def dot(x, y):
		"""Computes the dot product of two non-empty numpy.ndarray, using a
			for-loop. The two arrays must have the same dimensions.
		Args:
			x: has to be an numpy.ndarray, a vector.
			y: has to be an numpy.ndarray, a vector.
		Returns:
			The dot product of the two vectors as a float.
			None if x or y are empty numpy.ndarray.
			None if x and y does not share the same dimensions.
		Raises:
			This function should not raise any Exception.
		"""
		length_x = len(x)
		length_y = len(y)
		if (length_x != length_y):
			return None
		if not length_x:
			return None
		answer = 0.0
		for i in range(length_x):
			answer += x[i] * y[i]
		return answer

	def mat_mat_prod(x, y):
		"""Computes the product of two non-empty numpy.ndarray, using a
			for-loop. The two arrays must have compatible dimensions.
		Args:
			x: has to be an numpy.ndarray, a matrix of dimension m * n.
			y: has to be an numpy.ndarray, a vector of dimension n * p.
		Returns:
			The product of the matrices as a matrix of dimension m * p.
			None if x or y are empty numpy.ndarray.
			None if x and y does not share compatibles dimensions.
		Raises:
			This function should not raise any Exception.
		"""
		m = x.shape[0]
		if len(x.shape) == 1:
			n = 1
		else :
			n = x.shape[1]
		if y.shape[0] != n:
			return None
		if len(y.shape) == 1:
			p = 1
		else:
			p = y.shape[1]
		copy_y = y.transpose()
		answer = []
		for i in range(m):
			for j in range(p):
					answer.append(dot(x[i], copy_y[j]))
		the = np.array(answer)
		# print(the)
		return the

	def predict_(self, X):
		"""
		Description:
			Prediction of output using the hypothesis function (linear model).
		Args:
			theta: has to be a numpy.ndarray, a vector of dimension (number of
			features + 1, 1).
			X: has to be a numpy.ndarray, a matrix of dimension (number of
			training examples, number of features).
		Returns:
			pred: numpy.ndarray, a vector of dimension (number of the training
			examples,1).
			None if X does not match the dimension of theta.
		Raises:
			This function should not raise any Exception.
		"""
		m = X.shape[0] #nb_features
		n = X.shape[1] #nb_training_examples
		nb_features = self.theta.shape[0] - 1
		if n != nb_features:
			# print("Incompatible dimension match between X and theta.")
			return None
		answer = []
		for feature in range(m):
			elem = self.theta[0]
			for example in range(n):
				elem = elem + (X[feature][example] * self.theta[example + 1])
			answer.append(elem)
		pred = np.array(answer)
		# print(pred)
		return pred

	def cost_elem_(self, X, Y):
		"""
		Description:
			Calculates all the elements (0.5/M) * (y_pred - y)^2 of the cost
			function.
		Args:
			theta: has to be a numpy.ndarray, a vector of dimension (number of
			features + 1, 1).
			X: has to be a numpy.ndarray, a matrix of dimension (number of
			training examples, number of features).
		Returns:
			J_elem: numpy.ndarray, a vector of dimension (number of the training
			examples,1).
			None if there is a dimension matching problem between X, Y or theta.
		Raises:
			This function should not raise any Exception.
		"""
		y_pred = self.predict_(X)
		m = Y.shape[0]
		my_sum = []
		for i in range(m):
			elem = (y_pred[i] - Y[i])**2
			elem = elem * (0.5 / m)
			my_sum.append(elem)
		answer = np.array(my_sum)
		# print(answer)
		return answer


	def cost_(self, X, Y):
		"""
		Description:
			Calculates the value of cost function.
		Args:
			theta: has to be a numpy.ndarray, a vector of dimension (number of
			features + 1, 1).
			X: has to be a numpy.ndarray, a vector of dimension (number of
			training examples, number of features).
		Returns:
			J_value : has to be a float.
			None if X does not match the dimension of theta.
		Raises:
			This function should not raise any Exception.
		"""
		answer = 0.0
		my_sum = self.cost_elem_(X, Y)
		# answer = self.sum_(my_sum)
		for i in range(my_sum.shape[0]):
			answer += my_sum[i]
		# print(answer)
		return answer

	def fit_(self, X, Y, alpha, n_cycle):
		"""
		Description:
			Performs a fit of Y(output) with respect to X.
		Args:
				THETA	(n + 1,	1)
				X		(m,		n)
				Y		(m,		1)

				# PREDICT	(n,		1)
				# N_X		(n,		1)
				N_THETA	(m + 1,	1)
			theta: has to be a numpy.ndarray, a vector of dimension (number of
			features + 1, 1).
			X: has to be a numpy.ndarray, a matrix of dimension (number of
			training examples, number of features).
			Y: has to be a numpy.ndarray, a vector of dimension (number of
			training examples, 1).
			alpha: un float positif
			n_cycle: un integer positif
		Returns:
			new_theta: numpy.ndarray, a vector of dimension (number of the
			features +1,1).
			None if there is a matching dimension problem.
		Raises:
			This function should not raise any Exception.
		"""
		m = X.shape[0]#examples
		n = X.shape[1]#features
		coef = alpha / m
		for iter in range(n_cycle):
			predict_error = (self.predict_(X)) - Y
			new_theta = [self.theta[0] - sum_(predict_error) * coef]
			for feature in range(n):
				correc_theta = sum_(dot(predict_error, X.T[feature])) * coef
				new_theta.append(self.theta[feature + 1] - correc_theta)
			self.theta = np.array(new_theta)

		return self.theta

X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89.,
144.]])
Y = np.array([[23.], [48.], [218.]])
mylr = MyLinearRegression([[1.], [1.], [1.], [1.], [1]])
print(mylr.predict_(X))
# array([[8.], [48.], [323.]])
print(mylr.cost_elem_(X,Y))
# array([[37.5], [0.], [1837.5]])
print(mylr.cost_(X,Y))
# 1875.0
print(X)
print(Y)
print(mylr.theta)
mylr.fit_(X, Y, alpha = 1.6e-4, n_cycle=200000)
print(mylr.theta)
# array([[18.023..], [3.323..], [-0.711..], [1.605..], [-0.1113..]])
print(mylr.predict_(X))
# array([[23.499..], [47.385..], [218.079...]])
print(mylr.cost_elem_(X,Y))
# array([[0.041..], [0.062..], [0.001..]])
print(mylr.cost_(X,Y))
# 0.1056..
