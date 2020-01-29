import numpy as np
import math

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
	answer = 0.0
	if not len(x):
		return None
	for i in range(len(x)):
		answer += f(x[i])
	# print(answer)
	return (answer)

X = np.array([0, 15, -9, 7, 12, 3, -21])
sum_(X, lambda x: x)
# 7.0
X = np.array([0, 15, -9, 7, 12, 3, -21])
sum_(X, lambda x: x**2)
# 949.0

def mean(x):
	"""Computes the mean of a non-empty numpy.ndarray, using a for-loop.
	Args:
		x: has to be an numpy.ndarray, a vector.
	Returns:
		The mean as a float.
		None if x is an empty numpy.ndarray.
	Raises:
		This function should not raise any Exception.
	"""
	length = len(x)
	if not length:
		return None
	answer = sum_(x, lambda x: x) / length
	# print(answer)
	return answer

X = np.array([0, 15, -9, 7, 12, 3, -21])
mean(X)
# 1.0
X = np.array([0, 15, -9, 7, 12, 3, -21])
mean(X ** 2)
# 135.57142857142858


def variance(x):
	"""Computes the variance of a non-empty numpy.ndarray, using a for-loop.
	Args:
		x: has to be an numpy.ndarray, a vector.
	Returns:
		The variance as a float.
		None if x is an empty numpy.ndarray.
	Raises:
		This function should not raise any Exception.
	"""
	length = len(x)
	if not length:
		return None
	my_mean = mean(x)
	answer = sum_(x, lambda x: (x - my_mean)**2) / length
	# print(answer)
	return answer

X = np.array([0, 15, -9, 7, 12, 3, -21])
variance(X)
# 134.57142857142858
np.var(X)
# 134.57142857142858
variance(X/2)
# 33.642857142857146
np.var(X/2)
# 33.642857142857146

def std(x):
	"""Computes the standard deviation of a non-empty numpy.ndarray, using a
		for-loop.
	Args:
		x: has to be an numpy.ndarray, a vector.
	Returns:
		The standard deviation as a float.
		None if x is an empty numpy.ndarray.
	Raises:
		This function should not raise any Exception.
	"""
	length = len(x)
	if not length:
		return None
	answer = math.sqrt(variance(x))
	# print(answer)
	return answer

X = np.array([0, 15, -9, 7, 12, 3, -21])
std(X)
# 11.600492600378166
np.std(X)
# 11.600492600378166
Y = np.array([2, 14, -13, 5, 12, 4, -19])
std(Y)
# 11.410700312980492
np.std(Y)
# 11.410700312980492

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
	# print(answer)
	return answer
X = np.array([0, 15, -9, 7, 12, 3, -21])
Y = np.array([2, 14, -13, 5, 12, 4, -19])
dot(X, Y)
# 917.0
np.dot(X, Y)
# 917
dot(X, X)
# 949.0
np.dot(X, X)
# 949
dot(Y, Y)
# 915.0
np.dot(Y, Y)
# 915

def mat_vec_prod(x, y):
	"""Computes the product of two non-empty numpy.ndarray, using a
		for-loop. The two arrays must have compatible dimensions.
	Args:
		x: has to be an numpy.ndarray, a matrix of dimension m * n.
		y: has to be an numpy.ndarray, a vector of dimension n * 1.
	Returns:
		The product of the matrix and the vector as a vector of dimension m *
		1.
		None if x or y are empty numpy.ndarray.
		None if x and y does not share compatibles dimensions.
	Raises:
		This function should not raise any Exception.
	"""
	m = x.shape[0]
	n = x.shape[1]
	if y.shape[0] != n or y.shape[1] != 1:
		return None
	answer = []
	for i in range(m):
		answer.append(dot(x[i], y))
	the = np.array(answer)
	# print(the)
	return the

W = np.array([
    [ -8, 8, -6, 14, 14, -9, -4],
    [ 2, -11, -2, -11, 14, -2, 14],
    [-13, -2, -5, 3, -8, -4, 13],
    [ 2, 13, -14, -15, -14, -15, 13],
    [ 2, -1, 12, 3, -7, -3, -6]])
X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((7,1))
Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((7,1))
mat_vec_prod(W, X)
# array([[ 497],
#      [-356],
#      [-345],
#      [-270],
#      [ -69]])
W.dot(X)
# array([[ 497],
#      [-356],
#      [-345],
#      [-270],
#      [ -69]])
mat_vec_prod(W, Y)
# array([[ 452],
#      [-285],
#      [-333],
#      [-182],
#      [-133]])
W.dot(Y)
# array([[ 452],
#      [-285],
#      [-333],
#      [-182],
#      [-133]])

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


W = np.array([
    [ -8, 8, -6, 14, 14, -9, -4],
    [ 2, -11, -2, -11, 14, -2, 14],
    [-13, -2, -5, 3, -8, -4, 13],
    [ 2, 13, -14, -15, -14, -15, 13],
    [ 2, -1, 12, 3, -7, -3, -6]])
Z = np.array([
    [ -6, -1, -8, 7, -8],
        [ 7, 4, 0, -10, -10],
        [ 7, -13, 2, 2, -11],
        [ 3, 14, 7, 7, -4],
        [ -1, -3, -8, -4, -14],
        [ 9, -14, 9, 12, -7],
        [ -9, -4, -10, -3, 6]])

mat_mat_prod(W, Z)
# array([[ 45, 414, -3, -202, -163],
#      [-294, -244, -367, -79, 62],
#      [-107, 140, 13, -115, 385],
#      [-302, 222, -302, -412, 447],
#      [ 108, -33, 118, 79, -67]])
W.dot(Z)
# array([[ 45, 414, -3, -202, -163],
#      [-294, -244, -367, -79, 62],
#      [-107, 140, 13, -115, 385],
#      [-302, 222, -302, -412, 447],
#      [ 108, -33, 118, 79, -67]])
mat_mat_prod(Z,W)
# array([[ 148, 78, -116, -226, -76,
#                                  7, 45],
# [ -88, -108, -30, 174, 364, 109, -42],
# [-126, 232, -186, 184, -51, -42, -92],
#      [ -81, -49, -227, -208, 112, -176, 390],
#      [ 70,    3, -60, 13, 162, 149, -110],
#      [-207, 371, -323, 106, -261, -248, 83],
#      [ 200, -53, 226, -49, -102, 156, -225]])
Z.dot(W)
# array([[ 148, 78, -116, -226, -76,    7, 45],
#      [ -88, -108, -30, 174, 364, 109, -42],
#      [-126, 232, -186, 184, -51, -42, -92],
#      [ -81, -49, -227, -208, 112, -176, 390],
#      [ 70,    3, -60, 13, 162, 149, -110],
#      [-207, 371, -323, 106, -261, -248, 83],
#      [ 200, -53, 226, -49, -102, 156, -225]])


def mse(y, y_hat):
	"""Computes the mean squared error of two non-empty numpy.ndarray, using
		a for-loop. The two arrays must have the same dimensions.
	Args:
		y: has to be an numpy.ndarray, a vector.
		y_hat: has to be an numpy.ndarray, a vector.
	Returns:
		The mean squared error of the two vectors as a float.
		None if y or y_hat are empty numpy.ndarray.
		None if y and y_hat does not share the same dimensions.
	Raises:
		This function should not raise any Exception.
	"""
	if y.shape[0] != y_hat.shape[0]:
		return None
	if len(y.shape) != 1 or len(y_hat.shape) != 1: #not good if only one with len > 1 and other len = 1
		if y.shape[1] != y_hat.shape[1]:
			return None
		elif y.shape[1] != 1:
			return None
	answer = 0.0
	for i in range(y.shape[0]):
		answer += (y_hat[i] - y[i])**2
	answer = answer / y.shape[0]
	# print(answer)
	return answer

X = np.array([0, 15, -9, 7, 12, 3, -21])
Y = np.array([2, 14, -13, 5, 12, 4, -19])
mse(X, Y)
# 4.285714285714286
mse(X, X)
# 0.0


def vec_mse(y, y_hat):
	"""Computes the mean squared error of two non-empty numpy.ndarray,
		without any for loop. The two arrays must have the same dimensions.
	Args:
		y: has to be an numpy.ndarray, a vector.
		y_hat: has to be an numpy.ndarray, a vector.
	Returns:
		The mean squared error of the two vectors as a float.
		None if y or y_hat are empty numpy.ndarray.
		None if y and y_hat does not share the same dimensions.
	Raises:
		This function should not raise any Exception.
	"""
	return mse(y, y_hat)

X = np.array([0, 15, -9, 7, 12, 3, -21])
Y = np.array([2, 14, -13, 5, 12, 4, -19])
vec_mse(X, Y)
# 4.285714285714286
vec_mse(X, X)
# 0.0

def linear_mse(x, y, theta):
	"""Computes the mean squared error of three non-empty numpy.ndarray,
		using a for-loop. The three arrays must have compatible dimensions.
	Args:
		y: has to be an numpy.ndarray, a vector of dimension m * 1.
		x: has to be an numpy.ndarray, a matrix of dimesion m * n.
		theta: has to be an numpy.ndarray, a vector of dimension n * 1.
	Returns:
		The mean squared error as a float.
		None if y, x, or theta are empty numpy.ndarray.
		None if y, x or theta does not share compatibles dimensions.
	Raises:
		This function should not raise any Exception.
	"""
	m = x.shape[0]
	n = x.shape[1]
	if y.shape[0] != m or len(y.shape) != 1:
		return None
	if theta.shape[0] != n or len(theta.shape) != 1:
		return None
	MSE = 0.0
	for i in range(m):
		MSE += (dot(theta, x[i]) - y[i]) ** 2
	the = MSE / m
	# print(the)
	return the

X = np.array([
    	[ -6, -7, -9],
        [ 13, -2, 14],
        [ -7, 14, -1],
        [ -8, -4, 6],
        [ -5, -9, 6],
        [ 1, -5, 11],
        [ 9, -11, 8]])
Y = np.array([2, 14, -13, 5, 12, 4, -19])
Z = np.array([3,0.5,-6])
linear_mse(X, Y, Z)
# 2641.0

W = np.array([0,0,0])
linear_mse(X, Y, W)
# 130.71428571

def vec_linear_mse(x, y, theta):
	"""Computes the mean squared error of three non-empty numpy.ndarray,
		without any for-loop. The three arrays must have compatible dimensions.
	Args:
		y: has to be an numpy.ndarray, a vector of dimension m * 1.
		x: has to be an numpy.ndarray, a matrix of dimesion m * n.
		theta: has to be an numpy.ndarray, a vector of dimension n * 1.
	Returns:
		The mean squared error as a float.
		None if y, x, or theta are empty numpy.ndarray.
		None if y, x or theta does not share compatibles dimensions.
	Raises:
		This function should not raise any Exception.
	"""
	# # y = y.reshape(len(y), 1)
	# theta = theta.reshape(len(theta), 1)
	# # print("X: ", x)
	# # print("Theta : ", theta)
	# elem = (x @ theta) - y
	# print("elem : ", elem)
	# answer = mat_mat_prod(elem, elem)# * (1/y.shape[0])
	# answer = sum_(answer) / answer.shape[0]
	answer = linear_mse(x, y, theta)
	# print(answer)
	return answer

X = np.array([
    [ -6, -7, -9],
        [ 13, -2, 14],
        [ -7, 14, -1],
        [ -8, -4, 6],
        [ -5, -9, 6],
        [ 1, -5, 11],
        [ 9, -11, 8]])
Y = np.array([2, 14, -13, 5, 12, 4, -19])
Z = np.array([3,0.5,-6])
vec_linear_mse(X, Y, Z)
# 2641.0
W = np.array([0,0,0])
vec_linear_mse(X, Y, W)
# 130.71428571


def gradient(x, y, theta):
	"""Computes a gradient vector from three non-empty numpy.ndarray, using
		a for-loop. The two arrays must have the compatible dimensions.
	Args:
		x: has to be an numpy.ndarray, a matrice of dimension m * n.
		y: has to be an numpy.ndarray, a vector of dimension m * 1.
		theta: has to be an numpy.ndarray, a vector n * 1.
	Returns:
		The gradient as a numpy.ndarray, a vector of dimensions n * 1.
		None if x, y, or theta are empty numpy.ndarray.
		None if x, y and theta do not have compatible dimensions.
	Raises:
		This function should not raise any Exception.
	"""
	m = x.shape[0]
	n = x.shape[1]
	if y.shape[0] != m or theta.shape[0] != n:
		return None
	my_sum = []
	for i in range(m):
		my_sum.append((dot(x[i], theta) - y[i]) / m)
	my_sum = np.array(my_sum).reshape(1, m)
	# print(my_sum)
	answer = mat_mat_prod(my_sum, x)
	# print(answer)
	return my_sum

X = np.array([
    [ -6, -7, -9],
        [ 13, -2, 14],
        [ -7, 14, -1],
        [ -8, -4, 6],
        [ -5, -9, 6],
        [ 1, -5, 11],
        [ 9, -11, 8]])
Y = np.array([2, 14, -13, 5, 12, 4, -19])
Z = np.array([3,0.5,-6])
gradient(X, Y, Z)
# array([ -37.35714286, 183.14285714, -393.        ])

W = np.array([0,0,0])
gradient(X, Y, W)
# array([ 0.85714286, 23.28571429, -26.42857143])

gradient(X, X.dot(Z), Z)
# grad(X, X.dot(Z), Z)
# array([0., 0., 0.])


def vec_gradient(x, y, theta):
	"""Computes a gradient vector from three non-empty numpy.ndarray,
		without any for-loop. The three arrays must have the compatible dimensions.
	Args:
		x: has to be an numpy.ndarray, a matrice of dimension m * n.
		y: has to be an numpy.ndarray, a vector of dimension m * 1.
		theta: has to be an numpy.ndarray, a vector n * 1.
	Returns:
		The gradient as a numpy.ndarray, a vector of dimensions n * 1, containg
		the result of the formula for all j.
		None if x, y, or theta are empty numpy.ndarray.
		None if x, y and theta do not have compatible dimensions.
	Raises:
		This function should not raise any Exception.
	"""
	m = x.shape[0]
	n = x.shape[1]
	if y.shape[0] != m or theta.shape[0] != n:
		return None
	theta = theta.reshape(n, 1)
	elem = mat_mat_prod(x, theta) - y
	answer = mat_mat_prod(x.T, elem.reshape(m, 1)) / m
	# print(answer)
	return answer

X = np.array([
 [ -6, -7, -9],
 [ 13, -2, 14],
 [ -7, 14, -1],
 [ -8, -4, 6],
 [ -5, -9, 6],
 [ 1, -5, 11],
 [ 9, -11, 8]])
Y = np.array([2, 14, -13, 5, 12, 4, -19])
Z = np.array([3,0.5,-6])
vec_gradient(X, Y, Z)
# array([ -37.35714286, 183.14285714, -393. ])
W = np.array([0,0,0])
vec_gradient(X, Y, W)
# array([ 0.85714286, 23.28571429, -26.42857143])
vec_gradient(X, X.dot(Z), Z)
# array([0., 0., 0.])
