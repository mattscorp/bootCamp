import numpy as np
# some_file.py
import sys
# sys.path.append('../Day00')
# insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1, "/Users/ldevelle/42/Bootcamp_Python/Machine_learning/Day00/sum.py")
from day00 import vec_gradient
from day00 import sum_, dot

def prod_n_and_nplus1(n,n_plus_1):
	elem = n_plus_1[0]
	for j in range(len(n)):
		elem += n[j] * n_plus_1[j + 1]
	return elem

def predict_(theta, X):
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
	nb_features = theta.shape[0] - 1
	if n != nb_features:
		# print("Incompatible dimension match between X and theta.")
		return None
	answer = []
	for feature in range(m):
		elem = theta[0]
		for example in range(n):
			elem = elem + (X[feature][example] * theta[example + 1])
		answer.append(elem)
	pred = np.array(answer)
	# print(pred)
	return pred


X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
theta1 = np.array([[2.], [4.]])
predict_(theta1, X1)
# array([[2], [6], [10], [14.], [18.]])

X2 = np.array([[1], [2], [3], [5], [8]])
theta2 = np.array([[2.]])
predict_(theta2, X2)
# Incompatible dimension match between X and theta.
# None
X3 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8.,
80.]])
theta3 = np.array([[0.05], [1.], [1.], [1.]])
predict_(theta3, X3)
# array([[22.25], [44.45], [66.65], [88.85]])



def cost_elem_(theta, X, Y):
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
	y_pred = predict_(theta, X)
	m = Y.shape[0]
	my_sum = []
	for i in range(m):
		elem = (y_pred[i] - Y[i])**2
		elem = elem * (0.5 / m)
		my_sum.append(elem)
	answer = np.array(my_sum)
	# print(answer)
	return answer


def cost_(theta, X, Y):
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
	my_sum = cost_elem_(theta, X, Y)
	for i in range(my_sum.shape[0]):
		answer += my_sum[i]
	# print(answer)
	return answer

X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
theta1 = np.array([[2.], [4.]])
Y1 = np.array([[2.], [7.], [12.], [17.], [22.]])
cost_elem_(theta1, X1, Y1)
# array([[0.], [0.1], [0.4], [0.9], [1.6]])
cost_(theta1, X1, Y1)
# 3.0
X2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8.,
80.]])
theta2 = np.array([[0.05], [1.], [1.], [1.]])
Y2 = np.array([[19.], [42.], [67.], [93.]])
cost_elem_(theta2, X2, Y2)
# array([[1.3203125], [0.7503125], [0.0153125], [2.1528125]])
cost_(theta2, X2, Y2)
# 4.238750000000004


def fit_(theta, X, Y, alpha, n_cycle):
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
	# for iter in range(n_cycle):
	# 	print(iter)
	# 	theta = theta - (alpha * vec_gradient(X, Y, theta))
	coef = alpha / m
	for iter in range(n_cycle):
		predict_error = (predict_(theta, X)) - Y
		new_theta = [theta[0] - sum_(predict_error) * coef]
		for feature in range(n):
			correc_theta = sum_(dot(predict_error, X.T[feature])) * coef
			new_theta.append(theta[feature + 1] - correc_theta)
		theta = np.array(new_theta)
	print("New theta", theta)
	return theta

X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
Y1 = np.array([[2.], [6.], [10.], [14.], [18.]])
theta1 = np.array([[1.], [1.]])
theta1 = fit_(theta1, X1, Y1, alpha = 0.01, n_cycle=2000)
# print(theta1)
# array([[2.0023..],[3.9991..]])
# print(predict_(theta1, X1))
# array([2.0023..], [6.002..], [10.0007..], [13.99988..], [17.9990..])

X2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8.,
80.]])
Y2 = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
theta2 = np.array([[42.], [1.], [1.], [1.]])
theta2 = fit_(theta2, X2, Y2, alpha = 0.0005, n_cycle=42000)
# print(theta2)
# array([[41.99..],[0.97..], [0.77..], [-1.20..]])
# print(predict_(theta2, X2))
# array([[19.5937..], [-2.8021..], [-25.1999..], [-47.5978..]])


from mylinearregression import MyLinearRegression as MyLR

X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
Y1 = np.array([[2.], [6.], [10.], [14.], [18.]])
theta1 = np.array([[1.], [1.]])
zut = MyLR(theta1)
print(zut.fit_(theta2, X2, Y2, alpha = 0.0005, n_cycle=42000))

X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89.,
144.]])
Y = np.array([[23.], [48.], [218.]])
mylr = MyLR([[1.], [1.], [1.], [1.], [1]])
mylr.predict_(X)
# array([[8.], [48.], [323.]])
mylr.cost_elem_(X,Y)
# array([[37.5], [0.], [1837.5]])
mylr.cost_(X,Y)
# 1875.0
mylr.fit_(X, Y, alpha = 1.6e-4, n_cycle=200000)
mylr.theta
# array([[18.023..], [3.323..], [-0.711..], [1.605..], [-0.1113..]])
mylr.predict_(X)
# array([[23.499..], [47.385..], [218.079...]])
mylr.cost_elem_(X,Y)
# array([[0.041..], [0.062..], [0.001..]])
mylr.cost_(X,Y)
# 0.1056..
