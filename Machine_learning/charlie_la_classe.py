import numpy as np

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
        self.theta = np.array(theta)

    def fit_(self, X, Y, alpha=1.0, n_cycle=1000):
        """
        Description:
            Performs a fit of Y(output) with respect to X.
        Args:
            theta: has to be a numpy.ndarray, a vector of dimension (number of
        features + 1, 1).
            X: has to be a numpy.ndarray, a matrix of dimension (number of
        training examples, number of features).
            Y: has to be a numpy.ndarray, a vector of dimension (number of
        training examples, 1).
        Returns:
            new_theta: numpy.ndarray, a vector of dimension (number of the
        features +1,1).
            None if there is a matching dimension problem.
        Raises:
            This function should not raise any Exception.
        """
        if self.theta.size == 0 or X.size == 0 or Y.size == 0:
            print("size of theta or X is 0 in fit_() function")
            return None 
        elif self.theta.shape[1] != 1:
            print("error in theta dimensions in fit_() function")
            return None
        elif Y.shape[1] != 1:
            print("error in Y dimensions in fit_() function")
            return None
        if self.theta.shape[0] != X.shape[1] + 1:
            print("Incompatible dimension match between x and theta in fit_().")
            return None
        if X.shape[0] != Y.shape[0]:
            print("Inconpatible dimensions match btw x and y in fit_()")
            return None
        X = np.insert(X, 0, 1, axis=1)
        new_theta = np.empty(self.theta.shape)
        while n_cycle:
            self.theta = self.theta - alpha  * self.vec_gradient(X, Y)
            n_cycle -= 1
        return self.theta

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
        if self.theta.size == 0 or X.size == 0:
            print("size of theta or X is 0 in predict_() function")
            return None 
        elif self.theta.shape[1] != 1:
            print("error in theta dimensions in predict_() function")
            return None
        if self.theta.shape[0] != X.shape[1] + 1:
            print("Incompatible dimension match between X and theta.")
            return None
        X = np.insert(X, 0, 1, axis=1)
        return np.array([np.dot(row,self.theta) for row in X])

    def cost_elem_(self, X, Y):
        """
        Description:
            Calculates all the elements 0.5*M*(y_pred - y)^2 of the cost
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
        if self.theta.size == 1 or X.size == 0:
            print("size of theta or X is 0 in cost_elem() function")
            return None 
        elif self.theta.shape[1] != 1:
            print("error in theta dimensions in cost_elem() function")
            return None
        if self.theta.shape[0] != X.shape[1] + 1:
            print("Incompatible dimension match between X and theta in cost_elem().")
            return None
        X = np.insert(X, 0, 1, axis=1)
        return 0.5 * (1 / X.shape[0]) * np.array([(np.dot(row, self.theta) - elem) ** 2 for row, elem in zip(X, Y)])

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
        if self.theta.size == 0 or X.size == 0:
            print("size of theta or X is 0 in cost_() function")
            return None 
        elif self.theta.shape[1] != 1:
            print("error in theta dimensions in cost_() function")
            return None
        if self.theta.shape[0] != X.shape[1] + 1:
            print("Incompatible dimension match between X and theta in cost_().")
            return None
        return 0.5 * self.vec_mse(self.predict_(X), Y)

    def vec_mse(self, y, y_hat):
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
        if y.size == 0 or y_hat.size == 0:
            return None
        elif y.shape[0] != y_hat.shape[0]:
            return None
        y = y.reshape(1, y.shape[0])[0]
        y_hat = y_hat.reshape(1, y_hat.shape[0])[0]
        return  (np.dot(y_hat - y, y_hat - y) ) / y.shape[0]

    def vec_gradient(self, x, y):
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
        if x.size == 0 or y.size == 0 or self.theta.size == 0:
            print("one argument of function vec_gradient was of size zero")
            return None
        if x.shape[0] != y.shape[0]:
            print("Iconpatible dimension between y and x in vec_gradient()")    
            return None
        if x.shape[1] != self.theta.shape[0]:
            print("incompatible dimension between x and theta in vec_gradient()")
            return None
        return (np.dot(x.transpose(), np.subtract(x.dot(self.theta), y))) / x.shape[0]
