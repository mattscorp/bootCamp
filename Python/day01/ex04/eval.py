def check_args(coefs, words):
	if (type(coefs) == type([])):
		for i in coefs:
			if type(i) != type(2.0):
				print("All elements of coefs must be float")
				exit()
	else:
		print("Coefs needs to be a list")
	if (type(words) == type([])):
		for i in words:
			if type(i) != type(""):
				print("All elements of words must be string")
				exit()
	else:
		print("Words needs to be a list")

class Evaluator:
	"""docstring for Evaluator."""
	@staticmethod
	def zip_evaluate(coefs, words):
		check_args(coefs, words)
		if (len(words) != len(coefs)):
			return -1
		result = 0
		for count in zip(coefs, words):
		    result += count[0] * len(count[1])
		return result

	@staticmethod
	def enumerate_evaluate(coefs, words):
		check_args(coefs, words)
		if (len(words) != len(coefs)):
			return -1
		result = 0
		enum_coefs = list(enumerate(coefs))
		enum_words = list(enumerate(words))
		for i in range(len(words)):
		    result += enum_coefs[i][1] * len(enum_words[i][1])
		return result
