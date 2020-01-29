from eval import Evaluator

#  Code a class Evaluator, that has two static functions named: zip_evaluate and
# enumerate_evaluate.
# The goal of these 2 functions is to calculate the sum of the lengths of every words of a given
# list weighted by a list a coefs.
# The lists coefs and words have to be the same length. If this is not the case, the function
# should return -1.
# You have to obtain the desired result using zip in the zip_evaluate function, and with
# enumerate in the enumerate_evaluate function.
words = ["Le", "Lorem", "Ipsum", "est", "simple"]
coefs = [1.0, 2.0, 1.0, 4.0, 0.5]
res = 0
res = Evaluator.zip_evaluate(coefs, words)
print(res)
words = ["Le", "Lorem", "Ipsum", "n'", "est", "pas", "simple"]
coefs = [0.0, -1.0, 1.0, -12.0, 0.0, 42.42]
res = 0
res = Evaluator.zip_evaluate(coefs, words)
print(res)
words = ["Le", "Lorem", "Ipsum", "n'", "est", "pas", "simple"]
coefs = [0.0, -1.0, 1.0, -12.0, 0.0, 42.42]
res = Evaluator.enumerate_evaluate(coefs, words)
print(res)
words = ["Le", "Lorem", "Ipsum", "est", "simple"]
coefs = [1.0, 2.0, 1.0, 4.0, 0.5]
res = Evaluator.enumerate_evaluate(coefs, words)
print(res)
