import sys
import string
import math


if (len(sys.argv) is not 3):
	sys.exit("Wrong number of arguments")
if (sys.argv[1].isdigit()):
	sys.exit("Argument 1 cant be a number")
if (not sys.argv[2].isdigit() or int(sys.argv[2]) <= 0):
	sys.exit("Argument 2 needs to be a positive number")
words = str(sys.argv[1])
max_size = int(sys.argv[2])
result=[]
for c in string.punctuation:
	words = words.replace(c, '')
words = words.split()
for word in words:
	if (len(word) > max_size):
		result.append(word)

print(result)
