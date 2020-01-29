import sys
import string

if (len(sys.argv) == 3):
	a = int(sys.argv[1])
	b = int(sys.argv[2])
	print("Sum: ", int (a + b))
	print("Difference: ", int (a - b))
	print("Product: ", int (a * b))
	if (b != 0):
		print("Quotient: ", float (a / b))
		print("Remainder: ", int (a % b))
	else:
		print("Quotient: ", "Error")
		print("Remainder: ", "Error")
else:
	print("Usage:\t\tpython operations.py nb nb")
	print("Example:\tpython operations.py 10 37")
