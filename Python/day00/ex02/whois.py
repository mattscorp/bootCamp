import sys
string = ''
i = len(sys.argv)
if (i > 1):
	if (i == 2):
		if (sys.argv[1].isdigit()):
			nb = int(sys.argv[1]);
			if (nb == 0):
				print("I'm Zero.")
			elif (nb % 2):
				print("I'm Odd.")
			else:
				print("I'm Even.")
		else:
			print("Error")
	else:
		print("Error")
