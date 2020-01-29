t = (2, 4)

com = 0

print("The " + str(len(t)) + ' ' + "numbers, are:", end = '')
for i in t:
	if com:
		print(',', end = '')
	print(" " + str(i), end = '')
	com = 1
