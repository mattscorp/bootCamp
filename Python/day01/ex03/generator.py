import datetime
import string

# function prototype
def generator(text, sep=" ", option=None):
	""""
	The function can take an optional argument.
	The options are:
	• "shuffle": shuffle the list of words.
	• "unique": return a list where each word appears only once.
	• "ordered": alphabetically sort the words.
	"""
	try:
		assert type(text) is type('')
		assert type(sep) is type('')
	except:
		print("Error: Function takes string as input")
		exit()

	tab = []
	tab = text.split(sep)
	if (option == None):
		for elem in tab:
			yield elem
	elif (option == "shuffle"):
		while len(tab):
			double = 0
			t = str(datetime.datetime.now())
			(h, seed) = t.split('.')
			rand = int(seed) % int(len(tab))
			yield tab[rand]
			tab.pop(rand)

	elif (option == "unique"):
		uniq = []
		for elem in tab:
			double = 0
			for el in uniq:
				if (el == elem):
					double += 1
			if (double == 0):
				uniq.append(elem)
				yield elem
	elif (option == "ordered"):
		tab.sort()
		for elem in tab:
			yield elem
	else:
		print("Error: Option can be: None, shuffle, ordered, unique")
