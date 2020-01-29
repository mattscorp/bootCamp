from generator import generator

text="Le Lorem Ipsum est Le du Le simplement du faux texte."
for word in generator(text, sep=" ", option="shuffle"):
	print(word)
