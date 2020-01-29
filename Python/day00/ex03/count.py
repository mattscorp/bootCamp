import sys
import string

def text_analyzer(txt=''):
	if not txt:
		print("What is the text to analyze ?")
		txt = raw_input(">> ")

	print "The text contains", len(txt), "characters:"
	print "- ", sum(1 for c in txt if c.isupper()), "upper letters"
	print "- ", sum(1 for c in txt if c.islower()), "lower letters"
	data = sum(1 for c in txt if string.punctuation.find(c) != -1)
	print "- ", data, "punctuation marks"
	print "- ", sum(1 for c in txt if c.isspace()), "spaces"
