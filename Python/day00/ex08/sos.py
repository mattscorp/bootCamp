import sys
import string

MORSE_CODE_DICT = { 'A':'.-', 'B':'-...',
					'C':'-.-.', 'D':'-..', 'E':'.',
					'F':'..-.', 'G':'--.', 'H':'....',
					'I':'..', 'J':'.---', 'K':'-.-',
					'L':'.-..', 'M':'--', 'N':'-.',
					'O':'---', 'P':'.--.', 'Q':'--.-',
					'R':'.-.', 'S':'...', 'T':'-',
					'U':'..-', 'V':'...-', 'W':'.--',
					'X':'-..-', 'Y':'-.--', 'Z':'--..',
					'1':'.----', '2':'..---', '3':'...--',
					'4':'....-', '5':'.....', '6':'-....',
					'7':'--...', '8':'---..', '9':'----.',
					'0':'-----'}
result = ""
for word in sys.argv[1:]:
	for letter in word:
		if letter in MORSE_CODE_DICT:
			result += MORSE_CODE_DICT[letter]
		elif letter.upper() in MORSE_CODE_DICT:
			result += MORSE_CODE_DICT[letter.upper()]
		elif letter is ' ':
			result += "/"
		else:
			sys.exit("ERROR")
		result += " "
	result += "/"
	result += " "
result = result[:-2]
print(result)
