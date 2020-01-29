import sys
import math
import random


def instruction():
	print("This is an interactive guessing game!")
	print("You have to enter a number between 1 and 99 to find out the secret number.")
	print("Type 'exit' to end the game.")
	print("Good luck!")

number = random.randint(1, 99)

def get_nb(guess = -10):
	print("What's your guess between 1 and 99?")
	guess = input(">>>  ")
	while (not guess.isdigit() or int(guess) < 1 or int(guess) > 99):
		if (guess == 'exit'):
			sys.exit("GoodBye !")
		if (not guess.isdigit()):
			print("Please enter a number")
		else:
			print("Your number must be between 1 and 99")
		guess = input(">>>  ")
	return int(guess)

instruction()

guess = -10
tries = 1
guess = get_nb()
while (number != guess):
	if (number < guess):
		print("Too high!")
	else:
		print("Too low!")
	guess = get_nb()
	tries += 1

print("Congratulations, you've got it!")
if (number == 42):
	print("The answer to the ultimate question of life," ,
			" the universe and everything is 42.")
if (tries > 1):
	print("You won in " + str(tries) + " attempts!")
else:
	print("Congratulations! You got it on your first try!")
