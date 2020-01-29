
import sys

ingredients = {	'sandwich': ['ham', 'bread', 'cheese', 'tomatoes'],
				'cake': ['flour', 'sugar', 'eggs'], 
				'salad': ['avocado', 'arugula', 'tomatoes', 'spinach']}
cookbook = {'sandwich': {'ingredients': ingredients['sandwich'], 'type': 'lunch','time': 10},
			'cake': {'ingredients': ingredients['cake'], 'type': 'dessert', 'time': 60},
			'salad': {'ingredients': ingredients['salad'], 'type': 'lunch', 'time': 15}}

def print_all():
	print(cookbook.keys())
	print(cookbook.values())
	print("== The Whole dict ==\n", cookbook)

def valid_recipe_arg(recipe_name = ''):
	while ( len(recipe_name) <= 0
			or not recipe_name.isalpha()):
		recipe_name = input("your parameter is invalid\n>>>")
	return recipe_name

def add_recipe():
	recipe_name = input("What is the name of the recipe you want to add ?\n>>>")
	recipe_name = valid_recipe_arg(recipe_name)
	ingredients_list = []
	is_exit = ''
	while not (is_exit == 'exit'):
		ingredient = input("Enter an ingredient\n>>>")
		ingredients_list.append(valid_recipe_arg(ingredient))
		is_exit = input("Ingredient added, if you wanna stop adding ingredient write exit :)\n>>>")
		if (is_exit == 'exit'):
			break
	ingredients[recipe_name] = ingredients_list
	recipe_type = input("What is the type of meal you added ?\n>>>")
	recipe_type = valid_recipe_arg(recipe_type)

	recipe_timer = input("How long will it take to prepare ?\n>>>")
	while (not recipe_timer.isnumeric()
			or int(recipe_timer) < 0):
			recipe_timer = input("your parameter is invalid\n>>>")
	cookbook[recipe_name] = {'ingredients': ingredients[recipe_name], 'type': recipe_type, 'time': int(recipe_timer)}
	

def get_recipe():
	recipe_name = ''
	while (recipe_name not in cookbook.keys()):
		print("Please enter a valid recipe such as :", cookbook.keys())
		recipe_name = input(">>>")

	return recipe_name

def print_recipe(recipe_name):
	print("Here is the ultimate recipe for the " + str(recipe_name) + ':')
	print("ingredients are : ", cookbook[recipe_name]['ingredients'])
	print("You can have this for", cookbook[recipe_name]['type'])
	print("it will take ", cookbook[recipe_name]['time'], "minutes to cook")

def del_recipe(recipe_name):
	del cookbook[recipe_name]
	del ingredients[recipe_name]
	print(cookbook)

def main():
	while (1):
		print("\n1: Add a recipe\n2: Delete a recipe\n"
			+ "3: Print a recipe\n4: Print the cookbook\n5: Print all the dico\n"
			+ "6: Quit")
		opt = input()
		while (not opt.isnumeric() or (int(opt) < 1 or int(opt) > 6)):
			opt = input("please enter a valid input\n>>>")
		opt = int(opt)
		if (opt == 1):
			add_recipe()
		elif (opt == 2):
			recipe_name = get_recipe()
			del_recipe(recipe_name)
		elif (opt == 3):
			recipe_name = get_recipe()
			print_recipe(recipe_name)
		elif (opt == 4):
			print(cookbook)
		elif (opt == 5):
			print_all()
		elif (opt == 6):
			sys.exit("See you later")

main()

