import datetime
from recipe import Recipe

class Book:
	def __init__(self, name = None, recipes_list = {"starter": [], "lunch": [], "dessert": [], }):
		try:
			assert type('') == type(name)
		except:
			print("Name must be a string")
			exit()
		else:
			self.name = name
		self.recipes_list = recipes_list
		self.creation_date = datetime.time()
		self.last_update = datetime.time()

	def get_recipe_by_name(self, name):
		for key, value in self.recipes_list.items():
			for v_name in value:
				if v_name.name == name:
					return v_name.name

	def get_recipes_by_types(self, recipe_type):
		try:
			assert type('') == type(recipe_type)
			assert recipe_type is "starter" or recipe_type is "lunch" or recipe_type is "dessert"
		except:
			print("recipe_type must be a string of value lunch, starter or dessert")
			exit()
		else:
			# print(self.recipes_list[recipe_type])
			# for elem in self.recipes_list[recipe_type]:
			# 	print(elem)
			return(self.recipes_list[recipe_type])

	def add_recipe(self, recipe):
		tmp = Recipe()
		try:
			assert type(recipe) == type(tmp)
		except:
			print("recipe must be an object of type Recipe")
			exit()
		else:
			try:
				assert type('') == type(recipe.recipe_type)
				assert recipe.recipe_type is "starter" or recipe.recipe_type is "lunch" or recipe.recipe_type is "dessert"
			except:
				print("recipe_type must be a string of value lunch, starter or dessert")
				exit()
			else:
				self.recipes_list[recipe.recipe_type].append(recipe)
		self.last_update = datetime.time()
