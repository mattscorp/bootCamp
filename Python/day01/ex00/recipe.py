class Recipe:
	def __init__(self, name = "None", cooking_lvl = 1, ingredients = [], description = "None", recipe_type = "lunch"):
		tmp = ""
		try:
			assert type(tmp) == type(name)
		except:
			print("Name must be a string")
			exit()
		else:
			self.name = name

		nb = 9
		try:
			cooking_lvl = int(cooking_lvl)
			assert cooking_lvl in range(1,5)
		except ValueError:
			print("Cooking level must be an int")
			exit()
		except AssertionError:
			print("Cooking level must be in range 1 to 5")
			exit()
		else:
			self.cooking_lvl = cooking_lvl

		lst_tmp = []
		try:
			assert type(lst_tmp) == type(ingredients)
			for ing in ingredients:
				assert type(tmp) == type(ing)
		except:
			print("Ingredients must be list of strings")
			exit()
		else:
			self.ingredients = ingredients

		try:
			assert type(tmp) == type(description)
		except:
			print("Description must be a string")
			exit()
		else:
			self.description = description

		try:
			assert type(tmp) == type(recipe_type)
			assert recipe_type is "starter" or recipe_type is "lunch" or recipe_type is "dessert"
		except:
			print("recipe_type must be a string of value lunch, starter or dessert")
			exit()
		else:
			self.recipe_type = recipe_type


	def __str__(self):
		txt = ''
		txt += "Name = "
		txt += self.name
		txt += "\n"
		txt += "cooking_lvl = "
		txt += str(self.cooking_lvl)
		txt += "\n"
		txt += "ingredients = "
		for idx, ing in enumerate(self.ingredients):
			txt += str(ing)
			if (idx < len(self.ingredients) - 1):
				txt += ", "
			else:
				txt += "\n"
		txt += "Description = "
		txt += str(self.description)
		txt += "\n"
		txt += "recipe_type = "
		txt += str(self.recipe_type)
		txt += "\n"
		return txt
