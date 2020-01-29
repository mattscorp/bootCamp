from recipe import Recipe
from book import Book

a = Recipe(name = "Hamburger", cooking_lvl="1", ingredients=["e","r"], description="Super bon", recipe_type="lunch")
c = Recipe(name = "Bouff", cooking_lvl="1", ingredients=["e","r"], description="jc bon", recipe_type="lunch")
d = Recipe(name = "uff", cooking_lvl="1", ingredients=["e","r"], description="jc bon", recipe_type="starter")
b = Book("Super LIIIIVRE")
print("1")
b.add_recipe(a)
b.add_recipe(c)
b.add_recipe(d)
print("1")
print(b.get_recipe_by_name(a.name))
print("1")
b.get_recipes_by_types(a.recipe_type)
b.get_recipes_by_types(d.recipe_type)
print("1")
