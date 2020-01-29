def is_string(string):
	try:
		assert type('') == type(string)
	except:
		print("Argument must be a string")
		exit()
	else:
		return string

def is_bool(string):
	try:
		assert type(True) == type(string)
	except:
		print("Argument must be a boolean")
		exit()
	else:
		return string

class GotCharacter:
	def __init__(self, first_name=None, is_alive=True):
		is_string(first_name)
		self.first_name = first_name
		is_bool(is_alive)
		self.is_alive = is_alive
	def __str__(self):
		txt = ''
		txt += "Name is: "
		txt += self.first_name
		txt += "\n"
		txt += "Player is alive : "
		txt += str(self.is_alive)
		return txt


class Stark(GotCharacter):
	"""
		Stark is the family who rule the North
	"""
	def __init__(self, first_name=None, is_alive=True):
		super().__init__(first_name=first_name, is_alive=is_alive)
		self.family_name = "Stark"
		self.house_words = "Winter is Coming"
	def __str__(self):
		txt = ''
		txt += "Family name is: "
		txt += self.family_name
		txt += "\n"
		txt += "House words are : "
		txt += self.house_words
		txt += "\n"
		txt += super().__str__()
		return txt

class Lannister(GotCharacter):
	"""
		Lannister is rich BITCH
	"""
	def __init__(self, first_name=None, is_alive=True):
		super().__init__(first_name=first_name, is_alive=is_alive)
		self.family_name = "Lannister"
		self.house_words = "Always pay his debts"
	def __str__(self):
		txt = ''
		txt += "Family name is: "
		txt += self.family_name
		txt += "\n"
		txt += "House words are : "
		txt += self.house_words
		txt += "\n"
		txt += super().__str__()
		return txt

class BG_du_89(GotCharacter):
	"""
		LES BG DE LA BOURGOGNE, sisi RPZ Guy Roux
	"""
	def __init__(self, first_name=None, is_alive=True):
		super().__init__(first_name=first_name, is_alive=is_alive)
		self.family_name = "BG du 89"
		self.house_words = "Les plus gros bg de la galaxie"
	def __str__(self):
		txt = ''
		txt += "Family name is: "
		txt += self.family_name
		txt += "\n"
		txt += "House words are : "
		txt += self.house_words
		txt += "\n"
		txt += super().__str__()
		return txt
