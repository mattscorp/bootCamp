import numpy as np
import pandas as pd

class FileLoader:
	def load(self, path):
		#The argument of this method is the file path of the dataset to load. It must
		#display a message specifying the dimensions of the dataset (e.g. 340 x 500). The method
		#returns the dataset loaded as a pandas.DataFrame.
		data = pd.read_csv(path)
		print(path, "File shape is : ", data.shape)
		return data

	def display(self, df, n):
		#Takes a pandas.DataFrame and an integer as arguments. This
		#method displays the first n rows of the dataset if n is positive, or the last n rows if n is
		#negative.
		if n >= 0:
			print(df.head(n))
		else:
			print(df.tail(-n))
