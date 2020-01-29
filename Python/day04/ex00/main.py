from FileLoader import FileLoader
import numpy as np
import pandas as pd

path = "./athlete_events.csv"
n = 10
print(path, n)
MY_DataFrame = FileLoader()
open = MY_DataFrame.load(path)
MY_DataFrame.display(open, n)
print(open.columns)

def YoungestFellah(open):
	youngest = {}
	youngest['m'] = open.sort_values(['Sex','Age'], ascending=[False, True])['Age'].iloc[0]
	youngest['f'] = open.sort_values(['Sex','Age'], ascending=[True, True])['Age'].iloc[0]
	print(youngest)
	return youngest


def ProportionBySport(df, year, sport, gender = 'M'):
	good_year = df[df.Year == year]
	# print(good_year)
	good_gender = good_year[good_year.Sex == gender]
	# print(good_gender)
	no_duplicate = good_gender.drop_duplicates('Name')
	# print(no_duplicate)
	proportion = no_duplicate['Sport'].value_counts(normalize = True)
	value = proportion[sport]
	print(value)
	return value
	# print(proportion)

def howManyMedals(df, name):
	answer = {}
	athlete = df[df.Name == name]
	max_year = athlete.sort_values(['Year'], ascending=[False])['Year'].iloc[0]
	min_year = athlete.sort_values(['Year'], ascending=[True])['Year'].iloc[0]
	for year in range(min_year, max_year + 1):
		this_year = athlete.loc[athlete['Year'] == year]
		count = this_year['Medal'].value_counts().to_frame()
		if count.size:
			gold = 0
			silver = 0
			bronze = 0
			for (index_label, row_series) in count.iterrows():
				# print(index_label)
				# print(row_series.iloc[0])
				if index_label == 'Bronze' :
					bronze = row_series.iloc[0]
				elif index_label == 'Silver' :
					silver = row_series.iloc[0]
				elif index_label == 'Gold' :
					gold = row_series.iloc[0]
			answer[year] = {'G': gold, 'S': silver, 'B': bronze}
	print (answer)
	return answer

# {1992: {'G': 1, 'S': 0, 'B': 1}, 1994: {'G': 0, 'S': 2, 'B': 1}, 1998: {'G': 0, 'S': 0, 'B': 0}, 2002: {'G': 2, 'S': 0, 'B': 0}, 2006: {'G': 1, 'S': 0, 'B': 0}}

class SpatioTemporalData:
	"""docstring for SpatioTemporalData."""

	def __init__(self, data):
		self.data = data

	def where(self, year):
		answer = []
		answer.append(self.data[self.data['Year'] == year]['City'].iloc[0])
		print(answer)
		return answer

	def when(self, city):
		answer = []
		answer = self.data[self.data['City'] == city]['Year'].drop_duplicates().to_list()
		print(answer)
		return answer


def HowManyMedalsByCountry(df, Nation):
	answer = {}
	athlete = df[df.Team == Nation]
	max_year = athlete.sort_values(['Year'], ascending=[False])['Year'].iloc[0]
	min_year = athlete.sort_values(['Year'], ascending=[True])['Year'].iloc[0]
	for year in range(min_year, max_year + 1):
		this_year = athlete.loc[athlete['Year'] == year].drop_duplicates(['Medal', 'Event'])
		# Where to drop the duplicates from team sports ?
		# print(this_year)
		count = this_year['Medal'].value_counts().to_frame()
		if count.size:
			gold = 0
			silver = 0
			bronze = 0
			for (index_label, row_series) in count.iterrows():
				# print(index_label)
				# print(row_series.iloc[0])
				if index_label == 'Bronze' :
					bronze = row_series.iloc[0]
				elif index_label == 'Silver' :
					silver = row_series.iloc[0]
				elif index_label == 'Gold' :
					gold = row_series.iloc[0]
			answer[year] = {'G': gold, 'S': silver, 'B': bronze}
	print (answer)
	return answer

YoungestFellah(open)
ProportionBySport(open, 2004, 'Tennis', 'F')
howManyMedals(open, 'Kjetil Andr Aamodt')
sp = SpatioTemporalData(open)
sp.where(1896)
sp.where(2016)
sp.when('Athina')
sp.when('Paris')
HowManyMedalsByCountry(open, 'France')


open.hist(open, subset=['Age'])
#['Age', 'Height', 'Weight', 'Year'])
