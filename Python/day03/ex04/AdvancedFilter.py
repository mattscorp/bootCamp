
from ScrapBooker import ScrapBooker as sbk

from ColorFilter import ColorFilter as clf

class AdvancedFilter:
	@staticmethod
	def mean_blur(array, level=1):
		blurred = array.copy()
		array_red = clf.to_red(array)
		array_blue = clf.to_blue(array)
		array_green = clf.to_green(array)
		if (level < 1):
			return array.copy()
		size = level * 2 + 1
		div = (size) * (size)
		for row in range(level, array.shape[0] - level):
			for col in range(level, array.shape[1] - level):
				pos = (row - level, col - level)
				zone = sbk.crop(array_red, (size, size), pos)
				blurred[row][col][0] = zone.sum() / div
				zone = sbk.crop(array_green, (size, size), pos)
				blurred[row][col][1] = zone.sum() / div
				zone = sbk.crop(array_blue, (size, size), pos)
				blurred[row][col][2] = zone.sum() / div
		return blurred
