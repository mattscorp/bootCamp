from ImageProcessor import ImageProcessor
from ScrapBooker import ScrapBooker

imp = ImageProcessor()
arr = imp.load("./42AI.png")
scrap = ScrapBooker()
#arr = scrap.crop(arr, [100, 100], [100, 100])
# arr = scrap.thin(arr, 2, 0)
arr = scrap.mosaic(arr, [2,3])
imp.display(arr)
