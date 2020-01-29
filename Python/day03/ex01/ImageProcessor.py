import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class ImageProcessor:
	def load(wesh, path):
		#opens the .png file specified by the path argument and returns an array
		#with the RGB values of the image pixels.
		#It must display a message specifying the dimensions of the image (e.g. 340 x 500).
		img = mpimg.imread(path)
		print("Loading image of dimensions " + str(len(img)) + " x " + str(len(img[0])))
		return img

	def display(bg_du_89, array):
		#takes a NumPy array as an argument and displays the corresponding
		#RGB image.
		imgplot = plt.imshow(array)
		plt.show()
		pass
