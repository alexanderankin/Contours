import cv2
from matplotlib import pyplot as plt

### convenience plotting functions
def ss(thing):
	cv2.imwrite("about/%s.tif" % (ra.random()), thing)
	plt.subplot(121),plt.imshow(thing,cmap = 'gray')
	plt.show()

def dd(thing, square):
	extra = thing.copy()
	cv2.drawContours(extra, [square], -1, (0,255,60), 3)
	ss(extra)