import cv2
import numpy as np
import math

# https://code.google.com/p/pythonxy/source/browse/src/python/OpenCV/DOC/samples/python2/squares.py?spec=svn.xy-27.cd6bf12fae7ae496d581794b32fd9ac75b4eb366&repo=xy-27&r=cd6bf12fae7ae496d581794b32fd9ac75b4eb366
def angle_cos(p0, p1, p2):
		d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
		return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

# ranking of shapes
def rank(square, img):
	width = img.shape[0]
	height = img.shape[1]
	formatted = np.array([[s] for s in square], np.int32)
	x,y,wid,hei = cv2.boundingRect(formatted)
	max_distance_from_center = math.sqrt(((width / 2))**2 + ((height / 2))**2)
	distance_from_center = math.sqrt(((x + wid / 2) - (width / 2))**2 + ((y + hei / 2) - (height / 2))**2)

	height_above_horizontal = (height / 2) - y if y + hei > height / 2 else hei
	width_left_vertical = (width / 2) - x if x + wid > width / 2 else wid
	horizontal_score = abs(float(height_above_horizontal) / hei - 0.5) * 2
	vertical_score = abs(float(width_left_vertical) / wid - 0.5) * 2

	if cv2.contourArea(formatted) / (width * height) > 0.98:
		return 5 # max rank possible otherwise - penalize boxes that are the whole image heavily
	else:
		bounding_box = np.array([[[x,y]], [[x,y+hei]], [[x+wid,y+hei]], [[x+wid,y]]], dtype = np.int32)
		# every separate line in this addition has a max of 1
		return (distance_from_center / max_distance_from_center +
			cv2.contourArea(formatted) / cv2.contourArea(bounding_box) +
			cv2.contourArea(formatted) / (width * height) +
			horizontal_score +
			vertical_score)

# visual alternative to drawing lines
def mask_image(img, square, opacity = 0.80):
	overlay = img.copy()
	cv2.fillPoly(overlay, [square], (255, 255, 255))
	inverse_overlay = cv2.bitwise_not(overlay)
	img2 = cv2.bitwise_xor(inverse_overlay, img)
	cv2.addWeighted(img2, opacity, img, 1 - opacity, 0, img)

