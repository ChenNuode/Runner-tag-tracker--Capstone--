# import the necessary packages
import os
import pytesseract
import cv2
from nms import nms
import numpy as np

import utils
from decode import decode
from draw import drawPolygons
from transform import four_point_transform
from math import sqrt

from collections import namedtuple
from skimage.filters import threshold_local
from skimage import segmentation
from skimage import measure
from imutils import perspective
import imutils


def text_detection(image, east, min_confidence, width, height,feedback = True,padding = 0.0):
	# load the input image and grab the image dimensions
	orig = image.copy()
	(origHeight, origWidth) = image.shape[:2]

	# set the new width and height and then determine the ratio in change
	# for both the width and height
	(newW, newH) = (width, height)
	ratioWidth = origWidth / float(newW)
	ratioHeight = origHeight / float(newH)

	# resize the image and grab the new image dimensions
	image = cv2.resize(image, (newW, newH))
	(imageHeight, imageWidth) = image.shape[:2]

	# define the two output layer names for the EAST detector model that
	# we are interested -- the first is the output probabilities and the
	# second can be used to derive the bounding box coordinates of text
	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]


	net = cv2.dnn.readNet(east)

	# construct a blob from the image and then perform a forward pass of
	# the model to obtain the two output layer sets
	blob = cv2.dnn.blobFromImage(image, 1.0, (imageWidth, imageHeight), (123.68, 116.78, 103.94), swapRB=True, crop=False)

	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)

	confidenceThreshold = min_confidence #min confidence for text detection
	nmsThreshold = 0.2

	# decode the blob info
	(rects, confidences, baggage) = decode(scores, geometry, confidenceThreshold)

	offsets = []
	thetas = []
	for b in baggage:
		# get the angle and position information for each unrotated bounding box
		offsets.append(b['offset'])
		thetas.append(b['angle'])

	##########################################################
	
	# convert rects to polygons
	polygons = utils.rects2polys(rects, thetas, offsets, newW, newH, ratioWidth, ratioHeight,padding)

	#get the indice in the list for exact polygon where text belongs, applying non-maximal suppression
	indicies = nms.polygons(polygons, confidences, nms_function=nms.malisiewicz.nms, confidence_threshold=confidenceThreshold,
							 nsm_threshold=nmsThreshold)

	boxptlist = []
	for indice in indicies:
		indicies = np.array(indicies).reshape(-1)
		
		#extracts the coords of 4 points of polygon
		drawpolys = np.array(polygons)[indicies]
		
		if feedback == True:
			name = nms.malisiewicz.nms.__module__.split('.')[-1].title()

			# draw the polygon (rotated bounding box) on frame
			drawOn = orig.copy()
			drawPolygons(drawOn, drawpolys, ratioWidth, ratioHeight, (0, 255, 0), 2)

			cv2.imshow("Detect text",drawOn)

		boxptlist.append(drawpolys)

	return boxptlist

def text_detection_command(imglol, feedback = True, tosave = False):
	padding = 0.06 #padding around text

	#get appropriate dimensions(multiple of 32) for image scaling for text detection
	feedx = 32 * round(imglol.shape[1]/32)
	feedy = 32 * round(imglol.shape[0]/32)

	#get the returned coords of the bounding box where text belongs
	yaystuff2 = text_detection(image=imglol, east='frozen_east_text_detection.pb', min_confidence=0.7, width=feedx, height=feedy, feedback = feedback, padding = padding)
	
	r_text = [] #initiate a list to store text

	for item in yaystuff2:

		yaystuff = item[0]

		warped = perspective.four_point_transform(imglol, yaystuff) # 4 point transform the rotated bounding box to get a fixed rectangle frame, correcting orientation

		warped = cv2.fastNlMeansDenoisingColored(warped,None,3,4,7,14) # cleanse image/remove noise
		warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
		warped = cv2.adaptiveThreshold(warped,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,3) #threshold to get the characters

		if feedback == True:
			cv2.imshow("Transformed text",warped)
			#cv2.waitKey(50)
	

		if tosave != False: #save the file for future ML purposes
			savenum = str(len(os.listdir('train_data/'))+1)
			filestring = "train_data/" + savenum +".jpg"
			cv2.imwrite(filestring,warped)


		#intiate tesseract, with ENG language, run with LSTM and mode is set where whole image is a word
		config = ("-l eng --oem 1 --psm 8")
		
		text = pytesseract.image_to_string(warped, config=config)
		

		if feedback == True:
			print("OCR TEXT")
			print("========")
			print(text)


		r_text.append(text)

	return r_text #returned list of text found by OCR


if __name__ == '__main__':
	inputimg = cv2.imread("sample_photos/2003.jpg")
	inputimg = cv2.resize(inputimg, (300,400))
	output = text_detection_command(inputimg, True)
	cv2.waitKey(0)