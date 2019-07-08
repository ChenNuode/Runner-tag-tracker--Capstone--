#import libaries
import numpy as np
import cv2
import re
from text_detection import text_detection_command

def cleartextmess(s): #used to process the text returned by OCR
	smallstring = s.lower()
	#find continuous 4 digit strings in the string if any
	stuff = re.findall(r"(?<!\d)\d{4}(?!\d)", smallstring)

	#get the first one
	if len(stuff) != 0:
		return stuff[0]
	else:
		#could not find any valid runner ID numbers
		return False

def check_hit_line(boxypts,smalllist, mybuffer = 0):
	#get coords
	x1 = boxypts[0]
	y1 = boxypts[1]
	x2 = boxypts[2]
	y2 = boxypts[3] #bottom right

	#get x value of object's centroid
	ave_x = (x1 + x2)/2
	line_xdistance = abs(smalllist[1][0] - smalllist[0][0])
	distance_from_firstpt = abs(x1 - smalllist[0][0])

	# get the y value of finish line that runner must cross in order to "pass" it
	y_target = (smalllist[1][1] - smalllist[0][1])/line_xdistance*distance_from_firstpt + smalllist[0][1]
	# this is to account for diagonal finish lines 
	#	  ..
	#   ..
	# ..
	
	#make sure object's x value is within boundaries of finish line, and it's y coord is greater than the y_target
	if ((ave_x >= smalllist[0][0] and ave_x <= smalllist[1][0]) or (ave_x <= smalllist[0][0] and ave_x >= smalllist[1][0])) and y2 >= y_target - mybuffer:
		# has crossed the line
		return True
	else:
		# not crossed
		return False

def contain_within_frame(value, W, H, mode): #0 for xaxis, 1 for yaxis
	# this function keeps the values in bounding box coords within the frame
	if mode == 0:
		value = max(0,value)
		value = min(value, W)
	else:
		value = max(0,value)
		value = min(value, H)

	return value


def extract_text(boxcoords, myframe, feedback = False, savingimg = False, waittime = 500):
	# driver unction to prepare the frame for text detection and recognition

	(H, W) = myframe.shape[:2]

	startX = contain_within_frame(boxcoords[0],W,H,0)
	startY = contain_within_frame(boxcoords[1],W,H,1)
	endX = contain_within_frame(boxcoords[2],W,H,0)
	endY = contain_within_frame(boxcoords[3],W,H,1)

	#get the coordinates of frame containing the runner tag (approximately middle 50% of image)
	TAGstarty = (endY - startY)//4 * 1 + startY
	TAGendy = (endY - startY)//4 * 3 + startY
	persontagframe = myframe[ TAGstarty:TAGendy , startX:endX ]

	if feedback == True:
		cv2.imshow("Image of Interest", persontagframe)
	
	# function to detect text and OCR 
	datadict = text_detection_command(persontagframe,feedback,savingimg)

	resultslist = set()

	for item in datadict:
		smallstring = cleartextmess(item)
		if smallstring != False:
			# passed the test
			resultslist.add(smallstring)
	
	#sort list by biggest number
	resultslist = list(sorted(resultslist, reverse=True))
	oldwaittime = waittime
	
	if len(resultslist) != 0:
		#get highest
		person_num_tag = resultslist[0]
		print("Tag number:",person_num_tag)

		if feedback == False:
			cv2.imshow("Found it!", persontagframe)
			cv2.waitKey(0)
			print("Press any key to continue")
			cv2.destroyWindow("Found it!")
		waittime = 0
	else:
		# could not get runnertag

		if feedback == True:
			print("Error number tag could not be found")
		
		person_num_tag = None
		
	if feedback == True:
		if waittime == 0:
			print("Press any key to continue")
		cv2.waitKey(waittime)
		waittime = oldwaittime

	return person_num_tag

def recurse_read(objectID, rectlist, framenum_list, biglist, W, H, savingimg, maxDisappeared, feedback):
	
	print("\nSearching past frames\n")
	
	status = None

	#start counting from just b4 person got deleted from the tracker
	i = 2 + maxDisappeared

	#make sure they dont loop past the start of the tracker
	whatleft = len(rectlist)

	while status == None and i < whatleft:
		print("processing", int(framenum_list[-i]))

		#get the frame where they appear in
		myframelol = biglist[int(framenum_list[-i])]
		
		(cH, cW) = myframelol.shape[:2]
		sx = rectlist[-i][0]
		sy = rectlist[-i][1]
		ex = rectlist[-i][2]
		ey = rectlist[-i][3]

		sx = int(round(sx /W*cW))
		ex = int(round(ex /W*cW))
		sy = int(round(sy /H*cH)) 
		ey = int(round(ey /H*cH)) 

		#get the text
		status = extract_text( (sx,sy,ex,ey) , myframelol, feedback, savingimg)

		#go to previous frame and attempt to read text again
		i = i+1

	if status == None:
		print("It looks like the number tag cannot be recognised. Please key in manually (type 'nil' to ignore)")
		status = input("Number tag: ")
	
	return status

