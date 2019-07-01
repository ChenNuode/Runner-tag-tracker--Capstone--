# USAGE
# To read and write back out to video:
# python3 people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4 --output output/output_01.avi
#
# python3 people_counter.py -i videos/bigrun_2.mov -o output/output_01.avi

# To read from webcam and write back out to disk:
# python3 people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
#	--output output/webcam_output.avi

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from text_recognition import *
import re

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", type=str, default="mobilenet_ssd/MobileNetSSD_deploy.prototxt",
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", type=str, default="mobilenet_ssd/MobileNetSSD_deploy.caffemodel",
	help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str,
	help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.8,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=5,
	help="# of skip frames between detections")
args = vars(ap.parse_args())

text_recog_args = {"width":320, "height":640,"east":"frozen_east_text_detection.pb", "padding":0.2, "min_confidence":0.7}

finishlist = []

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# if a video path was not supplied, grab a reference to the webcam
if not args.get("input", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
	print("[INFO] opening video file...")
	vs = cv2.VideoCapture(args["input"])
	fps = vs.get(cv2.CAP_PROP_FPS)

# initialize the video writer (we'll instantiate later if need be)
writer = None

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None
maxDisappeared = 20

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=maxDisappeared, maxDistance=80)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
biglist = {}

#totalDown = 0
#totalUp = 0

#fps = FPS().start()

refPt = []
n = 0

def certifytag(rawstring):
	#set own requirements
	if len(rawstring) == 4:
		return True
	else:
		return False

def click_and_crop(event, x, y, flags, param):
	global refPt, n
	
	if n < 2:
		if event == cv2.EVENT_LBUTTONDOWN:
			
			mytuple = (x,y)
			refPt.append(mytuple)
			cv2.circle(frame, mytuple, 3, (0,0,255),-1)
			
			cv2.imshow("start_frame", frame)
			n += 1

	if n == 2:
		#colours in bgr
		cv2.line(frame, refPt[0], refPt[1], (0,0,255), 2)
		cv2.imshow("start_frame", frame)
		n += 1

def check_hit_line(boxypts,smalllist, mybuffer = 0): #0 for start, 1 for end 
	x1 = boxypts[0]
	y1 = boxypts[1]
	x2 = boxypts[2]
	y2 = boxypts[3] #bottom right

	ave_x = (x1 + x2)/2
	line_xdistance = abs(smalllist[1][0] - smalllist[0][0])
	distance_from_firstpt = abs(x1 - smalllist[0][0])
	y_target = (smalllist[1][1] - smalllist[0][1])/line_xdistance*distance_from_firstpt + smalllist[0][1]
	
	#print(y_target, y2)
	
	if ((ave_x >= smalllist[0][0] and ave_x <= smalllist[1][0]) or (ave_x <= smalllist[0][0] and ave_x >= smalllist[1][0])) and y2 >= y_target - mybuffer:
		# hit
		return True
	else:
		return False

def contain_within_frame(value, W, H, mode): #0 for xaxis, 1 for yaxis
	if mode == 0:
		value = max(0,value)
		value = min(value, W)
	else:
		value = max(0,value)
		value = min(value, H)

	return value


def extract_text(boxcoords, myframe, feedback = True):
	(H, W) = myframe.shape[:2]

	
	startX = contain_within_frame(boxcoords[0],W,H,0)
	startY = contain_within_frame(boxcoords[1],W,H,1)
	endX = contain_within_frame(boxcoords[2],W,H,0)
	endY = contain_within_frame(boxcoords[3],W,H,1)

	TAGstarty = (endY - startY)//4 * 1 + startY
	TAGendy = (endY - startY)//4 * 3 + startY
	persontagframe = myframe[ TAGstarty:TAGendy , startX:endX ]
	p2 = persontagframe.copy()
	#personframe = myframe[ startY:endY , startX:endX ]
	feedx = 32 * round(persontagframe.shape[1]/32)
	feedy = 32 * round(persontagframe.shape[0]/32)

	if feedback == True:
		cv2.imshow("Image of Interest", persontagframe)

	###  Pre-processing
	
	kernel = np.ones((1, 1), np.uint8)
	p2 = cv2.dilate(p2, kernel, iterations=1)
	p2 = cv2.erode(p2, kernel, iterations=1)
	# Apply blur to smooth out the edges
	p2 = cv2.medianBlur(p2,5)

	###
	if feedback == True:
		print("Scanning with un-processed frame(1)")
	datadict = readtext(persontagframe, feedx, feedy, feedback,text_recog_args,True)
	if feedback == True:
		print("Scanning with processed frame(2)")
	datadict2 = readtext(p2, feedx, feedy, feedback,text_recog_args,True)
	
	resultslist = set()

	for item in datadict:
		smallstring = re.sub("[^0-9]", "", item)
		if certifytag(smallstring) == True:
			# pass the test
			resultslist.add(smallstring)

	print("Used original, resultslist", resultslist)
	
	for item in datadict2:
		smallstring = re.sub("[^0-9]", "", item)
		if certifytag(smallstring) == True:
			# pass the test
			resultslist.add(smallstring)

	print("Used pre-processed, resultslist", resultslist)

	resultslist = list(sorted(resultslist, reverse=True))

	if len(resultslist) != 0:
		#get highest
		person_num_tag = resultslist[0]
		print("Tag number:",person_num_tag)

		if feedback == False:
			cv2.imshow("Found it!", persontagframe)
			cv2.waitKey(0)
			cv2.destroyWindow("Found it!")

		#cv2.imwrite(person_num_tag + ".jpg", personframe)
	else:
		if feedback == True:
			print("Error number tag could not be found")
		#person_num_tag = input("Whats the num tag")
		person_num_tag = None
		
	if feedback == True:
		cv2.waitKey(0)

		cv2.destroyWindow("Image of Interest")
	#cv2.imwrite("output/" + person_num_tag + ".jpg", personframe)
	return person_num_tag

def recurse_read(objectID, rectlist, framenum_list,mycounter):
	global biglist,W,H
	#print(rectlist)
	print("Searching past frames")
	#print(framenum_list)
	#print(vs.get(cv2.CAP_PROP_FRAME_COUNT))
	#print(len(biglist))
	status = None
	i = 1 + maxDisappeared
	j = mycounter
	#whatleft = len(rectlist) - mycounter
	whatleft = len(rectlist)

	"""
	while status == None and j >= 0:
		#print(i, framenum_list[-i], rectlist[-i])

		#vs.set(cv2.CAP_PROP_POS_FRAMES, framenum_list[-i])
		#flag, myframelol = vs.read()
		myframelol = biglist[int(framenum_list[j])]
		print("processing", int(framenum_list[j]))
		(cH, cW) = myframelol.shape[:2]
		sx = rectlist[j][0]
		sy = rectlist[j][1]
		ex = rectlist[j][2]
		ey = rectlist[j][3]

		sx = int(round(sx /W*cW))
		ex = int(round(ex /W*cW))
		sy = int(round(sy /H*cH)) 
		ey = int(round(ey /H*cH)) 

		status = extract_text( (sx,sy,ex,ey) , myframelol, False)

		j = j-1
	"""
	while status == None and i < whatleft:
		print("processing", int(framenum_list[-i]))
		#print(i, framenum_list[-i], rectlist[-i])

		#vs.set(cv2.CAP_PROP_POS_FRAMES, framenum_list[-i])
		#flag, myframelol = vs.read()
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

		#lolframe = imutils.resize(myframelol, width=500)
		
		status = extract_text( (sx,sy,ex,ey) , myframelol, False)
		#cv2.imshow("Backwards",lolframe)
		#cv2.waitKey(0)
		i = i+1

	if status == None:
		print("It looks like the number tag cannot be recognised. Please key in manually")
		status = input("Number tag: ")
	
	#print(status)
	return status


# loop over frames from the video stream
while True:
	# grab the next frame and handle if we are reading from either
	# VideoCapture or VideoStream
	counter = vs.get(cv2.CAP_PROP_POS_FRAMES)
	flag, frame = vs.read()

	# if we are viewing a video and we did not grab a frame then we
	# have reached the end of the video
	if not flag:
		#print("Total frames", totalFrames)
		#vs.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
		#print("cv2's max frames",vs.get(1))
		break

	superclean = frame.copy()
	superclean = imutils.resize(superclean, width=1000)
	# resize the frame to have a maximum width of 500 pixels (the
	# less data we have, the faster we can process it), then convert
	# the frame from BGR to RGB for dlib
	frame = imutils.resize(frame, width=500)


	if totalFrames == 0:
		cv2.putText(frame, "Click 2 points to indicate the finish line.",(14,14), cv2.FONT_HERSHEY_DUPLEX, 0.5,(0,0,255),1,cv2.LINE_AA)
		cv2.putText(frame, "Press r key to restart, c key to confirm",(14,30), cv2.FONT_HERSHEY_DUPLEX, 0.5,(0,0,255),1,cv2.LINE_AA)
		clone = frame.copy()
		cv2.namedWindow("start_frame")
		cv2.setMouseCallback("start_frame", click_and_crop)

		while True:
			cv2.imshow("start_frame", frame)
			key = cv2.waitKey(0) & 0xFF

			if key == ord("r"):
				frame = clone.copy()
				refPt = []
				n = 0

			elif key == ord("c") and n==3:
				#print("Coordinates are",refPt[0],refPt[1])
				break

		cv2.destroyWindow('start_frame')
		print(refPt)
		# start the frames per second throughput estimator
		fps = FPS().start()


	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	cleanframe = frame.copy()

	# if the frame dimensions are empty, set them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# if we are supposed to be writing a video to disk, initialize
	# the writer
	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(W, H), True)

	# initialize the current status along with our list of bounding
	# box rectangles returned by either (1) our object detector or
	# (2) the correlation trackers
	status = "Waiting"
	rects = []

	# check to see if we should run a more computationally expensive
	# object detection method to aid our tracker
	if totalFrames % args["skip_frames"] == 0:
		# set the status and initialize our new set of object trackers
		status = "Detecting"
		trackers = []

		# convert the frame to a blob and pass the blob through the
		# network and obtain the detections
		blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
		net.setInput(blob)
		detections = net.forward()

		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated
			# with the prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by requiring a minimum
			# confidence
			if confidence > args["confidence"]:
				# extract the index of the class label from the
				# detections list
				idx = int(detections[0, 0, i, 1])

				# if the class label is not a person, ignore it
				if CLASSES[idx] != "person":
					continue

				# compute the (x, y)-coordinates of the bounding box
				# for the object
				box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
				(startX, startY, endX, endY) = box.astype("int")

				# construct a dlib rectangle object from the bounding
				# box coordinates and then start the dlib correlation
				# tracker
				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(startX, startY, endX, endY)
				tracker.start_track(rgb, rect)

				# add the tracker to our list of trackers so we can
				# utilize it during skip frames
				trackers.append(tracker)
	else:
		# otherwise, we should utilize our object *trackers* rather than
		# object *detectors* to obtain a higher frame processing throughput
		
		# loop over the trackers
		for tracker in trackers:
			# set the status of our system to be 'tracking' rather
			# than 'waiting' or 'detecting'
			status = "Tracking"

			# update the tracker and grab the updated position
			tracker.update(rgb)
			pos = tracker.get_position()

			# unpack the position object
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			# add the bounding box coordinates to the rectangles list
			rects.append((startX, startY, endX, endY))

	# draw a horizontal line in the center of the frame -- once an
	# object crosses this line we will determine whether they were
	# moving 'up' or 'down'
	cv2.line(frame, refPt[0], refPt[1], (0,0,255), 2)

	# use the centroid tracker to associate the (1) old object
	# centroids with (2) the newly computed object centroids
	objects,myrects = ct.update(rects)

	if len(myrects) != 0:
		biglist[int(counter)] = superclean

	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		# check to see if a trackable object exists for the current
		# object ID
		to = trackableObjects.get(objectID, None)

		# if there is no existing trackable object, create one
		if to is None:
			to = TrackableObject(objectID, centroid,myrects[objectID],counter)

		# otherwise, there is a trackable object so we can utilize it
		# to determine direction
		else:
			# the difference between the y-coordinate of the *current*
			# centroid and the mean of *previous* centroids will tell
			# us in which direction the object is moving (negative for
			# 'up' and positive for 'down')
			#y = [c[1] for c in to.centroids]
			#direction = centroid[1] - np.mean(y)

			# check to see if the object has been counted or not
			if not to.counted:
				if check_hit_line(myrects[objectID],refPt):
					to.counted = True				
					finishlist.append((objectID,vs.get(cv2.CAP_PROP_POS_MSEC)/1000,counter,myrects[objectID],len(to.framenums)))

			to.centroids.append(centroid)
			to.rects.append(myrects[objectID])
			to.framenums.append(counter)


		# store the trackable object in our dictionary
		trackableObjects[objectID] = to

		# draw both the ID of the object and the centroid of the
		# object on the output frame
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0], centroid[1]),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		#cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
		c_rect = myrects[objectID]
		cv2.rectangle(frame, (c_rect[0],c_rect[1]), (c_rect[2],c_rect[3]), (255, 0, 0), 2)

	# construct a tuple of information we will be displaying on the
	# frame
	info = [
		("Status", status),
	]

	# loop over the info tuples and draw them on our frame
	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	# check to see if we should write the frame to disk
	if writer is not None:
		writer.write(frame)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# increment the total number of frames processed thus far and
	# then update the FPS counter
	#print(vs.get(cv2.CAP_PROP_POS_FRAMES))
	#print(totalFrames)

	totalFrames += 1
	fps.update()

# stop the timer and display FPS information
fps.stop()

print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

for item in finishlist:
	smallid = item[0]
	to = trackableObjects[smallid]
	
	counter = item[2]
	print("processing", counter)
	(cH, cW) = biglist[counter].shape[:2]

	sx = item[3][0]
	sy = item[3][1]
	ex = item[3][2]
	ey = item[3][3]

	sx = int(round(sx /W*cW))
	ex = int(round(ex /W*cW))
	sy = int(round(sy /H*cH)) 
	ey = int(round(ey /H*cH)) 

	tagnum = extract_text( (sx,sy,ex,ey) , biglist[counter], True)

	#tagnum = extract_text(myrects[objectID],superclean)

	if tagnum == None:
		tagnum = recurse_read(objectID, to.rects,to.framenums,item[4]-1)
		
	print(tagnum,"hit line at", item[1])


# check to see if we need to release the video writer pointer
if writer is not None:
	writer.release()

# if we are not using a video file, stop the camera video stream
if not args.get("input", False):
	vs.stop()

# otherwise, release the video file pointer
else:
	vs.release()

# close any open windows
cv2.destroyAllWindows()