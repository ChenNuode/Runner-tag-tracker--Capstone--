# Usage
# python3 people_counter.py -i videos/bigrun_2.mov -o output/output_01.avi

# import the necessary Pypi packages 
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import sys

# import my own files
from helper_funcs import *

#import pyimagesearch'es files for tracking objects (runners)
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
	help="path to input video file")
ap.add_argument("-c", "--confidence", type=float, default=0.6,
	help="minimum probability to filter weak detections")
ap.add_argument("-k", "--skip-frames", type=int, default=5,
	help="# of skip frames between detections")
ap.add_argument("-s", "--save-images", type=bool, default=False,
	help="Option to save image files")
ap.add_argument("-f", "--feedback", type=bool, default=False,
	help="Option to toggle all feedback, if option is turned off, only minimal feedback is given")

args = vars(ap.parse_args())
starttime = 0

# list to track finishers
finishlist = []

# initialize the list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("mobilenet_ssd/MobileNetSSD_deploy.prototxt", "mobilenet_ssd/MobileNetSSD_deploy.caffemodel")


# Set the video
print("[INFO] opening video file...")
vs = cv2.VideoCapture(args["input"])


# Set constant from when to remove the runners
maxDisappeared = 20

# instantiate our centroid tracker to track the runners, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject

ct = CentroidTracker(maxDisappeared=maxDisappeared, maxDistance=80)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far
totalFrames = 0

# initialize the frame dimensions (update it later when video frame is read)
W = None
H = None

#initialise the fps tracker to determine our processing speed
fps = None

#initialise the dictionary to store frames in video
biglist = {}

# create a list to store coordinates of finish line
refPt = []
# counter for finish line coords (see use later)
n = 0
#parameter whether to save the runner tag imgs (for future Machine Learning)
savingimg = args["save_images"]


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


# loop over frames from the video stream
while True:
	# grab the next frame
	
	counter = vs.get(cv2.CAP_PROP_POS_FRAMES)
	flag, frame = vs.read()

	# see if video has ended
	if not flag:
		break

	#get a copy of the frame to store in memory later,  resize it to a higher resolution
	superclean = frame.copy()
	superclean = imutils.resize(superclean, width=1000)


	# resize the frame we analysing to have a width of 500 pixels for faster processing
	frame = imutils.resize(frame, width=500)

	#use the start frame to set finish line coords
	if totalFrames == 0:
		cv2.putText(frame, "Click 2 points to indicate the finish line.",(14,14), cv2.FONT_HERSHEY_DUPLEX, 0.5,(0,0,255),1,cv2.LINE_AA)
		cv2.putText(frame, "Press r key to restart, c key to confirm",(14,30), cv2.FONT_HERSHEY_DUPLEX, 0.5,(0,0,255),1,cv2.LINE_AA)
		clone = frame.copy()
		cv2.namedWindow("start_frame")
		cv2.setMouseCallback("start_frame", click_and_crop)

		while True:
			cv2.imshow("start_frame", frame) #show frame for user to click on finish line coords
			key = cv2.waitKey(0) & 0xFF

			# reset all selected coords
			if key == ord("r"):
				frame = clone.copy()
				refPt = []
				n = 0

			# user has confirmed the coords
			elif key == ord("c") and n==3:
				break

		cv2.destroyWindow('start_frame')
		
		# start the frames per second throughput estimator
		fps = FPS().start()


	# then convert the frame from BGR to RGB for dlib (library we use for tracking)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	cleanframe = frame.copy()

	# if the frame dimensions are empty, set them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# initialize the current status along with the list of bounding
	# box rectangles returned by either (1) object detector (neural network) or
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
		# otherwise, we should utilize our object trackers instead of runner the neural network again to save processing time
		
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

	# draw the finish line
	cv2.line(frame, refPt[0], refPt[1], (0,0,255), 2)

	# use the centroid tracker to associate the (1) old object
	# centroids with (2) the newly computed object centroids
	objects,myrects = ct.update(rects)

	#if there are object's tracked, then load frame to memory, to save memory
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

		# otherwise, there is a trackable object and we check whether it has crossed the line
		else:

			# check to see if the object has been counted or not
			if not to.counted:
				# see if it has crossed finish line
				if check_hit_line(myrects[objectID],refPt):
					to.counted = True
					# append a tuple of (the ID of the finisher, the time he took to finish, the frame number where he finished at, a list of his bounding box coords)
					finishlist.append(  ( objectID , vs.get(cv2.CAP_PROP_POS_MSEC)/1000 - starttime , counter , myrects[objectID] )  )

			#add it to its histories of centroids, its bounding box and the frame number where it appeared
			to.centroids.append(centroid)
			to.rects.append(myrects[objectID])
			to.framenums.append(counter)


		# store the trackable object in our dictionary
		trackableObjects[objectID] = to

		# draw both the ID of the object and the bounding box of the
		# object on the output frame
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0], centroid[1]),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		c_rect = myrects[objectID]
		cv2.rectangle(frame, (c_rect[0],c_rect[1]), (c_rect[2],c_rect[3]), (255, 0, 0), 2)


	# draw the tracker-detector status onto frame
	
	cv2.putText(frame, "Press 's' key to start the timer", (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
	text = "{}: {}".format("Status", status)
	cv2.putText(frame, text, (10, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	elif key == ord("s"): #start the timing, users presses it when runners start running
		if starttime == 0:
			starttime = vs.get(cv2.CAP_PROP_POS_MSEC)/1000
			print("S key pressed, timer has started")

	# increment the total number of frames processed thus far and
	# then update the FPS counter
	totalFrames += 1
	fps.update()

if fps == None:
	print("Your input file does not exist")
	sys.exit()
	
# stop the timer and display FPS information
fps.stop()

print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# loop over finish line to read tags of runners

import csv
with open('tracked_results.csv', mode='w') as csv_file:
	csv_writer = csv.writer(csv_file, delimiter=',')
	csv_writer.writerow(["Runner ID", "Time elapsed"])

	for item in finishlist:
		#get his ID to get his tracker
		smallid = item[0]
		to = trackableObjects[smallid]
		
		#get the frame number where he crossed finish line and its respective frame
		counter = item[2]
		print("processing frame", counter)
		(cH, cW) = biglist[counter].shape[:2]

		sx = item[3][0]
		sy = item[3][1]
		ex = item[3][2]
		ey = item[3][3]

		#scale bounding box of coords (on frame of width 500) to a frame of width 1000
		sx = int(round(sx /W*cW))
		ex = int(round(ex /W*cW))
		sy = int(round(sy /H*cH)) 
		ey = int(round(ey /H*cH)) 
		
		# read the tag number of the person
		tagnum = extract_text( (sx,sy,ex,ey) , biglist[counter], True, savingimg, 0)

		# if tag number cannot be found, recurse through old frames to find the tag number
		if tagnum == None:
			tagnum = recurse_read(objectID, to.rects,to.framenums, biglist, W, H, savingimg, maxDisappeared, args["feedback"])
		
		if tagnum.lower() != "nil":
			print(tagnum,"hit line at", item[1])
			csv_writer.writerow([tagnum, str(item[1])])

vs.release()

# close any open windows
cv2.destroyAllWindows()
