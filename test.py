
# import cv2 library 
import cv2
import imutils
import os
import numpy as np
import pytesseract
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, help="path image file")
args = vars(ap.parse_args())

"""
# videoCapture method of cv2 return video object 

# Pass absolute address of video file 
cap = cv2.VideoCapture("videos/derrick_run2.mov") 
counter = 0
t1 = cap.get(7)
print("Supposed maxframes",t1)
#print(cap.get(7))
realdiff = None


target = 50

def lolol():
	global realdiff
	cap.set(cv2.CAP_PROP_POS_FRAMES, t1)
	realdiff = t1 - cap.get(1) + 1
	print("Difference", realdiff)
			
lolol()
finalval = t1-realdiff 
#pointtostop = (t1-realdiff)//5

def seeking(counter2,framediff = 0):
	global stallframe
	print(framediff)
	cap.set(cv2.CAP_PROP_POS_FRAMES, counter2)
	check , vid2 = cap.read()
	vid2 = imutils.resize(vid2, width=500)
	#cv2.putText(vid2, str(counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
	cv2.imshow("Display2",vid2)
	cv2.waitKey(3)
	s = checksame(stallframe,vid2)

	if s == False:
		if framediff > -10 and framediff < 1:
			seeking(counter2-1, framediff - 1)
		elif framediff == -10:
			#framediff at -10 status
			seeking(counter2+11, framediff + 11)
		elif framediff < 10:
			seeking(counter2+1, framediff + 1)
		else:
			print("Failed to find frame")						

	else:
		print("Frame matched")
		cv2.waitKey(0)

def checksame(orig, duplicate):
	if orig.shape == duplicate.shape:
		#print("The images have same size and channels")
		difference = cv2.subtract(orig, duplicate)
		b, g, r = cv2.split(difference)
		if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
			#print("The images are completely Equal")
			return True
		else:
			#print("Not quite same")
			#print(cv2.countNonZero(b),cv2.countNonZero(g),cv2.countNonZero(r))
			return False

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
stallframe = None

while True: 
	counter = cap.get(cv2.CAP_PROP_POS_FRAMES)
	#print(counter)
	check , vid = cap.read()
	if vid is None:
		break
	#counter += 1
	vid = imutils.resize(vid, width=500)
	#cv2.putText(vid, str(counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
	#cv2.imshow("Display",vid)
	#k = cv2.waitKey(1) & 0xFF
	if counter == target:
		stallframe = vid
		
		#tobesub = realdiff//4+1
		#print("Error correction reduced framecount by", tobesub)
		#counter = counter - tobesub
		#seeking(counter)

actualframecount = round(target/counter * finalval)
print(realdiff, counter - finalval, actualframecount, target, counter, finalval)
cv2.imshow("Display",stallframe)
seeking(actualframecount)

cap.release()
cv2.destroyAllWindows()

#cap = cv2.VideoCapture("videos/derrick_run2.mov")
#cap.set(cv2.CAP_PROP_POS_FRAMES, 127)
#check , vid2 = cap.read()
#vid2 = imutils.resize(vid2, width=500)
#cv2.imshow("Display2",vid2)
#cv2.waitKey(0) & 0xFF
#cap.release() 
#cv2.destroyAllWindows() 
"""

def get_string(img_path):
	# Read image using opencv
	img = cv2.imread(img_path)

	# Extract the file name without the file extension
	file_name = os.path.basename(img_path).split('.')[0].split()[0]

	# Create a directory for outputs
	#output_path = os.path.join(output_dir, file_name)
	#if not os.path.exists(output_path):
	  #  os.makedirs(output_path)

	# Rescale the image, if needed.
	img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

	# Convert to gray
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Apply dilation and erosion to remove some noise
	kernel = np.ones((1, 1), np.uint8)
	img = cv2.dilate(img, kernel, iterations=1)
	img = cv2.erode(img, kernel, iterations=1)
	#cv2.imshow("result",img)
	#cv2.waitKey(0)

	# Apply blur to smooth out the edges

	img2 = cv2.medianBlur(img,5)
	cv2.imshow("Mblur",img2)
	cv2.waitKey(0)

	img = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 5)
	cv2.imshow("result2",img)
	cv2.waitKey(0)

	# Recognize text with tesseract for python
	#config = ("--oem 0 -c tessedit_char_whitelist=0123456789")
	result = pytesseract.image_to_string(img, lang='eng', \
        config='--psm 10 --oem 1 -c tessedit_char_whitelist=0123456789')

	#result = pytesseract.image_to_string(img, config=config)
	print(result)
	


	## Save the filtered image in the output directory
	#save_path = os.path.join(output_path, file_name + "_filter_" + str(method) + ".jpg")
	#cv2.imwrite(save_path, img)

	# Recognize text with tesseract for python
	#esult = pytesseract.image_to_string(img, lang="eng")
	#print(result)
	#return result
	"""
	coords = np.column_stack(np.where(img > 0))
	angle = cv2.minAreaRect(coords)[-1]

	if angle < -45:
		angle = -(90 + angle)
	 
	# otherwise, just take the inverse of the angle to make
	# it positive
	else:
		angle = -angle

	(h, w) = img.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

	# draw the correction angle on the image so we can validate it
	cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
		(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	 
	# show the output image
	print("[INFO] angle: {:.3f}".format(angle))
	cv2.imshow("Input", img)
	cv2.imshow("Rotated", rotated)
	cv2.waitKey(0)
	"""

	cv2.destroyAllWindows()

get_string(args['input'])
