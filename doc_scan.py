# import the necessary packages
from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import cv2
import imutils

# Load the desired image and compute the ratio of the origibnal height with the
# new one and resize the image, accordingly. 
img = cv2.imread("images/test1.jpg")
rat = img.shape[0] / 500.0
orig = img.copy()
img = imutils.resize(img, height = 500)

# convert the image to grayscale, use the Gaussian blur to smooth the image,
# and find edges
gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gimg = cv2.GaussianBlur(gimg, (5, 5), 0)
eimg = cv2.Canny(gimg, 40, 100)

# show the original image and the edge detected image
print("Converting the Original Image to Edge Map")
cv2.imshow("Original Image", img)
cv2.imshow("Thresholded Image", eimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Compute the largest contours in the thresholded image and intialize them 
cont = cv2.findContours(eimg.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cont = imutils.grab_contours(cont)
cont = sorted(cont, key = cv2.contourArea, reverse = True)[:5]

# loop over the contours
for i in cont:
	# approximate the contour
	pmt = cv2.arcLength(i, True)
	aprm = cv2.approxPolyDP(i, 0.02 * pmt, True)

	# Find out the four approximated contour points in order to get the screen
	if len(aprm) == 4:
		count = aprm
		break
# show the outline of the document
print("Showing the Outline of the Document on the original image")
cv2.drawContours(img, [count], -1, (0, 255, 0), 2)
cv2.imshow("border", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply the four-point transform to the original image
wrp = four_point_transform(orig, count.reshape(4, 2) * rat)

# Apply the black and white paper effect
wrp = cv2.cvtColor(wrp, cv2.COLOR_BGR2GRAY)
Thresh = threshold_local(wrp, 11, offset = 10, method = "gaussian")
wrp = (wrp > Thresh).astype("uint8") * 255

# show the original and scanned images
print("Showing the original and scanned images")
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(wrp, height = 650))
cv2.waitKey(0)