from imutils.perspective import four_point_transform
import imutils
import cv2
from PIL import Image


# define the dictionary of digit segments so we can identify
# each digit on the thermostat
DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 1, 0): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9
}

# load the example image
imageGray = cv2.imread('electricityLCD1.png', 0)
cv2.imwrite('ayHaga.jpg', imageGray)

im = Image.open('ayHaga.jpg')
width, height = im.size  # Get dimensions
new_height = height / 2

left = (width - width) / 2
top = (height - height) / 2
right = ((width + width) / 2)
bottom = (height + height) / 2 - 380

cropped = im.crop((left, top, right, bottom))
#cropped.show()
cropped.save('outLCD.jpg')

image = cv2.imread('outLCD.jpg')
#cv2.imshow("image", image)
#cv2.waitKey()

# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map
image1 = imutils.resize(image, height=500)
gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 200, 255)

#cv2.imshow("image", edged)
#cv2.waitKey()

# find contours in the edge map, then sort them by their
# size in descending order
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
displayCnt = None

# loop over the contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # if the contour has four vertices, then we have found
    # the thermostat display
    if len(approx) == 4:
        displayCnt = approx
        break

# extract the thermostat display, apply a perspective transform
# to it
warped = four_point_transform(gray, displayCnt.reshape(4, 2))
output = four_point_transform(image, displayCnt.reshape(4, 2))

# going to LCD_OCR.py
cv2.imwrite("warpedIMG.jpg", warped)
#cv2.imshow("warped", warped)
#cv2.waitKey()

# threshold the warped image, then apply a series of morphological
# operations to cleanup the thresholded image
thresh = cv2.threshold(warped, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))

#cv2.imshow("thresh", thresh)
#cv2.waitKey()

thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

#cv2.imshow("morph", thresh)
#cv2.waitKey()
