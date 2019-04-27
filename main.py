import numpy as np
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
from PIL import Image

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

img1 = cv2.imread('example1.jpg')

cv2.imshow('grayScale', img1)
cv2.waitKey(0)

img = cv2.imread('example1.jpg', 0)

cv2.imshow('grayScale', img)
cv2.waitKey(0)

thresh = 127
img_bin = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]

cv2.imshow('binary', img_bin)
cv2.waitKey(0)

'''
gradient = cv2.morphologyEx(img_bw, cv2.MORPH_GRADIENT, kernel)

cv2.imshow('morphological', gradient)
cv2.waitKey(0)
'''

# Defining a kernel length
kernel_length = np.array(img).shape[1] // 80

# A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
# A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
# A kernel of (3 X 3) ones.
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
'''
# Morphological operation to detect vertical lines from an image
img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
cv2.imwrite("verticle_lines.jpg",verticle_lines_img)
# Morphological operation to detect horizontal lines from an image
img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
cv2.imwrite("horizontal_lines.jpg",horizontal_lines_img)

# Weighting parameters, this will decide the quantity of an image to be added to make a new image.
alpha = 0.5
beta = 1.0 - alpha
# This function helps to add two image with specific weight parameter to get a third image as summation of two image.
img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
(thresh, img_final_bin) = cv2.threshold(img_final_bin, 128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imwrite("img_final_bin.jpg",img_final_bin)
'''
# Find contours for image, which will detect all the boxes
contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Sort all the contours by top to bottom.
(contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")

idx = 0
for c in contours:
    # Returns the location and width,height for every contour
    x, y, w, h = cv2.boundingRect(c)
    if (w > 80 and h > 20) and w > 3*h:
        idx += 1
        new_img = img_bin[y:y+h, x:x+w]
        cv2.imwrite( str(idx)+'.png', new_img)
# If the box height is greater then 20, widht is >80, then only save it as a box in "cropped/" folder.
    if (w > 80 and h > 20) and w > 3*h:
        idx += 1
        new_img = img_bin[y:y+h, x:x+w]
        cv2.imwrite(str(idx) + '.png', new_img)


output = cv2.imread('1.png')
cv2.imshow('Output', output)
cv2.waitKey(0)

#------------------ Resizing ------------------
scale_percent = 150 # percent of original size
width = int(output.shape[1] * scale_percent / 100)
height = int(output.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(output, dim, interpolation = cv2.INTER_AREA)
#------------------------------------------------

cv2.imshow('resizedOutput', resized)
cv2.waitKey(0)

kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

opening = cv2.morphologyEx(resized, cv2.MORPH_OPEN, kernel2)
cv2.imshow('Opening', opening)
cv2.waitKey(0)

closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)
cv2.imshow('Closing', closing)
cv2.waitKey(0)

closingNOT = cv2.bitwise_not(closing)
cv2.imwrite('closingNOT.png', closingNOT)

'''
output2 = cv2.imread('1.png', 0)
contours2, hierarchy2 = cv2.findContours(output2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Sort all the contours by top to bottom.
(contours2, boundingBoxes) = sort_contours(contours2, method="top-to-bottom")

for c in contours2:
    cv2.drawContours(output2, c, -1, (0,255,0), 3)

cv2.imshow('drawCountours', output2)
cv2.waitKey(0)
'''


im = Image.open('closingNOT.png')
#im.show()
width, height = im.size   # Get dimensions

new_height = height/2

left = (width - width)/2
top = (height - new_height)/2 -14
right = ((width + width)/2)
bottom = (height + new_height)/2 +14

cropped = im.crop((left, top, right, bottom))
cropped.show()

#cv2.imshow('test', cropped)
#cv2.waitKey(0)

