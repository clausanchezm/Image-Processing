import cv2
import numpy as np
from matplotlib import pyplot as plt


#def toPolar(image):
#v   height, width, z = image.shape
#  toReturn = np.zeros_like(image)
#   radius = int(np.sqrt(height*height+width*width))
#   theta = int(np.arctan2(width, height))
#   for i in range(0, height-1):
#       for j in range(0, width-1):
#           x = radius * np.cos(theta)
#           y = radius * np.sin(theta)
#           toReturn[int(y), int(x)] = image[i, j]
#
#   return toReturn

def getDominantColor(im):
    colors, count = np.unique(im.reshape(-1,im.shape[-1]), axis=0, return_counts=True)
    color = colors[count.argmax()]
    x, y, z = im.shape
    C = np.zeros((x, y, z), dtype= "uint8")
    for i in range(0, x):
        for j in range(0, y):
            C[i, j] = color
    return C


def cartoonify(im):
    #uniform to be applied
    color = getDominantColor(im)
    imG = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    imBlur = cv2.medianBlur(imG, 9)
    # getting the edges of image
    #imEdges = cv2.adaptiveThreshold(imBlur, 255,  cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    edges = cv2.Canny(im, 90, 150)

    cartoon = cv2.bitwise_or(im, color, mask=edges)
    return cartoon


image = cv2.imread("images project1/pink.jpg")
# #image = toPolar(polarIm)
# image64 = polarIm.astype(np.float64)
#
# r = np.sqrt(((image64.shape[0]/2.0)**2) + ((image64.shape[1]/2.0)**2))
# polarizedIm = cv2.linearPolar(image64, (image64.shape[0]/2.0, image64.shape[1]/2.0), r, cv2.WARP_FILL_OUTLIERS)

# cv2.imshow('polar im',polarizedIm)
# cv2.imshow('og', polarIm)


cartooned = cartoonify(image)
cv2.imshow('blaketer', cartooned)

cv2.waitKey(0)
cv2.destroyAllWindows()


# https://pythonwife.com/bitwise-operators-and-masking-in-opencv/#:~:text=When%20we%20apply%20cv2.bitwise_or%20%28%29%20it%20displays%20as,union%20operation%20between%20the%20rectangle%20and%20circle%20images.
# bitwiseAND checks if the pixels are the same intensity, if so then resulting image has that intensity
#     bitwiseOR is takes both intensities if they same or not, so we can say its  a mix of both.
