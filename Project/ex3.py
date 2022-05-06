import cv2
import numpy as np


def toPolar(image):
    height, width, z = image.shape
    radius = int(np.sqrt((height/2)**2 + (width/2)**2))
    polarIm = cv2.linearPolar(image, (height/2, width/2), radius, cv2.WARP_FILL_OUTLIERS)
    return polarIm

#https://stackoverflow.com/questions/50899692/most-dominant-color-in-rgb-image-opencv-numpy-python
def getDominantColor(im):
    colors, count = np.unique(im.reshape(-1,im.shape[-1]), axis=0, return_counts=True)
    color = colors[count.argmax()]
    x, y, z = im.shape
    C = np.zeros((x, y, z), dtype = "uint8")
    for i in range(0, x):
        for j in range(0, y):
            C[i, j] = color
    return C


def cartoonify(im):
    # uniform to be applied
   # color = getDominantColor(im)
    imG = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # blur to have a smoother result
    imBlur = cv2.medianBlur(im, 9)
    # getting the edges of image
    # imEdges = cv2.adaptiveThreshold(imBlur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 71, 5)
    imBilateral = cv2.bilateralFilter(imBlur, 4, 100, 100)

    edges = cv2.Canny(imG, 90, 150)
    invEg = np.abs(255 - edges)
    cartoon = cv2.bitwise_and(imBilateral, imBilateral, mask=invEg)
    return cartoon


image = cv2.imread("images project1/pink.jpg")
image = cv2.resize(image, (400, 400))
polar = toPolar(image)

cv2.imshow('polar im', polar)
cv2.imshow('og', image)

cartooned = cartoonify(image)
cv2.imshow('cartoon', cartooned)
# cv2.imwrite('images project1/cartoonF.jpg', cartooned)

cv2.waitKey(0)
cv2.destroyAllWindows()


# https://pythonwife.com/bitwise-operators-and-masking-in-opencv/#:~:text=When%20we%20apply%20cv2.bitwise_or%20%28%29%20it%20displays%20as,union%20operation%20between%20the%20rectangle%20and%20circle%20images.
# bitwiseAND checks if the pixels are the same intensity, if so then resulting image has that intensity
#     bitwiseOR is takes both intensities if they same or not, so we can say its  a mix of both.
