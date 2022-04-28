import cv2
import numpy as np


#getting V and I
def getI(image):
    #normalize values
    im = np.float32(image) / 255
    #entire image per colour
    R = im[:,:,0]
    G = im[:,:,1]
    B = im[:,:,2]
    #intensity formula
    I = np.divide((R+G+B), 3)
    return I


def getV(image):
    im = np.float32(image) / 255
    R = im[:,:,0]
    G = im[:,:,1]
    B = im[:,:,2]
    V = np.maximum((np.maximum(R, G)), B)
    return V


stoneIm = cv2.imread("images project1/stone.jpg")
birdsIm = cv2.imread("images project1/peppers.jpg")
RGBstone = cv2.cvtColor(stoneIm, cv2.COLOR_BGR2RGB)
RGBbirds = cv2.cvtColor(birdsIm, cv2.COLOR_BGR2RGB)

#transforming into HSV
HSVstone = cv2.cvtColor(RGBstone, cv2.COLOR_RGB2HSV)
HSVbirds = cv2.cvtColor(RGBbirds, cv2.COLOR_RGB2HSV)

Ibirds = getI(RGBbirds)
Istone = getI(RGBstone)
Vbirds = getV(RGBbirds)
Vstone = getV(RGBstone)

cv2.imshow('Original Birds', birdsIm)
cv2.imshow('Original stone', stoneIm)
cv2.imshow('HSV birds', HSVbirds)
cv2.imshow('HSV stone', HSVstone)
cv2.imshow('I Birds image', Ibirds)
cv2.imshow('I Stone image', Istone)
cv2.imshow('V Birds image', Vbirds)
cv2.imshow('V Stone image', Vstone)
cv2.imshow('R', birdsIm[:,:,2])


cv2.waitKey(0)
cv2.destroyAllWindows()
