import cv2
import numpy as np

#open image
BGRImage = cv2.imread("images project1/peppers.jpg")
#convert to RGB
RGBImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2RGB)

HSVImage = cv2.cvtColor(RGBImage, cv2.COLOR_RGB2HSV)

cv2.imshow('RGB image', RGBImage)
cv2.imshow('HSV image', HSVImage)
cv2.imshow('BGR image', BGRImage)



#transforming from HSV to HSI
#all values between 0 and 1
bgr = np.float32(BGRImage) / 255
b = bgr[:,:,0]
g = bgr[:,:,1]
r = bgr[:,:,2]



def calcIntensity(b,g,r):
    return np.divide(b+r+g, 3)

def calcSaturation(b, g, r ):
    min = np.minimum(np.minimum(r, g), b)
    sat = 1 - (3/ (r+g+b+ 0.00001) * min)
    return sat

def calcHue(b, g, r):
     num = np.multiply(0.5, ((r-g) + (r -b)))
     den = np.sqrt(np.power((r-g), 2) + np.multiply((r-b),(g-b)))
     theta = np.arccos(np.divide(num, den+0.00001))

     if g.all() >= b.all():
         hue = theta
         return hue
     else:
         hue = 360 - theta
         return hue

hsi = cv2.merge(((calcHue(b, g, r)), calcSaturation(b,g,r), calcIntensity(b,g,r)))
cv2.imshow('HSI Image', hsi)
cv2.waitKey(0)
cv2.destroyAllWindows()



