import glob

import cv2
import cv2 as c
import matplotlib.pyplot as plt
import numpy as np


def changeIntensity(image, power):
    image = np.abs(image)**power
    return image


def changeBrightness(image, add):
    res = image + add
    return res


def getHSV(image):
    res = c.cvtColor(image, c.COLOR_BGR2RGB)
    res = c.cvtColor(image, c.COLOR_RGB2HSV)
    return res

def getMatrixData(list):
    M = []
    for lis in list:
        M[,:]= lis.flatten()


face1 = c.imread('images project 2/woman.jpg')
face2 = c.imread('images project 2/manReshaped.jpg')
face3 = c.imread('images project 2/kid2 - Copy.jpg')
size = face1.shape[0], face1.shape[1]
# sizes arent the same 1090x1050 for kid2, and 1050x1090 for man and woman
# print(face3.shape)
# print(face1.shape)

# saving all variations/face into a list of images
dis_F1 = []
for img in glob.glob("images project 2//womanVariation/*.JPG"):
    dis_F1.append(cv2.imread(img))

dis_F2 = []
for img in glob.glob("images project 2//boyVariations/*.JPG"):
    dis_F2.append(cv2.imread(img))

dis_F3 = []
for img in glob.glob("images project 2//manVars/*.JPG"):
    dis_F3.append(cv2.imread(img))

M = getMatrixData(dis_F1)
print(M)

# create matrix with variations
# f1_i08 = changeIntensity(face1, 0.8)
# c.imshow('mane', f1_i08)

# f1_i2 = changeIntensity(face1, 2)
# f2_i08 = changeIntensity(face2, 0.8)
# f2_i2 = changeIntensity(face2, 2)
# f3_i08 = changeIntensity(face3, 0.8)
# f3_i2 = changeIntensity(face3, 2)
#
# f1_b60 = changeBrightness(face1, 60)





# c.imshow('cropeed',face3)
# c.imshow('F1 INTESITY 60',f1_b60)
# c.imshow('cropeed',f1_i08)
# c.imshow('cropeed',f2_i2)
# c.imshow('cropeed',f2_i08)
# c.imshow('cropeed',f3_i08)
# c.imshow('cropeed',f3_i2)
c.waitKey(0)
c.destroyAllWindows()


