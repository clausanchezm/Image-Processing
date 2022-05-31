import glob

import cv2
import cv2 as c
import matplotlib.pyplot as plt
import numpy as np


def changeBrightness(image, add):
    res = image.astype(np.uint8) + add
    return res


def negativePI(im):
    res = np.abs(255 - im)
    return res


def getHSV(image):
    res = c.cvtColor(image, c.COLOR_BGR2RGB)
    res = c.cvtColor(image, c.COLOR_RGB2HSV)
    return res


def getMatrixData(list):
    matrix = np.matrix([var.flatten() for var in list])
    return matrix


def computePCA(data):
    mean, vectors = c.PCACompute(data, mean=None)
    # getting array of length of the # of eigenvectors, 1:1 correspondance
    weights = getWeights(data, vectors, mean)
    eigenFace = vectors * weights
    # we want the weighted sum of all eigenvectors
    eigenFace = eigenFace.sum()
    reFace = mean + eigenFace
    return reFace


# weights are computed by computing the difference between the
# variation and the mean face and then multiplied by their corresponding eigenV
def getWeights(d, v, m):
    w = []
    for i in range(d.shape[0]):
        diff = np.squeeze(np.asarray(d[i], axis=0)) - m
        w[i] = np.dot(v[i], diff)
    return w


face1 = c.imread('images project 2/woman.jpg')
face2 = c.imread('images project 2/manReshaped.jpg')
face3 = c.imread('images project 2/kid2 - Copy.jpg')
# sizes arent the same 1090x1050 for kid2, and 1050x1090 for man and woman
# print(face3.shape)
# print(face1.shape)

# saving all variations/face into a list of images
# dis_F1 = []
# for img in glob.glob("images project 2//womanVariation/*.JPG"):
#     dis_F1.append(cv2.imread(img))
#
# dis_F2 = []
# for img in glob.glob("images project 2//boyVariations/*.JPG"):
#     dis_F2.append(cv2.imread(img))
#
# dis_F3 = []
# for img in glob.glob("images project 2//manVars/*.JPG"):
#     dis_F3.append(cv2.imread(img))
#
# # get the variations matrices
# dF1 = getMatrixData(dis_F1)
# dF2 = getMatrixData(dis_F2)
# dF3 = getMatrixData(dis_F3)

# get the eigenvectors, values


# wB90 = changeBrightness(face1, 89)
# mB90 = changeBrightness(face2, 89)
# bB90 = changeBrightness(face3, 89)

# wN = negativePI(face1)
# mN = negativePI(face2)
# bN = negativePI(face3)

c.imshow('woman B+90', face1)
c.imshow('man B+90', face2)
c.imshow('babyboy B+90', face3)
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



c.waitKey(0)
c.destroyAllWindows()
