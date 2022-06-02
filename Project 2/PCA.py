import glob
import cv2 as c
import numpy as np


def getMatrixData(variations):
    matrix = np.matrix([var.flatten() for var in variations])
    return matrix


def computePCA(data):
    mean, vectors = c.PCACompute(data, mean= None)

    # print('shape mean ' , mean.shape[0] ,mean.shape[1])
    # print('data sh', data.shape[0])
    # print('shape eigen ' , vectors.shape ,vectors.shape[1])

    # getting array of length of the # of eigenvectors, 1:1 correspondance
    weights = getWeights(data, vectors, mean)
    eigenFace = vectors * weights
    # we want the weighted sum of all eigenvectors
    eigenFace = eigenFace.sum()
    reFace = mean + eigenFace
    r = reFace.astype(np.uint8).reshape((1050, 1090, 3))
    return r


# similar procedure that when we recontruct one face
# face A and face B
def combineFacesPCA(dA, dB):
    meanA, vA = c.PCACompute(dA, mean=None)
    meanB, vB = c.PCACompute(dB, mean=None)
    # get the weights by using the mean of A and the data and eigenvectors of B
    weights = getWeights(dB, vB, meanA)
    eigenF = vB * weights
    new = meanA + eigenF.sum()
    new = new.astype(np.uint8).reshape((1090, 1050, 3))
    return new


def getDifference(face, mean):

    # face = np.squeeze(np.asarray(face), axis= 0)
    # mean = np.squeeze(np.asarray(mean), axis= 0)
    diff = face - mean
    return diff


# weights are computed by computing the difference between the
# variation and the mean face and then multiplied by their corresponding eigenV
def getWeights(d, v, m):
    w = []
    for i in range(d.shape[0]):
        diff = getDifference(d[i], m)
        w.append(np.dot(diff, v[i]))
    return w


face1 = c.imread('images project 2/woman.jpg')

face2 = c.imread('images project 2/manReshaped.jpg')
face3 = c.imread('images project 2/kid2 - Copy.jpg')

# sizes arent the same 1090x1050 for kid2, and 1050x1090 for man and woman
# print(face3.shape)
# print(face1.shape)

# saving all variations/face into a list of images
dis_F1 = []
for img in glob.glob("images project 2//womanVariation/*.JPG"):
    dis_F1.append(c.imread(img))

dis_F2 = []
for img in glob.glob("images project 2//boyVariations/*.JPG"):
    dis_F2.append(c.imread(img))
#
dis_F3 = []
for img in glob.glob("images project 2//manVars/*.JPG"):
    dis_F3.append(c.imread(img))

# get the variations matrices
dF1 = getMatrixData(dis_F1)
# dF2 = getMatrixData(dis_F2)
dF3 = getMatrixData(dis_F3)

# get the eigenvectors, values
womanPCA = computePCA(dF1)
# boyPCA = computePCA(dF2)
# manPCA = computePCA(dF3

womanPCA = c.resize(womanPCA, (400, 400))
womanManPCA = combineFacesPCA(dF1, dF3)
wmPCA = c.resize(womanManPCA, (400, 400))
c.imshow('man and woman', wmPCA)

c.imshow('woman PCA', womanPCA)
# c.imshow('boy PCA', boyPCA)
# c.imshow('man PCA', manPCA)

# create matrix with variations


c.waitKey(0)
c.destroyAllWindows()
