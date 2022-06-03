import glob
import cv2 as c
import numpy as np


def getMatrixData(variations):
    matrix = np.matrix([var.flatten() for var in variations])
    return matrix

def reshapeEigenV(eigenV, size):
    eigen_faces = []
    for i in range (eigenV.shape[0]):
        v = eigenV[i].reshape(size)
        eigen_faces.append(v)
    return eigen_faces


def computePCA(data ):
    mean, vectors = c.PCACompute(data, mean= None)
    # print(vectors)

    # print('shape mean ' , mean.shape[0] ,mean.shape[1])
    # print('data sh', data.shape[0])
    # print('shape eigen ' , vectors.shape ,vectors.shape[1])

    # getting array of length of the # of eigenvectors, 1:1 correspondance
    weights = getWeights(data, vectors, mean)
    print(weights)
    # eigenFace = reshapeEigenV(vectors, size)
    eigenFs = vectors * weights
    # we want the weighted sum of all eigenvectors
    sumEF = eigenFs.sum()
    reFace = mean + sumEF
    return reFace.astype(np.uint8)


# similar procedure that when we recontruct one face
# face A and face B
def combineFacesPCA(dA, dB):
    meanA, vA = c.PCACompute(dA, mean=None)
    meanB, vB = c.PCACompute(dB, mean=None)
    # get the weights by using the mean of A and the data and eigenvectors of B
    weights = getWeights(dB, vB, meanA)
    eigenF = vB * weights
    new = meanA + eigenF.sum()
    new = new.astype(np.uint8)
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
        w.append(np.dot(diff, v[i]).astype(np.uint8))
    return w


face1 = c.imread('images project 2/woman.jpg')

face2 = c.imread('images project 2/manReshaped.jpg')
face3 = c.imread('images project 2/kid2 - Copy.jpg')

# sizes arent the same 1090x1050 for kid2, and 1050x1090 for man and woman
# print(face3.shape)
# print(face1.shape)

# saving all variations/face into a list of images
dis_F1 = []
shapeF1 = None
for img in glob.glob("images project 2//womanvariations/*.jpg"):
    dis_F1.append(c.imread(img))
    shapeF1 = c.imread(img).shape

dis_F2 = []
shapeF2 = None
for img in glob.glob("images project 2//boyvariations/*.jpg"):
    dis_F2.append(c.imread(img))
    shapeF2 = c.imread(img).shape
#
dis_F3 = []
shapeF3 = None
for img in glob.glob("images project 2//manvariations/*.jpg"):
    dis_F3.append(c.imread(img))
    shapeF3 = c.imread(img).shape

# get the variations matrices
dF1 = getMatrixData(dis_F1)
dF2 = getMatrixData(dis_F2)
dF3 = getMatrixData(dis_F3)

# get the eigenvectors, values
# mean_1, vectorsW = c.PCACompute(dF1, mean= None)
# x = vectorsW[0].reshape((shapeF1))
# mean_2, vectorsB = c.PCACompute(dF2, mean= None)
# y = vectorsB[0].reshape(shapeF2)
# mean_3, vectorsM = c.PCACompute(dF3, mean= None)
# z = vectorsM[0].reshape(shapeF3)
# c.imwrite("Task 4.1-woman 1st eigenface.jpg", np.abs(x)/np.max(x)*255)
# c.imwrite("Task 4.1-boy 1st eigenface.jpg", np.abs(y)/np.max(y)*255)
# c.imwrite("Task 4.1-man 1st eigenface.jpg", np.abs(z)/np.max(z)*255)

# get the eigenvectors, values
# womanPCA = computePCA(dF1)
# wA = womanPCA.reshape(shapeF1)
# print(womanPCA)
boyPCA = computePCA(dF2)
bA = boyPCA.reshape(shapeF2)
manPCA = computePCA(dF3)
mA = manPCA.reshape(shapeF3)

# c.imshow("4.2avg woman", wA/np.max(wA))
c.imshow("4.2 avg boy",  bA)
c.imshow("4.2 avg man", mA/np.max(mA))
#
# womanPCA = c.resize(womanPCA, (400, 400))
print(f"{shapeF1}  {shapeF3} ")
# womanManPCA = combineFacesPCA(dF1, dF3).reshape()
# c.imshow("Task 4.3", womanManPCA.reshape(1090, 1050, 3))


c.waitKey(0)
c.destroyAllWindows()
# womanPCA = computePCA(dF1)
# # boyPCA = computePCA(dF2)
# # manPCA = computePCA(dF3
#
# womanPCA = c.resize(womanPCA, (400, 400))
# womanManPCA = combineFacesPCA(dF1, dF3)
# wmPCA = c.resize(womanManPCA, (400, 400))
# c.imshow('man and woman', wmPCA)
#
# c.imshow('woman PCA', womanPCA)
# # c.imshow('boy PCA', boyPCA)
# # c.imshow('man PCA', manPCA)
#
# # create matrix with variations
#
#
# c.waitKey(0)
# c.destroyAllWindows()
