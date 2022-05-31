import cv2 as c
import numpy as np
import scipy
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops, regionprops_table


def openANDclose(image, circle):
    opened = open(image, circle)
    closed = close(opened, circle)
    return closed


def circle_structure(n):
    struct = np.zeros((2 * n + 1, 2 * n + 1))
    x, y = np.indices((2 * n + 1, 2 * n + 1))
    #creating a mask ( eveything is black except the circle we just created )
    mask = (x - n)**2 + (y - n)**2 <= n**2
    struct[mask] = 1
    return struct


def granulo(image, sizes):
    for n in sizes:
        bit_and = c.bitwise_and(image, circle_structure(n))
        print(bit_and)
    return


def open(image, kernel):
    eroded = c.erode(image, kernel)
    dilated = c.dilate(eroded, kernel)
    return dilated

def close(image, kernel):
    dilated = c.dilate(image, kernel)
    eroded = c.erode(dilated, kernel)
    return eroded

# image passed has already been closed and opened
def countSE(imageoC, upperBound, lowerBound):
    label_im = label(imageoC)
    regions = regionprops(label_im)
    masks = []
    bbox = []
    list_of_index = []
    for num, x in enumerate(regions):
        area = x.area
        convex_area = x.convex_area
        if (num != 0 and (area > 10) and (convex_area / area < upperBound)
                and (convex_area / area > lowerBound)):
            masks.append(regions[num].convex_image)
            bbox.append(regions[num].bbox)
            list_of_index.append(num)
    count = len(masks)
    return count


def granulometry(image, maxsize):
    count = np.zeros(maxsize)
    for i in range(1, maxsize):
        circle = c.getStructuringElement(c.MORPH_ELLIPSE, (i,i))
        morph = openANDclose(image, circle)
        found = countSE(morph, 1.2, 0.8)
        if found is not None:
            count[i] = found
        else:
            print('no circles found with radius ' + i)
        print(count)

    return count


oranges = c.imread('images project 2/oranges.jpg')
orangeTree = c.imread('images project 2/orangetree.jpg')
granu = c.imread('images project 2/granulometry2.jpg')
oranges = np.uint8(oranges)
ot = np.uint8(orangeTree)
oranges = c.cvtColor(oranges, c.COLOR_BGR2RGB)
ot = c.cvtColor(ot, c.COLOR_BGR2RGB)
# ot = c.resize(ot, (400, 400))
# oranges = c.resize(oranges, (400, 400))
orangesB = c.cvtColor(oranges, c.COLOR_RGB2GRAY)
otB = c.cvtColor(ot, c.COLOR_RGB2GRAY)

# twixing the param for most convinience
num, orangeT = c.threshold(orangesB, 127, 255, c.THRESH_BINARY)
num2, otT = c.threshold(otB, 135, 255, c.THRESH_BINARY)

# using this size was suitable for both images
circleB = c.getStructuringElement(c.MORPH_ELLIPSE, (7, 7))

openO = open(orangeT, circleB)
closedO = close(openO, circleB)
openT = open(otT, circleB)
closedT = close(openT, circleB)

print(countSE(closedO, 1.56, 0.6))
print(countSE(closedT, 1.06, 0.95))
# print(granulometry(granu, 20))
gran = granulometry(granu, 100)
print(gran)



# # another way to count circles
# detected_circles = c.HoughCircles(ot,
#                    c.HOUGH_GRADIENT, 1, 20, param1 = 70,
#                param2 = 20, minRadius = 1, maxRadius = 40)
# counter =0
# for cir in detected_circles[0, :]:
#     counter = counter + 1
#
# print('counter', counter)


# circle = np.ones(())


# c.imshow('og', ot)
# # c.imshow('orange', )
# c.imshow('orangeT', otT)

c.waitKey(0)
c.destroyAllWindows()

