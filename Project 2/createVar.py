import cv2 as c
import numpy as np


def changeBrightness(image, add):
    res = image.astype(np.uint8) + add
    return res


def getHSV(image):
    res = c.cvtColor(image, c.COLOR_BGR2RGB)
    res = c.cvtColor(res, c.COLOR_RGB2HSV)
    return res


face1 = c.imwrite('images project 2/womanVariation/woman.jpg')
face2 = c.imread('images project 2/boyVariations/kid2 - Copy.jpg')
face3 = c.imread('images project 2/manVars/manReshaped.jpg')

# face1 = c.resize(face1, (400, 400))
# face2 = c.resize(face2, (400, 400))
# face3 = c.resize(face3, (400, 400))
f1B = c.cvtColor(face1, c.COLOR_BGR2GRAY)
f2B = c.cvtColor(face2, c.COLOR_BGR2GRAY)
f3B = c.cvtColor(face3, c.COLOR_BGR2GRAY)

neg1 = np.abs(255- f1B)
neg2 = np.abs(255- f2B)
neg3 = np.abs(255- f3B)
#
# # c.imwrite('negW.jgp', neg1)
c.imwrite('images project 2/womanVariation/greyW.jpg', f1B)
c.imwrite('images project 2/womanVariation/negW.jpg', neg1)
c.imwrite('images project 2/boyVariations/greyB.jpg', f2B)
c.imwrite('images project 2/boyVariations/negB.jpg', neg2)


# c.imwrite('negM.jpg', neg3)
print("done")

c.waitKey(0)
c.destroyAllWindows()
