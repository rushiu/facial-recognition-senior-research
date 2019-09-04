import numpy as np
from scipy import misc
import cv2
import glob

#                  LE 			RE 		   N 		  LM 		 RM
template_pts = [(27, 53), (60, 53), (93, 144), (129, 180), (60, 181)]

for image_path in glob.glob("./transformedImages/*/*.png"):
    imName = image_path[19:-4]
    print(imName)

    img = cv2.imread(image_path)
    print(img.shape)
				
    cv2.imwrite('patchesF'+imName+'p1.jpg', cv2.resize(img[:, :], (188, 228)))
    cv2.imwrite('patchesF'+imName+'p2.jpg', cv2.resize(img[0:150, :], (188, 228)))
    cv2.imwrite('patchesF'+imName+'p3.jpg', cv2.resize(img[0:140, :], (188, 228)))
    cv2.imwrite('patchesF'+imName+'p4.jpg', cv2.resize(img[0:170, :], (188, 228)))
    cv2.imwrite('patchesF'+imName+'p5.jpg', cv2.resize(img[0:180, 0:130], (188, 228)))
    cv2.imwrite('patchesF'+imName+'p6.jpg', cv2.resize(img[0:180, 55:], (188, 228)))
    cv2.imwrite('patchesF'+imName+'p7.jpg', cv2.resize(img[100:, 0:140], (188, 228)))
    cv2.imwrite('patchesF'+imName+'p8.jpg', cv2.resize(img[100:, 40:], (188, 228)))


'''
#old p2
import numpy as np
from scipy import misc
import cv2
import glob

#                  LE 			RE 		   N 		  LM 		 RM
template_pts = [(27, 53), (60, 53), (93, 144), (129, 180), (60, 181)]


for image_path in glob.glob("./transformedImages/*/*.png"):
    imName = image_path[19:-4]
    print(imName)

    img = cv2.imread(image_path)

				
    cv2.imwrite('patches/p2'+imName+'.jpg', img[0:150, :])
'''
'''
img[:, :], #Normal pic 
img[0:150, :], #Above nose center - take normal pic and crop above nose center
img[0:140, :], #Below mouth corners
img[0:170, :], #Above mouth corners
img[0:180, 0:130], #Bottom right corner at RM - crop the normal pic for these
img[0:180, 55:], #Bottom left corner at LM
img[100:, 0:140], #Top right corner at RE
img[100:, 40:]] #Top left corner at LE
'''

