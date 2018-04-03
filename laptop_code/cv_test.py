import cv2
import numpy as np
import time
SIZE = 100
img = np.zeros([SIZE,SIZE,3])

img[:,:,0] = np.ones([SIZE,SIZE])*64
cv2.imshow("test", img)
cv2.waitKey(0)
img[:,:,0] = np.zeros([SIZE, SIZE])
img[:,:,1] = np.ones([SIZE, SIZE]) * 64
cv2.imshow("test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
