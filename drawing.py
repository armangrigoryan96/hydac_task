import matplotlib.pyplot as plt
import numpy as np
import cv2
img = cv2.imread('1.png')
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray)
plt.show()


