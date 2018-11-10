import numpy as np
import cv2
from helper import displayEpipolarF
from submission import eightpoint, ransacF

im1 = cv2.imread('../data/im1.png')[:, :, ::-1]
im2 = cv2.imread('../data/im2.png')[:, :, ::-1]
corresp =  np.load('../data/some_corresp_noisy.npz')
pts1 = corresp['pts1']
pts2 = corresp['pts2']
M = np.max(im1.shape)


print('======no RANSAC=======')
F1 = eightpoint(pts1, pts2, M)
print(F1)
displayEpipolarF(im1, im2, F1)

print('======RANSAC=======')
F2, _ = ransacF(pts1, pts2, M)
print(F2)
displayEpipolarF(im1, im2, F2)