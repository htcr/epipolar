import numpy as np
import cv2
from helper import displayEpipolarF
from submission import eightpoint

im1 = cv2.imread('../data/im1.png')[:, :, ::-1]
im2 = cv2.imread('../data/im2.png')[:, :, ::-1]
corresp =  np.load('../data/some_corresp.npz')
pts1 = corresp['pts1']
pts2 = corresp['pts2']
M = np.max(im1.shape)

F = eightpoint(pts1, pts2, M)
print(F)
np.savez('q2_1.npz', F=F, M=M)
displayEpipolarF(im1, im2, F)