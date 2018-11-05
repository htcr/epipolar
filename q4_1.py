import numpy as np
import cv2
from submission import eightpoint, epipolarCorrespondence
from helper import epipolarMatchGUI

im1 = cv2.imread('../data/im1.png')[:, :, ::-1]
im2 = cv2.imread('../data/im2.png')[:, :, ::-1]
corresp =  np.load('../data/some_corresp.npz')
pts1 = corresp['pts1']
pts2 = corresp['pts2']
M = np.max(im1.shape)

F = eightpoint(pts1, pts2, M)

np.savez('q4_1.npz', F=F, pts1=pts1, pts2=pts2)

epipolarMatchGUI(im1, im2, F)

