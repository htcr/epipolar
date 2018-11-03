import numpy as np
import cv2
from helper import displayEpipolarF
from submission import sevenpoint
import matplotlib.pyplot as plt

im1 = cv2.imread('../data/im1.png')[:, :, ::-1]
im2 = cv2.imread('../data/im2.png')[:, :, ::-1]
corresp =  np.load('../data/some_corresp.npz')
pts1 = corresp['pts1']
pts2 = corresp['pts2']

# select 7 points
N = pts1.shape[0]
select_ids = [53, 17, 43, 46, 27, 56, 77]
pts1 = pts1[select_ids, :].copy()
pts2 = pts2[select_ids, :].copy()

M = np.max(im1.shape)

Farray = sevenpoint(pts1, pts2, M)

for idx, F in enumerate(Farray):
    print('visualizing F%d' % idx)
    print(F)
    displayEpipolarF(im1, im2, F)

print('Enter the id of F which is correct:')
correct_id = int(input())
assert 0 <= correct_id < len(Farray)
F_correct = Farray[correct_id]
np.savez('q2_2.npz', F=F_correct, M=M, pts1=pts1, pts2=pts2)

'''
visualizing F0
[[ 1.37155686e-08 -8.61003099e-08 -8.27427609e-04]
 [ 2.16183531e-07  1.90452142e-09 -2.92701083e-05]
 [ 7.89066675e-04  1.84620407e-05  3.37863104e-03]]
visualizing F1
[[-1.48732736e-07  2.26439993e-06 -2.62220314e-04]
 [-1.97195996e-06 -2.63484236e-06  2.05684451e-03]
 [ 3.33774809e-04 -8.47499098e-04 -1.28974831e-01]]
visualizing F2
[[-8.44908260e-08  1.33486960e-06 -4.85737551e-04]
 [-1.10663534e-06 -1.59211276e-06  1.23186829e-03]
 [ 5.13824821e-04 -5.05045554e-04 -7.66342464e-02]]
'''