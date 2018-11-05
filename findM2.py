'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, p1, p2, R and P to q3_3.mat
'''
import numpy as np
import cv2
from submission import eightpoint, essentialMatrix, triangulate
from helper import camera2
import matplotlib.pyplot as plt

im1 = cv2.imread('../data/im1.png')[:, :, ::-1]
im2 = cv2.imread('../data/im2.png')[:, :, ::-1]
corresp =  np.load('../data/some_corresp.npz')
pts1 = corresp['pts1']
pts2 = corresp['pts2']
M = np.max(im1.shape)

F = eightpoint(pts1, pts2, M)

intrinsics = np.load('../data/intrinsics.npz')
K1, K2 = intrinsics['K1'], intrinsics['K2']
E = essentialMatrix(F, K1, K2)

# M1, M2, camera extrinsics
# let camera 1 be the center of the world
M1 = np.zeros((3, 4), dtype=np.float32)
M1[[0, 1, 2], [0, 1, 2]] = 1
# obtain four possible M2s from E
M2s = camera2(E)
print(M2s.shape)

# reproject error
err = np.Inf
best_M2_idx = -1

C1 = K1 @ M1

# recovered point clouds, (N, 3)
Ps = list()
C2s = list()
# get best M2
for i in range(M2s.shape[2]):
    M2 = M2s[:, :, i]
    C2 = K2 @ M2
    C2s.append(C2)
    P, cur_err = triangulate(C1, pts1, C2, pts2)
    Ps.append(P)
    if cur_err < err:
        err = cur_err
        best_M2_idx = i
    print('Reprojection error of M2_%d: %f' % (i, cur_err))

M2 = M2s[:, :, best_M2_idx]
P = Ps[best_M2_idx]
C2 = C2s[best_M2_idx]

# visual verify
p_homo = np.concatenate((P.transpose(), np.ones((1, pts2.shape[0]))), axis=0)
pts2_project = C2 @ p_homo
xs_project = pts2_project[0, :] / pts2_project[2, :]
ys_project = pts2_project[1, :] / pts2_project[2, :]
xs = pts2[:, 0]
ys = pts2[:, 1]
ax = plt.subplot()
ax.imshow(im2)
ax.scatter(x=xs, y=ys, color=(0, 1, 0))
ax.scatter(x=xs_project, y=ys_project, color=(1, 0, 0))
plt.show()

np.savez('q3_3.npz', M2=M2, C2=C2, P=P)



    