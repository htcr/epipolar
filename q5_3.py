import numpy as np
import cv2
from submission import eightpoint, ransacF, epipolarCorrespondence, triangulate, essentialMatrix, bundleAdjustment
from helper import camera2

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

im1 = cv2.imread('../data/im1.png')[:, :, ::-1]
im2 = cv2.imread('../data/im2.png')[:, :, ::-1]
corresp =  np.load('../data/some_corresp_noisy.npz')
pts1 = corresp['pts1']
pts2 = corresp['pts2']
M = np.max(im1.shape)

print('%d corresps' % pts1.shape[0])

F, inliers = ransacF(pts1, pts2, M)

pts1, pts2 = pts1[inliers, :], pts2[inliers, :]


use_temple = False

if use_temple:
    templeCoords = np.load('../data/templeCoords.npz')
    x1s = templeCoords['x1']
    y1s = templeCoords['y1']


    x2s, y2s = list(), list()

    # get x2 y2 from x1 y1
    for i in range(x1s.shape[0]):
        x1, y1 = x1s[i, 0], y1s[i, 0]
        x2, y2 = epipolarCorrespondence(im1, im2, F, x1, y1)
        x2s.append(x2)
        y2s.append(y2)

    x2s, y2s = np.array(x2s).reshape(-1, 1), np.array(y2s).reshape(-1, 1)

    pts1 = np.concatenate((x1s, y1s), axis=1)
    pts2 = np.concatenate((x2s, y2s), axis=1)

    N = pts2.shape[0]
    ax = plt.subplot()
    ax.imshow(im2)
    for i in range(N):
        x, y = pts2[i, :]
        ax.text(x, y, ('%d' % i), fontsize=10, color=(0, 1, 0))
    plt.show()

    # play
    #F = eightpoint(pts1, pts2, M)
    F, inliers = ransacF(pts1, pts2, M)
    pts1, pts2 = pts1[inliers, :], pts2[inliers, :]
    # play


# find M2 ============

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

C1 = K1 @ M1

# recovered point clouds, (N, 3)
Ps = list()
C2s = list()
chose_M2_idx = 2
# get best M2
for i in range(M2s.shape[2]):
    M2 = M2s[:, :, i]
    C2 = K2 @ M2
    C2s.append(C2)
    P, cur_err = triangulate(C1, pts1, C2, pts2)
    if np.min(P[:, 2]) > 0:
        chose_M2_idx = i
    Ps.append(P)
    print('Reprojection error of M2_%d: %f' % (i, cur_err))

# choose a best M2
# chose_M2_idx = 2
print('chosen M2 idx: %d' % chose_M2_idx)
M2 = M2s[:, :, chose_M2_idx]
P = Ps[chose_M2_idx]
C2 = C2s[chose_M2_idx]

# find M2 ============
# bundle adjustment

M2, P = bundleAdjustment(K1, M1, pts1, K2, M2, pts2, P)
C2 = K2 @ M2



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xmin, xmax = np.min(P[:, 0]), np.max(P[:, 0])
ymin, ymax = np.min(P[:, 1]), np.max(P[:, 1])
zmin, zmax = np.min(P[:, 2]), np.max(P[:, 2])

ax.set_xlim3d(xmin, xmax)
ax.set_ylim3d(ymin, ymax)
ax.set_zlim3d(zmin, zmax)

ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='b', marker='o')
plt.show()