import numpy as np
import cv2
import matplotlib.pyplot as plt

im1 = cv2.imread('../data/im1.png')[:, :, ::-1]
im2 = cv2.imread('../data/im2.png')[:, :, ::-1]
corresp =  np.load('../data/some_corresp.npz')
pts1 = corresp['pts1']
pts2 = corresp['pts2']

N = pts1.shape[0]

ax = plt.subplot()
ax.imshow(im1)
for i in range(N):
    x, y = pts1[i, :]
    ax.text(x, y, ('%d' % i), fontsize=10, color=(0, 1, 0))
plt.show()