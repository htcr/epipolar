"""
Homework4.
Replace 'pass' by your implementation.
"""

import numpy as np
import helper
import cv2

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    N = pts1.shape[0]

    # normalize pts
    pts1, pts2 = pts1/float(M), pts2/float(M)
    x1s = pts1[:, 0] #(N,)
    y1s = pts1[:, 1]
    x2s = pts2[:, 0]
    y2s = pts2[:, 1]
    
    # construct columns of A
    c0 = x2s * x1s
    c1 = x2s * y1s
    c2 = x2s
    c3 = y2s * x1s
    c4 = y2s * y1s
    c5 = y2s
    c6 = x1s
    c7 = y1s
    c8 = np.ones((N,), dtype=np.float32)
    
    A = np.stack((c0, c1, c2, c3, c4, c5, c6, c7, c8), axis=1)
    
    # solve a raw f
    U, singular_vals, Vt = np.linalg.svd(A)
    # f is the last column of V, so the last raw of Vt
    f = Vt[-1, :] #(9,)
    F_raw = f.reshape(3, 3)

    # adjust to rank2 by shutting the last singular value to 0
    U, singular_vals, Vt = np.linalg.svd(F_raw)
    S = np.zeros((3, 3), dtype=np.float32)
    for i in range(2):
        S[i, i] = singular_vals[i]
    F_norm = U @ S @ Vt
    
    # local minimization
    F_norm = helper.refineF(F_norm, pts1, pts2)
    
    # now get the F for unormalized coordinates
    # normalization transform
    T = np.zeros((3, 3), dtype=np.float32)
    T[0, 0] = 1.0 / M
    T[1, 1] = 1.0 / M
    T[2, 2] = 1.0

    F_unnorm = T.transpose() @ F_norm @ T
    return F_unnorm
    


'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    N = pts1.shape[0]

    # normalize pts
    pts1, pts2 = pts1/float(M), pts2/float(M)
    x1s = pts1[:, 0] #(N,)
    y1s = pts1[:, 1]
    x2s = pts2[:, 0]
    y2s = pts2[:, 1]
    
    # construct columns of A
    c0 = x2s * x1s
    c1 = x2s * y1s
    c2 = x2s
    c3 = y2s * x1s
    c4 = y2s * y1s
    c5 = y2s
    c6 = x1s
    c7 = y1s
    c8 = np.ones((N,), dtype=np.float32)
    
    A = np.stack((c0, c1, c2, c3, c4, c5, c6, c7, c8), axis=1)
    
    # solve a raw f
    U, singular_vals, Vt = np.linalg.svd(A)
    # fk is the last kth column of V, so the last raw of Vt
    f1, f2 = Vt[-1, :], Vt[-2, :] #(9,)
    F1, F2 = f1.reshape(3, 3), f2.reshape(3, 3)

    a, b = F1-F2, F2

    fun = lambda x: np.linalg.det(x*a + b)
    fun0 = fun(0)
    fun1 = fun(1)
    fun_1 = fun(-1)
    fun2 = fun(2)
    fun_2 = fun(-2)

    c0 = fun0
    c1 = (2.0/3)*(fun1-fun_1) - (1.0/12)*(fun2-fun_2)
    c3 = (1.0/12)*(fun2 - fun_2) - (1.0/6)*(fun1-fun_1)
    c2 = fun1 - c0 - c1 - c3

    roots = np.roots([c3, c2, c1, c0])
    roots_imag_mag = np.abs(np.imag(roots))
    eps = 0.000001
    roots = roots[roots_imag_mag < eps]

    # now get the F for unormalized coordinates
    # normalization transform
    T = np.zeros((3, 3), dtype=np.float32)
    T[0, 0] = 1.0 / M
    T[1, 1] = 1.0 / M
    T[2, 2] = 1.0

    Fs = list()
    for root in roots:
        F_norm = root*a + b
        F_unnorm = T.transpose() @ F_norm @ T
        Fs.append(F_unnorm)
    
    return Fs

    

'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    return K2.transpose() @ F @ K1


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    Ps = list()
    err = 0
    assert pts1.shape[0] == pts2.shape[0]
    N = pts1.shape[0]
    for i in range(N):
        x1, y1 = pts1[i, :]
        x2, y2 = pts2[i, :]
        # construct A
        A0 = C1[0, :] - x1*C1[2, :]
        A1 = C1[1, :] - y1*C1[2, :]
        A2 = C2[0, :] - x2*C2[2, :]
        A3 = C2[1, :] - y2*C2[2, :]
        A = np.stack((A0, A1, A2, A3), axis=0)
        # solve w, just find the null space
        U, s, Vt = np.linalg.svd(A)
        w_raw = Vt[-1, :] #(4,)
        w_3d = w_raw[0:3] / w_raw[3] #(3,)
        Ps.append(w_3d)
        
        # get reproject error
        w_homo = np.zeros((4, 1), dtype=np.float32)
        w_homo[0:3, 0] = w_3d
        w_homo[3, 0] = 1
        p1_rep = C1 @ w_homo
        p2_rep = C2 @ w_homo
        
        x1_rep, y1_rep = p1_rep[0:2, 0] / p1_rep[2, 0]
        x2_rep, y2_rep = p2_rep[0:2, 0] / p2_rep[2, 0]

        err += (x1_rep-x1)**2 + (y1_rep-y1)**2 + \
               (x2_rep-x2)**2 + (y2_rep-y2)**2
        
    P = np.stack(Ps, axis=0)
    return P, err
        



'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def get_kernel_response(im, x, y, kxs, kys):
    # im.shape = (h, w, c)
    h, w = im.shape[0:2]
    xs, ys = kxs+x, kys+y
    xs, ys = np.clip(xs, 0, h-1).astype(np.int32), np.clip(ys, 0, h-1).astype(np.int32)
    # (Nk, c)
    response = im[ys, xs, :]
    return response

def epipolarCorrespondence(im1, im2, F, x1, y1):
    # homogeneous
    p1 = np.array([[x1], [y1], [1]], dtype=np.float32)
    # epipolar line on im2, (3, 1)
    l2 = F @ p1
    # define search boundaries
    # because we don't expect matching point to be too far
    r = 30
    bl, bt, br, bb = x1-r, y1-r, x1+r, y1+r
    # get intersection points of l2 and boundaries. 
    # should pick 2 out of 4 points.
    ll = np.array([1, 0, -bl], dtype=np.float32).reshape(-1, 1)
    lt = np.array([0, 1, -bt], dtype=np.float32).reshape(-1, 1)
    lr = np.array([1, 0, -br], dtype=np.float32).reshape(-1, 1)
    lb = np.array([0, 1, -bb], dtype=np.float32).reshape(-1, 1)

    boundary_lines = [ll, lt, lr, lb]
    # intersection points
    intersect_points = [np.cross(l.reshape(-1), l2.reshape(-1)).reshape(-1, 1) for l in boundary_lines]
    # select the end points of search line segment
    end_points = list()
    for p in intersect_points:
        if np.abs(p[2, 0]) > 0.000001:
            ph = p / p[2, 0]
            max_dist = np.max(np.abs(ph[0:2, :]-p1[0:2, :]))
            if max_dist < r*(1.05):
                end_points.append(ph)
    assert len(end_points) >= 2
    # in corner cases may encounter this, just pick two far points
    search_begin = None
    search_end = None
    if len(end_points) > 2:
        p0 = end_points[0]
        for i in range(1, len(end_points)):
            p1 = end_points[i]
            if np.min(np.abs(p0[0:2, :]-p1[0:2, :])) > r*0.1:
                search_begin = p0[0:2, :]
                search_end = p1[0:2, :]
                break
    else:
        search_begin = end_points[0][0:2, :]
        search_end = end_points[1][0:2, :]
    assert search_begin is not None and search_end is not None
    
    # we now construct a kearnel template
    # kernel radius
    kr = 10
    kxs = np.arange(-kr, kr+1, 1.0)
    kys = kxs
    kxs, kys = np.meshgrid(kxs, kys)
    # relative sample positions
    kxs, kys = kxs.reshape(-1), kys.reshape(-1)
    # now generate gaussian sampling weight
    std = kr / 3.0
    kws = np.exp( -( (kxs**2 + kys**2)/(2*std**2) ) ) / (2*np.pi*std**2)
    
    # get source response
    r1 = get_kernel_response(im1, x1, y1, kxs, kys)

    # sweep the kernel over search segment
    steps = int(np.sum((search_begin-search_end)**2)**0.5)
    x_step = (search_end[0, 0] - search_begin[0, 0]) / steps
    y_step = (search_end[1, 0] - search_begin[1, 0]) / steps
    x2, y2 = search_begin[0, 0], search_begin[1, 0]
    min_dist = np.Inf
    best_x2, best_y2 = -1, -1
    for i in range(steps):
        # get target response
        r2 = get_kernel_response(im2, x2, y2, kxs, kys)
        # calc dist
        dists = np.sum((r2 - r1)**2, axis=1)**0.5 # (Nk,)
        dist = np.sum(dists*kws)
        if dist < min_dist:
            min_dist = dist
            best_x2, best_y2 = x2, y2

        x2, y2 = x2 + x_step, y2 + y_step
    
    # debug
    #im2_copy = im2.copy()
    #cv2.rectangle(im2_copy, (int(search_begin[0, 0]), int(search_begin[1, 0])), (int(search_end[0, 0]), int(search_end[1, 0])), color=(255, 255, 255))
    #im2[:, :, :] = im2_copy[:, :, :]

    return best_x2, best_y2, search_begin, search_end

'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
'''
def ransacF(pts1, pts2, M):
    # Replace pass by your implementation
    pass

'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    pass

'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    pass

'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    pass

'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    pass
