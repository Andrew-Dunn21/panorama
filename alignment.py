import math
import random

import cv2
import numpy as np

eTranslate = 0
eHomography = 1


def computeHomography(f1, f2, matches, A_out=None):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
            KeyPoint.pt holds a tuple of pixel coordinates (x, y)
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        A_out -- ignore this parameter. If computeHomography is needed
                 in other TODOs, call computeHomography(f1,f2,matches)
    Output:
        H -- 2D homography (3x3 matrix)
        Takes two lists of features, f1 and f2, and a list of feature
        matches, and estimates a homography from image 1 to image 2 from the matches.
    '''
    #BEGIN TODO 2
    # Construct the A matrix that will be used to compute the homography
    # based on the given set of matches among feature sets f1 and f2.

    A_n = len(matches)
    A = np.zeros((2*A_n, 9))
    for i in range(A_n):
        #Get the vals from the list
        xn,yn = f1[matches[i].queryIdx].pt
        xnp, ynp = f2[matches[i].trainIdx].pt
        #Put 'em in A using numpy slice-fu
        A[2*i,0:3] = xn,yn,1
        A[2*i,6:] = -xnp*xn, -xnp*yn, -xnp
        A[2*i+1, 3:] = xn, yn, 1, -ynp*xn, -ynp*yn, -ynp
    
    #END TODO

    if A_out is not None:
        A_out[:] = A

    x = minimizeAx(A) # find x that minimizes ||Ax||


    H = np.eye(3) # create the homography matrix

    #BEGIN TODO 3
    #Fill the homography H with the correct values
    
    x = np.reshape(x,(3,3)) #Squish the x
    H[:,:] = x[:,:] #Feed H the x
##    for j  in range(9):
##        H[j] = x[j]
##    H = np.reshape(H,(3,3))
    #Norm H
    H = H * (1/H[2,2])
    
##    print()
##    print(H)

    #END TODO

    return H

def alignPair(f1, f2, matches, m, nRANSAC, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        nRANSAC -- number of RANSAC iterations
        RANSACthresh -- RANSAC distance threshold

    Output:
        M -- inter-image transformation matrix
        Repeat for nRANSAC iterations:
            Choose a minimal set of feature matches.
            Estimate the transformation implied by these matches
            count the number of inliers.
        For the transformation with the maximum number of inliers,
        compute the least squares motion estimate using the inliers,
        and return as a transformation matrix M.
    '''

    #BEGIN TODO 4
    #Write this entire method.  You need to handle two types of
    #motion models, pure translations (m == eTranslate) and
    #full homographies (m == eHomography).  However, you should
    #only have one outer loop to perform the RANSAC code, as
    #the use of RANSAC is almost identical for both cases.

    #Your homography handling code should call computeHomography.
    #This function should also call getInliers and, at the end,
    #least_squares_fit.
    
    #First, set s based on the motion model
    if m == eTranslate:
        s = 1
    else:
        s = 4
        
    #RANSAC Loop
    best = []
    M = np.eye(3)
    print('match count: ' + str(len(matches)))
    for i in range(nRANSAC):
        #Get the points
        d_i = np.random.choice(matches, s, replace=False)
        #Build the model
        Mi = computeHomography(f1, f2, d_i)
##        if m == eTranslate:
##            Mi = np.eye(3)
##            Mi[0,2] = f2[d_i[0].trainIdx].pt[0] - f1[d_i[0].queryIdx].pt[0]
##            Mi[1,2] = f2[d_i[0].trainIdx].pt[1] - f1[d_i[0].queryIdx].pt[1]
##        else:
##            Mi = computeHomography(f1, f2, d_i)
        #Get the inliers
        innies = getInliers(f1, f2, d_i, Mi, RANSACthresh)
        
        #Update best if innies is the new best
        if len(innies) > len(best):
            print('\nprev best: ' + str(len(best)) + ' current ins #: ' + str(len(innies)))
            best = innies
    M = leastSquaresFit(f1, f2, matches, m, best)
    
    #END TODO
    return M
    

def getInliers(f1, f2, matches, M, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        M -- inter-image transformation matrix
        RANSACthresh -- RANSAC distance threshold

    Output:
        inlier_indices -- inlier match indices (indexes into 'matches')

        Transform the matched features in f1 by M.
        Store the match index of features in f1 for which the transformed
        feature is within Euclidean distance RANSACthresh of its match
        in f2.
        Return the array of the match indices of these features.
    '''

    inlier_indices = []

    for i in range(len(matches)):
        #BEGIN TODO 5
        # Determine if the ith matched feature f1[matches[i].queryIdx], when
        # transformed by M, is within RANSACthresh of its match in f2.
        # If so, append i to inliers
        #TODO-BLOCK-BEGIN
        
        #Get the first coords
        x,y = f1[matches[i].queryIdx].pt
        #Move them
        test = np.dot(M,np.array([[x],[y],[1]]))
        
        if test[2] != 0:
            test /= test[2]
        else:
            print('test:')
            print(test)
            print('M:')
            print(M)
            raise Exception("Can't divide by zero")
        x1,y1 = test[:2]
        x1 = x1[0]
        y1 = y1[0]
        #Get the second coords
        xp,yp = f2[matches[i].trainIdx].pt
        #Test them
        dist = ((xp-x)**2 + (yp-y)**2)**0.5
        if dist < RANSACthresh:#If they're good, keep 'em
            inlier_indices.append(i)
        
        #TODO-BLOCK-END
        #END TODO
##    print(len(inlier_indices), end=' ')
    return inlier_indices

def leastSquaresFit(f1, f2, matches, m, inlier_indices):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        inlier_indices -- inlier match indices (indexes into 'matches')

    Output:
        M - transformation matrix

        Compute the transformation matrix from f1 to f2 using only the
        inliers and return it.
    '''

    # This function needs to handle two possible motion models,
    # pure translations (eTranslate)
    # and full homographies (eHomography).

    M = np.eye(3)

    if m == eTranslate:
        #For spherically warped images, the transformation is a
        #translation and only has two degrees of freedom.
        #Therefore, we simply compute the average translation vector
        #between the feature in f1 and its match in f2 for all inliers.

        u = 0.0
        v = 0.0

        #BEGIN TODO 6 :Compute the average translation vector over all inliers.
        # Fill in the appropriate entries of M to represent the average
        # translation transformation.
        for j in range(len(inlier_indices)):
            match = matches[inlier_indices[j]]
            u += f2[match.trainIdx].pt[0] - f1[match.queryIdx].pt[0]
            v += f2[match.trainIdx].pt[1] - f1[match.queryIdx].pt[1]
        u = u / len(inlier_indices)
        v = v / len(inlier_indices)
        
        
        M[0:2,2] = u,v
        #END TODO

    elif m == eHomography:
        #BEGIN TODO 7
        #Compute a homography M using all inliers. This should call
        # computeHomography.
        in_matches = np.empty(len(inlier_indices), dtype=cv2.DMatch)
        for i in range(len(inlier_indices)):
            in_matches[i] = matches[inlier_indices[i]]
        
        M = computeHomography(f1, f2, in_matches)
        
        #END TODO

    else:
        raise Exception("Error: Invalid motion model.")

    return M


def minimizeAx(A):
    """ Given an n-by-m array A, return the 1-by-m vector x that minimizes
    ||Ax||^2 subject to ||x|| = 1.  This turns out to be the right singular
    vector of A corresponding to the smallest singular value."""
    return np.linalg.svd(A)[2][-1,:]
