import math
import sys

import cv2
import numpy as np


class ImageInfo:
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position


def imageBoundingBox(img, M):
    """
       This is a useful helper function that that takes an image and a
       transform, and computes the bounding box of the transformed image.

       INPUT:
         img: image to get the bounding box of
         M: the transformation to apply to the img
       OUTPUT:
         minX: int for the minimum X value of a corner
         minY: int for the minimum Y value of a corner
         maxX: int for the maximum X value of a corner
         maxY: int for the maximum Y value of a corner
    """
    #TODO 8 determine the outputs for this method.
    h,w = img.shape[:2]
    #Get the corners
    uL = np.array([[0],[0],[1]])
    uR = np.array([[w-1],[0],[1]])
    bL = np.array([[0],[h-1],[1]])
    bR = np.array([[w-1],[h-1],[1]])
    #Transform the corners
    uL = np.dot(M, uL)
    uR = np.dot(M, uR)
    bL = np.dot(M, bL)
    bR = np.dot(M, bR)
    #Choose the vals
    minY = min(uL[1], bL[1], uR[1], bR[1])
    maxY = max(uL[1], bL[1], uR[1], bR[1])
    minX = min(bL[0], bR[0], uR[0], uL[0])
    maxX = max(bL[0], bR[0], uR[0], uL[0])
    
    #TODO-BLOCK-END
    return int(minX), int(minY), int(maxX), int(maxY)


def accumulateBlend(img, acc, M, blendWidth):
    """
       INPUT:
         img: image to add to the accumulator
         acc: portion of the accumulated image where img should be added
         M: the transformation mapping the input image to the accumulator
         blendWidth: width of blending function. horizontal hat function
       OUTPUT:
         modify acc with weighted copy of img added where the first
         three channels of acc record the weighted sum of the pixel colors
         and the fourth channel of acc records a sum of the weights
    """
    # convert input image to floats
    img = img.astype(np.float64) / 255.0

    # BEGIN TODO 10: Fill in this routine
    #Set y,x remembering rows by columns
    y,x = acc.shape[:2]
##    print('acc.shape: ' + str(acc.shape))
    brush = np.linspace(0,1,blendWidth)
    for i in range(y):
        for j in range(x):
            #Map the output coords to the input img and norm
            trans = np.dot(M, np.array([[j],[i],[1]]))
            if i<5 and j < 5:
                print(trans)
            ix, iy = trans[:2]/trans[2]
            #Now we interpolate
            acc[i,j,:] += interp(img, ix, iy)
    
            
    # END TODO

def interp(img, ix, iy):
    """
        INPUT:
            img: input image (without an alpha channel)
            ix: the warped input x-coord
            iy: the warped input y-coord
        OUTPUT:
            pix: the ndarray of values to get put into acc
                 *includes bonus alpha channel at no extra cost*
    """
    #Find out what we're working with remembering rows by columns for beta, alfa
    beta, alfa = img.shape[:2]
    #Get some bounds for the box
    x1 = int(np.floor(ix))
    x2 = int(np.ceil(ix))
    y1 = int(np.floor(iy))
    y2 = int(np.ceil(iy))
    #Set up an output
    pix = np.zeros(4)
##    print((ix,x1,x2,iy,y1,y2))
    if ix >= alfa-1 or iy >= beta-1 or ix < 0 or iy < 0:
        return pix
    
    I = img[y2,x2] * ((ix - x1) * (iy - y1))
    II = img[y1,x2] * ((ix-x1) * (y2 - iy))
    III = img[y1,x1] * ((x2 - ix) * (y2 - iy))
    IV = img[y2,x1] * ((x2 - ix) * (iy - y1))
    pix[:3] = I + II + III + IV
    
    if pix.any() != 0:
        pix[3] = 1
        print('*',end='')
        return pix
    else:
##        print('%',end='')
        return np.zeros(4)



def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    # BEGIN TODO 11: fill in this routine
    acc[acc[:,:,3]>0] /= acc[acc[:,:,3]>0]
    img = acc
    img[:,:,3] = 1
    # END TODO
    return (img * 255).astype(np.uint8)


def getAccSize(ipv):
    """
       This function takes a list of ImageInfo objects consisting of images and
       corresponding transforms and returns useful information about the
       accumulated image.

       INPUT:
         ipv: list of ImageInfo objects consisting of image (ImageInfo.img) and
             transform(image (ImageInfo.position))
       OUTPUT:
         accWidth: Width of accumulator image(minimum width such that all
             tranformed images lie within acc)
         accWidth: Height of accumulator image(minimum height such that all
             tranformed images lie within acc)

         channels: Number of channels in the accumulator image
         width: Width of each image(assumption: all input images have same width)
         translation: transformation matrix so that top-left corner of accumulator image is origin
    """

    # Compute bounding box for the mosaic
    minX = sys.maxsize
    minY = sys.maxsize
    maxX = 0
    maxY = 0
    channels = -1
    width = -1  # Assumes all images are the same width
    M = np.identity(3)
    for i in ipv:
        M = i.position
        img = i.img
        _, w, c = img.shape
        if channels == -1:
            channels = c
            width = w

        # BEGIN TODO 9
        # add some code here to update minX, ..., maxY
        # this can (should) use the code you wrote for TODO 8

        tminX, tminY, tmaxX, tmaxY = imageBoundingBox(img, M)
        if tminX < minX :
            minX = tminX
        if tminY < minY :
            minY = tminY
        if tmaxX > maxX:
            maxX = tmaxX
        if tmaxY > maxY:
            maxY = tmaxY
        
        #TODO-BLOCK-END
        # END TODO

    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    print('accWidth, accHeight:', (accWidth, accHeight))
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])
##    if accWidth > (len(ipv)*width):
##        raise Exception("Seems like a sus width")
                   
    return accWidth, accHeight, channels, width, translation


def pasteImages(ipv, translation, blendWidth, accWidth, accHeight, channels):
    acc = np.zeros((accHeight, accWidth, channels + 1))
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        M = i.position
        img = i.img

        M_trans = translation.dot(M)
        accumulateBlend(img, acc, M_trans, blendWidth)

    return acc


def getDriftParams(ipv, translation, width):
    """ Computes parameters for drift correction.
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         translation: transformation matrix so that top-left corner of accumulator image is origin
         width: Width of each image(assumption: all input images have same width)
       OUTPUT:
         x_init, y_init: coordinates in acc of the top left corner of the
            panorama with half the left image cropped out to match the right side
         x_final, y_final: coordinates in acc of the top right corner of the
            panorama with half the right image cropped out to match the left side
    """
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        if count != 0 and count != (len(ipv) - 1):
            continue

        M = i.position

        M_trans = translation.dot(M)

        p = np.array([0.5 * width, 0, 1])
        p = M_trans.dot(p)

        # First image
        if count == 0:
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(ipv) - 1):
            x_final, y_final = p[:2] / p[2]

    return x_init, y_init, x_final, y_final

def blendImages(ipv, blendWidth, is360=False, A_out=None):
    """
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         blendWidth: width of the blending function
       OUTPUT:
         croppedImage: final mosaic created by blending all images and
         correcting for any vertical drift
    """
    accWidth, accHeight, channels, width, translation = getAccSize(ipv)
    acc = pasteImages(
        ipv, translation, blendWidth, accWidth, accHeight, channels
    )
    compImage = normalizeBlend(acc)

    # Determine the final image width
    outputWidth = (accWidth - width) if is360 else accWidth
    x_init, y_init, x_final, y_final = getDriftParams(ipv, translation, width)
    # Compute the affine transform
    A = np.identity(3)
    if is360:
        # BEGIN TODO 12
        # 497P: you aren't required to do this. 360 mode won't work.

        # 597P: fill in appropriate entries in A to trim the left edge and
        # to take out the vertical drift:
        #   Shift it left by the correct amount
        #   Then handle the vertical drift - using a shear in the Y direction
        # Note: warpPerspective does forward mapping which means A is an affine
        # transform that maps accumulator coordinates to final panorama coordinates
        raise Exception("TODO 12 in blend.py not implemented")
        # END TODO 12

    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )
    print("Yay! We did it!")

    return croppedImage

