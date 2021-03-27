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
    h,w = img.shape[:2]
    #Get the corners
    uL = np.array([[0],[0],[1]])
    uR = np.array([[w-1],[0],[1]])
    bL = np.array([[0],[h-1],[1]])
    bR = np.array([[w-1],[h-1],[1]])
    #Transform the corners
    uL = np.dot(M, uL)
    uL = uL/uL[2]
    uR = np.dot(M, uR)
    uR = uR/uR[2]
    bL = np.dot(M, bL)
    bL = bL/bL[2]
    bR = np.dot(M, bR)
    bR = bR/bR[2]
    #Choose the vals
    minY = min(uL[1], bL[1], uR[1], bR[1])
    maxY = max(uL[1], bL[1], uR[1], bR[1])
    minX = min(bL[0], bR[0], uR[0], uL[0])
    maxX = max(bL[0], bR[0], uR[0], uL[0])
    
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

    #Set y,x remembering rows by columns
    y,x = acc.shape[:2]
    #Add 4th img channel
    beta, alfa = img.shape[:2]
    img = np.dstack((img, np.ones((beta,alfa),dtype=img.dtype)))
    
    #Feather time (RIP runtime...)
    bl = blendWidth
    bw = np.linspace(0,1,blendWidth)
    wb = np.flip(bw)
    #Reshape the blending brush to make it broadcastable
    bw = np.dstack((bw,bw,bw,bw))
    bw = np.reshape(bw,(bl,4))
    wb = np.dstack((wb,wb,wb,wb))
    wb = np.reshape(wb,(bl,4))
    #Loop to apply weights
    for a in range(beta):#Rows Loop
        img[a,:bl,:] *= bw
        img[a,alfa-bl:,:] *= wb
    for c in range(alfa):#Cols Loop
        img[:bl,c,:] *= bw
        img[beta-bl:,c,:] *= wb

    M_i = np.linalg.inv(M)
    for i in range(y):
        for j in range(x):
            #Map the output coords to the input img and norm
            trans = np.dot(M_i, np.array([[j],[i],[1]]))
            ix, iy = trans[:2]
            ix = ix /trans[2]
            iy = iy /trans[2]
            #Now we interpolate
            if ix <= alfa-1 and ix >= 0:
                if iy <= beta-1 and iy >= 0:
                    acc[i,j,:] += interp(img, ix, iy)
            else:
                acc[i,j,:] += np.zeros(4)

def interp(img, ix, iy):
    """
        INPUT:
            img: input image
            ix: the warped input x-coord
            iy: the warped input y-coord
        OUTPUT:
            pix: the ndarray of values to get put into acc
                 
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

    if x2-x1 == 0:
        if y2-y1 == 0:
            pix = img[y1,x1]
    else:
        I = img[y2,x2] * ((ix - x1) * (iy - y1))
        II = img[y1,x2] * ((ix-x1) * (y2 - iy))
        III = img[y1,x1] * ((x2 - ix) * (y2 - iy))
        IV = img[y2,x1] * ((x2 - ix) * (iy - y1))
        pix = I + II + III + IV

    return pix



def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    
    #Get the dims
    y,x = acc.shape[:2]
    #Split into 4 channels for easy numpy-fu
    R,G,B,A = np.split(acc,4,axis=2)
    R = np.reshape(R,(y,x))
    G = np.reshape(G,(y,x))
    B = np.reshape(B,(y,x))
    A = np.reshape(A,(y,x))
    R[A>0] /= A[A>0]
    G[A>0] /= A[A>0]
    B[A>0] /= A[A>0]
    A[A>0] /= A[A>0]

    img = np.dstack((R,G,B,A))
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

        tminX, tminY, tmaxX, tmaxY = imageBoundingBox(img, M)
        minX = min(minX, tminX)
        minY = min(minY, tminY)
        maxX = max(maxX, tmaxX)
        maxY = max(maxY, tmaxY)

    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    print('accWidth, accHeight:', (accWidth, accHeight))
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])
                   
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
        raise Exception("360 degree panorama in blend.py not implemented")

    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )
    print("Yay! We did it!")

    return croppedImage

