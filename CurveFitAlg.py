import numpy as np
import cv2
import matplotlib.pyplot as plt

def exp_smoothing(img, alpha):
    out = []
    for i in range(len(img)):
        if i is 0:
            out.append(img[i])
        else:
            out.append(((alpha*img[i]) + (1-alpha)*out[i-1]))
    return out

def normalizedLaneDetect(a, thresh):
    out = []
    max_val = np.amax(a)
    for val in a:
        if  (val/max_val) >= thresh:
            out.append(1)
        else:
            out.append(0)   
    return out

def laneBound(a):
    critical_points = []
    polarity = 0
    last_val = 0
    for i in range(len(a)):
        if a[i] != last_val:
            critical_points.append(i)
            polarity += 1
            last_val = a[i]
    lhs = 0
    rhs = 0
    if polarity == 1:
        lhs = critical_points[0]
        rhs = len(a)
    elif polarity == 2:
        lhs = critical_points[0]
        rhs = critical_points[1]
    elif polarity == 3:
        lhs = critical_points[0]
        rhs = len(a)
    elif polarity >= 4:
        lhs = critical_points[0]
        rhs = critical_points[polarity-1]
    
    return lhs, rhs

def laneOut(lhs, rhs, length):
    out = []
    for i in range(length):
        if i >= lhs and i <= rhs:
            out.append(1)
        else:
            out.append(0)
    return out

def perspective_warp(img, 
                     dst_size=(1280,720),
                     src=np.float32([(0.43,0.65),(0.58,0.65),(0,1),(1,1)]),
                     dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result 
    # again, not exact, but close enough for our purposes
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped

def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist


img = cv2.imread('out-new1.jpg')

finalimg = cv2.imread('out-new.jpg')
output = finalimg.copy()
overlay = finalimg.copy()

img[np.where((img!=[200, 155, 75]).all(axis=2))] = [0, 0, 0]

img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

img[np.where((img!=0))] = 255

# Pipeline for Lane Detection
hist = get_hist(img)
hist = exp_smoothing(hist, 0.02)
hist = normalizedLaneDetect(hist, 0.4)
lhs, rhs = laneBound(hist)
print(lhs, rhs)
out = laneOut(lhs, rhs, len(hist))

# Add lane detection overlay
alpha = 0.5
height = img.shape[0]
cv2.rectangle(overlay, (lhs, height-90), (rhs, 1500), (0, 0, 255), -1)
cv2.addWeighted(overlay, alpha, output, 1-alpha, 0, output)

# plt.plot(out)
# plt.show()
    
cv2.imshow("Output Frame", output)

cv2.waitKey()

cv2.destroyAllWindows()
