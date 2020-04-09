# import pyrealsense2 as rs
import numpy as np
import cv2
import pickle
import sys
import time
sys.path.append("..")
from datatypes.FramePackage import FramePackage

def unpickle(fileName):
        try:
            with open(fileName, 'rb') as pickleIn:
                return pickle.load(pickleIn)
        except pickle.UnpicklingError as e:
            print('An exception occured: ', e)

frames = unpickle("recording.pickle")

count = 0
for frame in frames:
    count += 1
    if count < 100:
        continue
    img = frame.getColorFrame()
    cv2.imwrite('img.png', img.astype(np.uint8))
    img = cv2.cvtColor(frame.getColorFrame().astype(np.uint8), cv2.COLOR_RGBA2RGB)
    cv2.imshow("Output Frame", img.astype(np.uint8))
    
    key = cv2.waitKey(1) & 0xFF
	# check for 'q' key-press
    if key == ord("q"):
        #if 'q' key-pressed break out
        break
    
    # print(frame.getColorFrame().shape[1], frame.getColorFrame().shape[0])
    print(type(frame.getColorFrame()))
    time.sleep(0.05)
    break
input()
print(len(frames))
cv2.destroyAllWindows()