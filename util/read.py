import pyrealsense2 as rs
import numpy as np
import cv2
import pickle
import sys
sys.path.append("..")
from datatypes.FramePackage import FramePackage

def unpickle(fileName):
        try:
            with open(fileName, 'rb') as pickleIn:
                return pickle.load(pickleIn)
        except pickle.UnpicklingError as e:
            print('An exception occured: ', e)

frames = unpickle("recording.pickle")

print(frames[0].getDepthFrame())