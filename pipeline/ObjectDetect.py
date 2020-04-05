import cv2
import numpy as np
import threading
import math

from datatypes import FramePackage
from modules.arc_comms import ObjData

class ObjectDetect():
    img = []
    a = []

    def __init__(self, frame_pack):
        self.frame_pack = frame_pack
    
    def cartesian_distance(self, keypoint, pixel):
        x = keypoint.x - pixel[0]
        y = keypoint.y - pixel[1]
        return sqrt(x*x + y*y)
            
    def run(self):
        self.img = self.frame_pack.getColorFrame()
        depth_frame = self.frame_pack.getDepthFrame()
        overlay = self.img.copy()
        output = self.img.copy()

        # TODO: Replace colour code with that of obstacle class
        self.img[np.where((self.img!=[200, 155, 75]).all(axis=2))] = [0, 0, 0]

        self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)

        # self.img[np.where((self.img!=0))] = 255

        # Detect simple blobs
        detector = cv2.SimpleBlobDetector()
        keypoints = detector.detect(self.img)
        obj_pixels = np.where((self.img!=0))

        # Text parameters
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 1
        fontColor              = (255,255,255)
        lineType               = 2

        objects = []

        for keypoint in keypoints:
            count = 0
            for pixel in obj_pixels:
                distance = self.cartesian_distance(keypoint, pixel)
                if (distance < keypoint._size):
                    # Get average depth to obstacle from pixels
                    avg_depth = np.sum(depth_frame[obj_pixels])
                    count += 1
            avg_depth /= count
            objects.append(ObjData(keypoint.x, keypoint.y, avg_depth))
            cv2.addText(overlay,
                        avg_depth,
                        (keypoint.x, keypoint.y),
                        font,
                        fontScale,
                        fontColor,
                        lineType)
    
        alpha = 1
        cv2.addWeighted(overlay, alpha, output, 1-alpha, 0, output)
                
        return objects, output


        


