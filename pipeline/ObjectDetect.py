import cv2
import numpy as np
import threading
import math

from datatypes.FramePackage import FramePackage
from modules.arc_comms.ObjData import ObjData

class ObjectDetect():
    img = []
    a = []

    def __init__(self, frame_pack):
        self.frame_pack = frame_pack
    
    def cartesian_distance(self, keypoint, pixel):
        x = keypoint.pt[0] - pixel[0]
        y = keypoint.pt[1] - pixel[1]
        return math.sqrt(x*x + y*y)
            
    def run(self, overlay):
        self.img = self.frame_pack.getColorFrame()
        depth_frame = self.frame_pack.getDepthFrame()
        # overlay = self.img.copy()
        # output = self.img.copy()

        # TODO: Replace colour code with that of obstacle class
        self.img[np.where((self.img!=[200, 155, 75]).all(axis=2))] = [0, 0, 0]

        self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)

        # self.img[np.where((self.img!=0))] = 255

        # Detect simple blobs
        detector = cv2.SimpleBlobDetector_create()
        keypoints = detector.detect(self.img)
        obj_pixels = np.where((self.img!=0))

        # Text parameters
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 1
        fontColor              = (255,255,255)
        lineType               = 1

        objects = []

        for keypoint in keypoints:
            count = 0
            avg_depth = -1
            for pixel in obj_pixels:
                distance = self.cartesian_distance(keypoint, pixel)
                if (distance < keypoint.size):
                    # Get average depth to obstacle from pixels
                    avg_depth = np.sum(depth_frame[obj_pixels])
                    count += 1

            if count != 0:
                avg_depth /= count

            objects.append(ObjData(keypoint.pt[0], keypoint.pt[1], keypoint.size, avg_depth))
            cv2.addText(overlay,
                        avg_depth,
                        (keypoint.pt[0], keypoint.pt[1]),
                        font,
                        fontScale,
                        fontColor,
                        lineType)
    
        # alpha = 1
        # cv2.addWeighted(overlay, alpha, output, 1-alpha, 0, output)
                
        return objects, overlay


        



