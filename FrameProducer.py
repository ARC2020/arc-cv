## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import threading
import cv2

class FrameProducer(threading.Thread):
    
    def __init__(self):
        threading.Thread.__init__(self)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.enable()
    
    def enable(self):
        try:
            # Configure depth and color streams
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

            # Start streaming
            self.pipeline.start(config)
        finally:
            self.pipeline.stop()
        
    def disable(self):
        self.pipeline.stop()
    
    def run(self):
        try:
            while True:
                # Wait for a coherent pair of frames: depth and color
                frames = self.pipeline.wait_for_frames()
                # depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                # if not depth_frame or not color_frame:
                if not color_frame:
                    continue

                # Convert image to numpy array
                color_image = np.asanyarray(color_frame.get_data())
        finally:
            self.pipeline.stop()