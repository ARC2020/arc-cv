## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import threading
import cv2

from datatypes import FramePackage

class FrameProducer(threading.Thread):
    
    def __init__(self, width, height):
        threading.Thread.__init__(self)
        print("FRAMEPRODUCER: Starting Up...")
        self.buffer = None
        self.stop_condition = False
        self.width = width
        self.height = height
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.enable()
    
    def enable(self):
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgba8, 30)
    
    def stop(self):
        self.stop_condition = True

    def attach_pipe(self, buffer_pipe):
        self.buffer = buffer_pipe
        
    def disable(self):
        self.pipeline.stop()
    
    def run(self):
        try:
            # Start streaming
            self.pipeline.start(self.config)
            print("FRAMEPRODUCER: Stream Initialized.")
            while True:
                if self.stop_condition:
                    break
                # Wait for a coherent pair of frames: depth and color
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                # if not depth_frame or not color_frame:
                if not color_frame or not depth_frame:
                    continue

                # Convert image to numpy array
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                if self.buffer != None and not self.buffer.full():
                    frame_pack = FramePackage(color_image, depth_image)
                    self.buffer.put(frame_pack, False)
                    print("FRAMEPRODUCER: Frame Deployed.")

        finally:
            self.pipeline.stop()
            print("FRAMEPRODUCER: Shutting Down...")