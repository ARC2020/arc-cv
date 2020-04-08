from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jetson.inference
import jetson.utils
import cv2
import numpy as np

import ctypes
import sys
import os

import threading
from vidgear.gears import NetGear
from datatypes.FramePackage import FramePackage
from pipeline.LaneDetect import LaneDetect
from pipeline.ObjectDetect import ObjectDetect
from modules.arc_comms.NetworkPackage import NetworkPackage

class SegProducer(threading.Thread):
    def __init__(self, network, width, height):
        threading.Thread.__init__(self)
        print("SEGPRODUCER: Starting Up...")
        self.buffer = None
        self.dispatch_pipe = None
        self.stop_condition = False
        # load the segmentation network
        self.network = network
        self.net = jetson.inference.segNet(network, [])
        # set the alpha blending value
        self.net.SetOverlayAlpha(175.0)
        print("SEGPRODUCER: Network Initialized.")

        options = {'flag' : 0, 'copy' : False, 'track' : True}
        self.server = NetGear(address = '192.168.0.14', colorspace='COLORBGR2RGB', port = '5454', protocol = 'tcp',  pattern = 0, receive_mode = False, logging = True, **options)


        # set the alpha blending value
        # self.net.SetOverlayAlpha(175.0)

        # allocate the output images for the overlay & mask
        self.img_overlay = jetson.utils.cudaAllocMapped(width * height * 4 * ctypes.sizeof(ctypes.c_float))
        self.img_mask = jetson.utils.cudaAllocMapped(width * height * 4 * ctypes.sizeof(ctypes.c_float))
        print("SEGPRODUCER: CUDA Memory Initialized.")
    
    def get_id(self): 
  
        # returns id of the respective thread 
        if hasattr(self, '_thread_id'): 
            return self._thread_id 
        for id, thread in threading._active.items(): 
            if thread is self: 
                return id

    def stop(self):
        self.stop_condition = True
        thread_id = self.get_id() 
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 
              ctypes.py_object(SystemExit)) 
        if res > 1: 
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0) 
            print('Exception raise failure')

    def frame(self, frame):
        # Convert to CUDA format
        width = frame.shape[1]
        height = frame.shape[0]
        img = jetson.utils.cudaFromNumpy(frame)

        # process the segmentation network
        self.net.Process(img, width, height)

        # generate the overlay and mask
        filter_mode = "point"
        self.net.Overlay(self.img_overlay, width, height, filter_mode)
        self.net.Mask(self.img_mask, width, height, filter_mode)

        print("SEGPRODUCER: {:s} | Network {:.0f} FPS\r".format(self.network, self.net.GetNetworkFPS()))

        output_frame = jetson.utils.cudaToNumpy(self.img_overlay, width, height, 4)
        output_noalpha = output_frame[:,:,:3]
        final_output = output_noalpha.astype(np.uint8)

        return final_output
    
    def attach_pipe(self, buffer_pipe):
        self.buffer = buffer_pipe
    
    def attach_dispatch_pipe(self, dispatch_pipe):
        self.dispatch_pipe = dispatch_pipe
    
    def run(self):
        try:
            while True:
                if self.stop_condition:
                    break
                if self.buffer == None:
                    continue
                if not self.buffer.empty():
                    frame_pack = self.buffer.get_nowait()
                    frame = frame_pack.getColorFrame()
                    depth = frame_pack.getDepthFrame()
                    output_ready = self.frame(frame)
                    pipeline_frame_pack = FramePackage(output_ready, depth)
                    laneDetectModule = LaneDetect(pipeline_frame_pack)
                    laneData, laneOut = laneDetectModule.run(output_ready)
                    objectDetectModule = ObjectDetect(pipeline_frame_pack)
                    objects, objOut = objectDetectModule.run(laneOut)
                    
                    if self.dispatch_pipe != None and not self.dispatch_pipe.full():
                        networkPacket = NetworkPackage(laneData, objects)
                        self.dispatch_pipe.put(networkPacket, False)

                    # alpha = 1
                    # cv2.addWeighted(laneOut, alpha, output_ready, 1-alpha, 0, output_ready)
                    # cv2.addWeighted(objOut, alpha, output_ready, 1-alpha, 0, output_ready)

                    self.server.send(objOut)

                # # Show output window
                # cv2.imshow("Output Frame", output_ready)

                # key = cv2.waitKey(1) & 0xFF
                # # check for 'q' key-press
                # if key == ord("q"):
                #     #if 'q' key-pressed break out
                #     break
        finally:
            # # close output window
            # cv2.destroyAllWindows()
            self.server.close()
            print("SEGPRODUCER: Shutting Down...")