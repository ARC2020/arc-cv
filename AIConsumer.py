from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jetson.inference
import jetson.utils
import cv2

import ctypes
import sys
import os

import threading
from ThreadPool import ThreadPool

class AIConsumer(threading.Thread):
    instance = 0
    def __init__(self, network):
        threading.Thread.__init__(self)
        # load the segmentation network
        self.network = network
        self.net = jetson.inference.segNet(network, [])

        # set the alpha blending value
        self.net.SetOverlayAlpha(175.0)

        width = 1280
        height = 720

        # allocate the output images for the overlay & mask
        self.img_overlay = jetson.utils.cudaAllocMapped(width * height * 4 * ctypes.sizeof(ctypes.c_float))
        self.img_mask = jetson.utils.cudaAllocMapped(width//2 * height//2 * 4 * ctypes.sizeof(ctypes.c_float))
        
        if AIConsumer.instance != 0:
            del AIConsumer.instance
        AIConsumer.instance = self  
    
    @staticmethod
    def frame(frame):
        # Convert to CUDA format
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        width = image.shape[1]
        height = image.shape[0]
        img = jetson.utils.cudaFromNumpy(image)

        # process the segmentation network
        AIConsumer.instance.net.Process(image, width, height)

        # generate the overlay and mask
        filter_mode = "linear"
        AIConsumer.instance.net.Overlay(AIConsumer.instance.img_overlay, width, height, filter_mode)
        AIConsumer.instance.net.Mask(AIConsumer.instance.img_mask, width/2, height/2, filter_mode)
        
        print("{:s} | Network {:.0f} FPS\r".format(AIConsumer.instance.network, AIConsumer.instance.net.GetNetworkFPS()))

        output_frame = jetson.utils.numpyFromCuda(AIConsumer.instance.img_overlay)
        return output_frame

    def run(self):
        from moviepy.editor import VideoFileClip
        myclip = VideoFileClip('project_video.mp4')#.subclip(40,43)
        output_vid = 'output.mp4'
        clip = myclip.fl_image(AIConsumer.frame)
        clip.write_videofile(output_vid, audio=False)
