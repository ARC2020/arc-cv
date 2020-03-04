from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jetson.inference
import jetson.utils

import ctypes
import sys
import os

import threading
from ThreadPool import ThreadPool

class AIConsumer(threading.Thread):
    instance = 0
    def __init__(self):
        threading.Thread.__init__(self, network)
        # load the segmentation network
        self.network = network
        self.net = jetson.inference.segNet(network, [])

        # set the alpha blending value
        net.SetOverlayAlpha(175.0)

        # allocate the output images for the overlay & mask
        self.img_overlay = jetson.utils.cudaAllocMapped(opt.width * opt.height * 4 * ctypes.sizeof(ctypes.c_float))
        self.img_mask = jetson.utils.cudaAllocMapped(opt.width/2 * opt.height/2 * 4 * ctypes.sizeof(ctypes.c_float))
        
        del instance
        instance = self  
    
    @staticmethod
    def frame(image):
        # Convert to CUDA format
        width = image.shape[1]
        height = image.shape[0]
        img = jetson.utils.cudaFromNumpy(image)

        # process the segmentation network
        instance.net.Process(image, width, height)

        # generate the overlay and mask
        filter_mode = "linear"
        instance.net.Overlay(instance.img_overlay, width, height, filter_mode)
        instance.net.Mask(instance.img_mask, width/2, height/2, filter_mode)
        
        print("{:s} | Network {:.0f} FPS\r".format(instance.network, instance.net.GetNetworkFPS()))

        output_frame = jetson.utils.numpyFromCuda(instance.img_overlay)
        return output_frame

    def run(self):
        from moviepy.editor import VideoFileClip
        myclip = VideoFileClip('project_video.mp4')#.subclip(40,43)
        output_vid = 'output.mp4'
        clip = myclip.fl_image(AIConsumer.frame)
        clip.write_videofile(output_vid, audio=False)
