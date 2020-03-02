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
    def __init__(self):
        threading.Thread.__init__(self)
        # load the segmentation network
        net = jetson.inference.segNet("fcn-resnet18-deepscene", [])
    
    @staticmethod
    def frame(image):
        

    def run(self):                
        from moviepy.editor import VideoFileClip
        myclip = VideoFileClip('project_video.mp4')#.subclip(40,43)
        output_vid = 'output.mp4'
        clip = myclip.fl_image(AIConsumer.frame)
        clip.write_videofile(output_vid, audio=False)    
