#!/usr/bin/python
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import jetson.inference
import jetson.utils

import argparse
import ctypes
import sys

import pyrealsense2 as rs
import numpy as np
import cv2

from vidgear.gears import NetGear

options = {'flag' : 0, 'copy' : False, 'track' : True}

#change following IP address '192.168.x.xxx' with client's IP address
server = NetGear(address = '192.168.0.18', colorspace='COLORBGR2RGB', port = '5454', protocol = 'tcp',  pattern = 0, receive_mode = False, logging = True, **options)

# parse the command line
parser = argparse.ArgumentParser(description="Segment a live camera stream using an semantic segmentation DNN.", 
						   formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.segNet.Usage())

parser.add_argument("--network", type=str, default="fcn-resnet18-voc", help="pre-trained model to load, see below for options")
parser.add_argument("--filter-mode", type=str, default="linear", choices=["point", "linear"], help="filtering mode used during visualization, options are:\n  'point' or 'linear' (default: 'linear')")
parser.add_argument("--ignore-class", type=str, default="void", help="optional name of class to ignore in the visualization results (default: 'void')")
parser.add_argument("--alpha", type=float, default=175.0, help="alpha blending value to use during overlay, between 0.0 and 255.0 (default: 175.0)")
parser.add_argument("--camera", type=str, default="0", help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")
parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the segmentation network
net = jetson.inference.segNet(opt.network, sys.argv)

# set the alpha blending value
net.SetOverlayAlpha(opt.alpha)

# allocate the output images for the overlay & mask
img_overlay = jetson.utils.cudaAllocMapped(opt.width * opt.height * 4 * ctypes.sizeof(ctypes.c_float))
img_mask = jetson.utils.cudaAllocMapped(opt.width//2 * opt.height//2 * 4 * ctypes.sizeof(ctypes.c_float))

# create the camera and display
#camera = jetson.utils.gstCamera(opt.width, opt.height, opt.camera)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgba8, 30)

# Start streaming
pipeline.start(config)

# process frames until user exits
while True:
    # capture the image
    # Wait for a coherent pair of frames: depth and color
    try:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
                continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())


        # Convert to CUDA format
        # image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGBA)
        width = color_image.shape[1]
        height = color_image.shape[0]
        img = jetson.utils.cudaFromNumpy(color_image)

        # process the segmentation network
        net.Process(img, width, height, opt.ignore_class)

        # generate the overlay and mask
        net.Overlay(img_overlay, width, height, opt.filter_mode)
        net.Mask(img_mask, width//2, height//2, opt.filter_mode)

        output_frame = jetson.utils.cudaToNumpy(img_overlay, width, height, 4)
        # output = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
        # frame = np.copy(output_frame)
        # print(type(output_frame))
        # print(output_frame.dtype)
        ## print(output_frame.shape)

        output_noalpha = output_frame[:,:,:3]
        # final_output = np.zeros((width, height))

        # info = np.finfo(output_noalpha.dtype) # Get the information of the incoming image type
        # data = output_noalpha.astype(np.float32) / (info.max) # normalize the data to 0 - 1
        # data = 255 * data # Now scale by 255
        final_output = output_noalpha.astype(np.uint8)

        # cv2.normalize(output_noalpha, final_output, 0, 255, norm_type=cv2.NORM_MINMAX)
        # final_output = final_output.astype(np.uint8)

        # print(color_image.dtype)
        # print(color_image)
        # print("------------------------------")
        # print("------------------------------")
        # print(output_frame.dtype)
        # print(output_frame)
        # print("------------------------------")
        # print("------------------------------")
        # print(final_output.dtype)
        # print(final_output)
        
        server.send(final_output)

    except KeyboardInterrupt:
        break

pipeline.stop()
server.close()
