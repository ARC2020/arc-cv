from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys

import collections

import numpy as np
import scipy as scp
import scipy.misc
import tensorflow as tf
from ThreadPool import ThreadPool

sys.path.insert(1, 'incl')

from seg_utils import seg_utils as seg

try:
    # Check whether setup was done correctly

    import tensorvision.utils as tv_utils
    import tensorvision.core as core
except ImportError:
    # You forgot to initialize submodules
    print("Error: Could not import the submodules.")
    print("Error: Please execute:"
                  "'git submodule update --init --recursive'")
    exit(1)

default_run = 'KittiSeg_pretrained'

hypes = 0
image_pl = 0
prediction = 0
sess = 0
frame_count = 0

def inference(image):
        # Load and resize input image
    # image = scp.misc.imread(input_image)
    # if hypes['jitter']['reseize_image']:
        # Resize input only, if specified in hypes
    image_height = hypes['jitter']['image_height']
    image_width = hypes['jitter']['image_width']
    # print(image_height, image_width)
    image = scp.misc.imresize(image, size=(image_height, image_width),
                                interp='cubic')
    global frame_count
    if ((frame_count % 60) < 21):
        frame_count = frame_count + 1
        return image
    
    # Run KittiSeg model on image
    feed = {image_pl: image}
    softmax = prediction['softmax']
    output = sess.run([softmax], feed_dict=feed)

    # Reshape output from flat vector to 2D Image
    shape = image.shape
    output_image = output[0][:, 1].reshape(shape[0], shape[1])

    # Plot confidences as red-blue overlay
    # rb_image = seg.make_overlay(image, output_image)

    # Accept all pixel with conf >= 0.5 as positive prediction
    # This creates a `hard` prediction result for class street
    threshold = 0.5
    street_prediction = output_image > threshold

    # Plot the hard prediction as green overlay
    # green_image = tv_utils.fast_overlay(image, street_prediction)
    raw_image_overlay = tv_utils.fast_overlay(image, output_image)

    frame_count = frame_count + 1
    return raw_image_overlay

    # Save output images to disk.
    # output_base_name = input_image

    # raw_image_name = output_base_name.split('.')[0] + '_raw.png'
    # rb_image_name = output_base_name.split('.')[0] + '_rb.png'
    # green_image_name = output_base_name.split('.')[0] + '_green.png'

    # scp.misc.imsave(raw_image_name, output_image)
    # scp.misc.imsave(rb_image_name, rb_image)
    # scp.misc.imsave(green_image_name, green_image)

def main(_):
    tv_utils.set_gpus_to_use()

    runs_dir = 'RUNS'
    logdir = os.path.join(runs_dir, default_run)

    # Loading hyperparameters from logdir
    hypes_i = tv_utils.load_hypes_from_logdir(logdir, base_path='hypes')

    print("Info: Hypes loaded successfully.")

    # Loading tv modules (encoder.py, decoder.py, eval.py) from logdir
    modules = tv_utils.load_modules_from_logdir(logdir)
    print("Info: Modules loaded successfully. Starting to build tf graph.")

    # Create tf graph and build module.
    with tf.Graph().as_default():
        # Create placeholder for input
        image_pl_i = tf.placeholder(tf.float32)
        image = tf.expand_dims(image_pl_i, 0)

        # build Tensorflow graph using the model from logdir
        prediction_i = core.build_inference_graph(hypes_i, modules,
                                                image=image)

        print("Info: Graph build successfully.")

        # Create a session for running Ops on the Graph.
        sess_i = tf.Session()
        saver = tf.train.Saver()

        # Load weights from logdir
        core.load_weights(logdir, sess_i, saver)
    
        print("Info: Weights loaded successfully.")

    # input_image = "DATA/demo/demo.png"
    # print("Info: Starting inference using {} as input".format(input_image))
    global hypes
    hypes = hypes_i
    global image_pl
    image_pl = image_pl_i
    global prediction
    prediction = prediction_i
    global sess
    sess = sess_i
    global frame_count
    frame_count = 0

    from moviepy.editor import VideoFileClip
    myclip = VideoFileClip('project_video.mp4')#.subclip(40,43)
    output_vid = 'output.mp4'
    clip = myclip.fl_image(inference)
    clip.write_videofile(output_vid, audio=False)


    # inference(input_image, hypes, image_pl, prediction, sess)


    # print("Info: ")
    # print("Info: Raw output image has been saved to: {}".format(
    #     os.path.realpath(raw_image_name)))
    # print("Info: Red-Blue overlay of confs have been saved to: {}".format(
    #     os.path.realpath(rb_image_name)))
    # print("Info: Green plot of predictions have been saved to: {}".format(
    #     os.path.realpath(green_image_name)))

    # print("Info: ")
    # print("Warn: Do NOT use this Code to evaluate multiple images.")

    # print("Warn: Demo.py is **very slow** and designed "
    #                 "to be a tutorial to show how the KittiSeg works.")
    # print("Warn: ")
    # print("Warn: Please see this comment, if you like to apply demo.py to"
    #                 "multiple images see:")
    # print("Warn: https://github.com/MarvinTeichmann/KittiBox/"
    #                 "issues/15#issuecomment-301800058")

if __name__ == '__main__':
    tf.app.run()
