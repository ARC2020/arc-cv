from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys

import collections
import random

import numpy as np
import scipy as scp
import scipy.misc
import tensorflow as tf
import threading
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

class AIConsumer(threading.Thread):
    def __init__(self, numThreads):
        threading.Thread.__init__(self)
        self.hypes = 0
        self.logdir = 0
        self.prediction = 0
        self.image_pl = 0
        self.sessions = []
        self.numThreads = numThreads

    def createSession(self):
        # Create a session for running Ops on the Graph.
        sess = tf.Session()
        saver = tf.train.Saver()

        # Load weights from logdir
        core.load_weights(self.logdir, sess, saver)
    
        print("Info: Weights loaded successfully.")
        return sess
    
    @staticmethod
    def inference(data, hypes, prediction, image_pl, sess):
        # Load and resize input image
        # image = scp.misc.imread(input_image)
        # if hypes['jitter']['reseize_image']:
            # Resize input only, if specified in hypes
        print("Running inference: ")
        image_height = hypes['jitter']['image_height']
        image_width = hypes['jitter']['image_width']
        # print(image_height, image_width)
        image = scp.misc.imresize(data, size=(image_height, image_width),
                                    interp='cubic')
        
        # Run KittiSeg model on image
        feed = {image_pl: image}
        softmax = prediction['softmax']
        print("Running session...")
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

        scp.misc.imsave(str(random.randint(0, 1000)) + ".png", raw_image_overlay)


    def run(self):                
        workerPool = ThreadPool(self.numThreads, self.sessions)
        
        data = scp.misc.imread("DATA/demo/demo.png")
        
        for _ in range(self.numThreads):
            print("Adding task...")
            workerPool.add_task(AIConsumer.inference, data, self.hypes, self.prediction, self.image_pl)
        
        workerPool.wait_completion()
        print("Completed all tasks.")
    
    def begin(self):
        tv_utils.set_gpus_to_use()

        runs_dir = 'RUNS'
        self.logdir = os.path.join(runs_dir, default_run)

        # Loading hyperparameters from logdir
        self.hypes = tv_utils.load_hypes_from_logdir(self.logdir, base_path='hypes')

        print("Info: Hypes loaded successfully.")

        # Loading tv modules (encoder.py, decoder.py, eval.py) from logdir
        modules = tv_utils.load_modules_from_logdir(self.logdir)
        print("Info: Modules loaded successfully. Starting to build tf graph.")

        # Create tf graph and build module.
        with tf.Graph().as_default():

            # Create placeholder for input
            image_pl_i = tf.placeholder(tf.float32)
            image = tf.expand_dims(image_pl_i, 0)
            self.image_pl = image_pl_i

            # build Tensorflow graph using the model from logdir
            self.prediction = core.build_inference_graph(self.hypes, modules,
                                                    image=image)

            print("Info: Graph build successfully.")
            self.sessions = []
            for i in range(self.numThreads):
                self.sessions.append(self.createSession())

            print("Sessions created.")
        
        self.start()
    
