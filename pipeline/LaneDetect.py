import cv2
import numpy as np
import threading

from datatypes.FramePackage import FramePackage
from modules.arc_comms.LaneData import LaneData

class LaneDetect():
    img = []
    a = []

    def __init__(self, frame_pack):
        self.frame_pack = frame_pack

    def exp_smoothing(self, alpha):
        out = []
        for i in range(len(self.a)):
            if i is 0:
                out.append(self.a[i])
            else:
                out.append(((alpha*self.a[i]) + (1-alpha)*out[i-1]))
        return out

    def normalizedLaneDetect(self, thresh):
        out = []
        max_val = np.amax(self.a)
        for val in self.a:
            # print(type(self.a), val, max_val, thresh)
            if  (val/max_val) >= thresh:
                out.append(1)
            else:
                out.append(0)
        return out

    def laneBound(self):
        critical_points = []
        polarity = 0
        last_val = 0
        for i in range(len(self.a)):
            if self.a[i] != last_val:
                critical_points.append(i)
                polarity += 1
                last_val = self.a[i]
        lhs = 0
        rhs = 0
        if polarity == 1:
            lhs = critical_points[0]
            rhs = len(self.a)
        elif polarity == 2:
            lhs = critical_points[0]
            rhs = critical_points[1]
        elif polarity == 3:
            lhs = critical_points[0]
            rhs = len(self.a)
        elif polarity >= 4:
            lhs = critical_points[0]
            rhs = critical_points[polarity-1]
        
        return lhs, rhs

    def laneOut(self, lhs, rhs, length):
        out = []
        for i in range(length):
            if i >= lhs and i <= rhs:
                out.append(1)
            else:
                out.append(0)
        return out

    def perspective_warp(self,
                        dst_size=(1280,720),
                        src=np.float32([(0.43,0.65),(0.58,0.65),(0,1),(1,1)]),
                        dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])):
        img_size = np.float32([(self.img.shape[1],self.img.shape[0])])
        src = src*img_size
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result 
        # again, not exact, but close enough for our purposes
        dst = dst * np.float32(dst_size)
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(self.img, M, dst_size)
        return warped

    def get_hist(self):
        hist = np.sum(self.img[self.img.shape[0]//2:,:], axis=0)
        return hist
    
    def metric_convert(self, P, depth_frame, rhs):
        F = 543.45
        D = depth_frame[depth_frame.shape[0]-1][rhs-1]
        W = (P * D) / F
        return W
    
    def batch_convert(self, lhs, pob, rhs, depth_frame):
        lhs = self.metric_convert(rhs-lhs, depth_frame, rhs)
        pob = self.metric_convert(rhs-pob, depth_frame, rhs)
        return lhs, pob
        
    def run(self):
        self.img = self.frame_pack.getColorFrame()
        depth_frame = self.frame_pack.getDepthFrame()
        overlay = self.img.copy()
        output = self.img.copy()

        self.img[np.where((self.img!=[200, 155, 75]).all(axis=2))] = [0, 0, 0]

        self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)

        self.img[np.where((self.img!=0))] = 255

        # Pipeline for Lane Detection
        self.a = self.get_hist()
        self.a = self.exp_smoothing(0.02)
        self.a = self.normalizedLaneDetect(0.4)
        lhs_pixel, rhs_pixel = self.laneBound()
        # print(lhs_pixel, rhs_pixel)
        # out = self.laneOut(lhs_pixel, rhs_pixel, len(self.a))

        # Add lane detection overlay
        alpha = 0.5
        height = self.img.shape[0]
        cv2.rectangle(overlay, (lhs_pixel, height-90), (rhs_pixel, 500), (0, 0, 255), -1)
        cv2.addWeighted(overlay, alpha, output, 1-alpha, 0, output)

        pob_pixel = self.img.shape[1]/2
        lhs, pob = self.batch_convert(lhs_pixel, pob_pixel, rhs_pixel, depth_frame)
        laneData = LaneData(lhs, pob, lhs_pixel, pob_pixel)

        return laneData, output