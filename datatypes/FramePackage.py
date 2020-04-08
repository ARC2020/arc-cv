class FramePackage:
    color_frame = 0
    depth_frame = 0

    def __init__(self, color, depth):
        self.color_frame = color
        print("---------------------------------------------", type(depth))
        self.depth_frame = depth
    
    def getColorFrame(self):
        return self.color_frame
    
    def getDepthFrame(self):
        return self.depth_frame