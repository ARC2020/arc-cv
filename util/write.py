from vidgear.gears import VideoGear
from vidgear.gears import WriteGear
import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

# Start streaming
pipeline.start(config)

writer = WriteGear(output_filename = 'capture.mp4') #Define writer 

# infinite loop
while True:
    try:
        # read frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
                continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())


        # do something with frame here

        # write frame to writer
        writer.write(color_image, rgb_mode=True) 
        
        # Show output window
        # cv2.imshow("Output Frame", color_frame)

        key = cv2.waitKey(1) & 0xFF
        # check for 'q' key-press
        # if key == ord("q"):
        #     #if 'q' key-pressed break out
        #     break
    except KeyboardInterrupt:
        break

print("Shutting Down..")
cv2.destroyAllWindows()
# close output window

# safely close video stream
writer.close()
# safely close writer
pipeline.stop()