# import libraries
from vidgear.gears import NetGear
import cv2

options = {'flag' : 0, 'copy' : False, 'track' : False}

#change following IP address '192.168.x.xxx' with yours
client = NetGear(address = '192.168.1.17', port = '5454', protocol = 'tcp',  pattern = 0, receive_mode = True, logging = True, **options) #Define netgear client at Server IP address.

# infinite loop
while True:
	# receive frames from network
	frame = client.recv()

	# check if frame is None
	if frame is None:
		#if True break the infinite loop
		break

	# do something with frame here

	# Show output window
	cv2.imshow("Output Frame", frame)

	key = cv2.waitKey(1) & 0xFF
	# check for 'q' key-press
	if key == ord("q"):
		#if 'q' key-pressed break out
		break

# close output window
cv2.destroyAllWindows()
# safely close client
client.close()
