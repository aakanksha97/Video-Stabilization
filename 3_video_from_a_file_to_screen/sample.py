# Tutorial -> https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html#playing-video-from-file

import cv2

capture = cv2.VideoCapture("input.avi")

while(capture.isOpened() == True):
	# a, b = read() where 'a' is a boolean; it is true if there is a frame; 'b' is a frame
	is_frame, frame = capture.read()

	if( is_frame == True ):	
		#gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cv2.imshow('footage from file', frame)
		if(cv2.waitKey(1) == ord('q')):
			break
	else:
		break

capture.release()
cv2.destroyAllWindows()
