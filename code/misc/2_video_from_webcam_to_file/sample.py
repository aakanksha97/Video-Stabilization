# Tutorial -> https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html#saving-a-video

import cv2

# VideoCapture(<0 for 1st Webcam, 1 for 2nd webcam and so on...>)
capture = cv2.VideoCapture(0)
codec = cv2.VideoWriter_fourcc(*'XVID')
out_vid = cv2.VideoWriter('output.avi',codec, 20.0, (640,480))

while(capture.isOpened() == True):
	# a, b = read() where 'a' is a boolean; it is true if there is a frame; 'b' is a frame
	is_frame, frame = capture.read()

	if( is_frame == True ):
		#gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cv2.imshow('Webcam_footage', frame)
		out_vid.write(frame)
		if(cv2.waitKey(1) == ord('q')):
			break
	else:
		break

capture.release()
out_vid.release()
cv2.destroyAllWindows()

