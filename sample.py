# Tutorial -> https://www.youtube.com/watch?v=Jvf5y21ZqtQ

import cv2

# VideoCapture(<0 for 1st Webcam, 1 for 2nd webcam and so on...>)
capture = cv2.VideoCapture(0)

if( capture.isOpened() == False ):
	print("Unable to capture the video!")
	exit(0)

while True:
	# a, b = read() where 'a' is a boolean; it is true if there is a frame; 'b' is a frame
	is_frame, frame = capture.read()

	if( is_frame == True ):	
		#gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cv2.imshow('Video', frame)
		if(cv2.waitKey(1) == ord('q')):
			break

capture.release()
cv2.destroyAllWindows()
