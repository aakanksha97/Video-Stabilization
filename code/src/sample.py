#Tutorial -> https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html

import cv2
import numpy

capture = cv2.VideoCapture("shaky.mp4")


# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

ret_val_prev, prev_frame = capture.read()
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# To detect corner points
prev_points = cv2.goodFeaturesToTrack(prev_frame_gray, mask = None, **feature_params)

while( capture.isOpened() == True ):
	ret_val_cur, cur_frame = capture.read()
	if( ret_val_cur == True ):
		cur_frame_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

		# Tracks the corner points
		print("prev points : \n" + str(prev_points) + "\n-----------------------------\n")
		prev_points.astype(numpy.float32)
		cur_points, status, error = cv2.calcOpticalFlowPyrLK(prev_frame_gray, cur_frame_gray, prev_points, None, **lk_params)

		# Considering those points which are found in the cur_frame i.e status = 1
		#prev_filt_points = prev_points[status==1]
		#cur_filt_points = cur_points[status==1]

		# 3x3 Transformation matrix
		#trans_mat = cv2.findHomography(prev_filt_points,cur_filt_points)
		trans_mat = cv2.findHomography(prev_points,cur_points)

		print( "status : \n" + str(status) + "\n---------------------------" )

		if(cv2.waitKey(1) == ord('q')):
			break
	
		# Update for next iteration
		prev_frame = cur_frame.copy()
		prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

		# To detect corner points
		prev_points = cv2.goodFeaturesToTrack(prev_frame_gray, mask = None, **feature_params)

		if(prev_points == None):
			break
	else:
		break

cv2.destroyAllWindows()
capture.release()

