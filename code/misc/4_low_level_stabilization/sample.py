import cv2
import numpy

def store_translated_image():
	org_img = cv2.imread("cameraman.jpg", 0)
	cv2.imshow("original", org_img)

	rows, cols = org_img.shape
	affine_matrix = numpy.float32([[1,0,100],[0,1,50]])
	translated_img = cv2.warpAffine(org_img,affine_matrix,(cols,rows))
	cv2.imwrite("cameraman_translated.jpg", translated_img)

def remove_missed_points(status, prev_points, cur_points):
        prev_points_filt = []
        cur_points_filt = []

        for cur_stat, cur_prev_points, cur_cur_points in zip(status, prev_points, cur_points):
                if(cur_stat[0] == 1):
                        prev_points_filt.append(cur_prev_points)
                        cur_points_filt.append(cur_cur_points)

        prev_points_filt = numpy.array(prev_points_filt)
        cur_points_filt = numpy.array(cur_points_filt)
        return prev_points_filt, cur_points_filt

'''
                        *       *       *
                        *       #       *
                        *       *       *
'''


def visualize_points(image, points):
	min_intensity = 0
	max_intensity = 255
	threshold = 127
	for item in points:
		(row, col) = (int(item[0][0]), int(item[0][1]))
		
		if(image[row-1][col-1] > threshold):
			image[row-1][col-1] = min_intensity
		else:
			image[row-1][col-1] = max_intensity

		if(image[row-1][col] > threshold):
                        image[row-1][col] = min_intensity
                else:
                        image[row-1][col] = max_intensity
		
		if(image[row-1][col+1] > threshold):
                        image[row-1][col+1] = min_intensity
                else:
                        image[row-1][col+1] = max_intensity
		
		if(image[row][col+1] > threshold):
                        image[row][col+1] = min_intensity
                else:
                        image[row][col+1] = max_intensity
		
		if(image[row+1][col+1] > threshold):
                        image[row+1][col+1] = min_intensity
                else:
                        image[row+1][col+1] = max_intensity	

		if(image[row+1][col] > threshold):
                        image[row+1][col] = min_intensity
                else:
                        image[row+1][col] = max_intensity
		
		if(image[row+1][col-1] > threshold):
                        image[row+1][col-1] = min_intensity
                else:
                        image[row+1][col-1] = max_intensity
		

		if(image[row][col-1] > threshold):
                        image[row][col-1] = min_intensity
                else:
                        image[row][col-1] = max_intensity		

		cv2.imwrite("cameraman_points.jpg", image)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


prev_frame = cv2.imread("cameraman.jpg", 0)
cur_frame = cv2.imread("cameraman_translated.jpg", 0)
prev_frame.astype(numpy.float32)
cur_frame.astype(numpy.float32)

# To detect corner points
prev_points = cv2.goodFeaturesToTrack(prev_frame, mask = None, **feature_params)
prev_points.astype(numpy.float32)

# Tracks the corner points
cur_points, status, error = cv2.calcOpticalFlowPyrLK(prev_frame, cur_frame, prev_points, None, **lk_params)

prev_points_updated, cur_points = remove_missed_points(status, prev_points, cur_points)
cur_points.astype(numpy.float32)
prev_points_updated.astype(numpy.float32)

visualize_points(prev_frame, prev_points_updated)
exit(0)

trans_mat = cv2.findHomography(cur_points, prev_points_updated, method=cv2.LMEDS)
trans_mat = trans_mat[0]

transformed_frame = numpy.empty(cur_frame.shape)
transformed_frame = cv2.warpPerspective(src=cur_frame, dst=transformed_frame, M=trans_mat, dsize=(cur_frame.shape[1], cur_frame.shape[0]))

cv2.imwrite("cameraman_predicted.jpg", transformed_frame)
