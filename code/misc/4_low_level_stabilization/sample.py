import cv2
import numpy
import time

min_intensity = 0
max_intensity = 255

def store_translated_image(filename):
	org_img = cv2.imread(filename, 0)

	rows, cols = org_img.shape
	affine_matrix = numpy.float32([[1,0,100],[0,1,50]])
	translated_img = cv2.warpAffine(org_img,affine_matrix,(cols,rows))
	cv2.imwrite("translated.jpg", translated_img)

def reverse_translate():
	org_img = cv2.imread("translated.jpg", 0)
	rows, cols = org_img.shape
	affine_matrix = numpy.float32([[1, 0, -100], [0, 1, -50]])
	rev_translated_image = cv2.warpAffine(org_img,affine_matrix,(cols,rows))
	cv2.imwrite("rev_translated.jpg", rev_translated_image)

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
def visualize_points(image, points, filename):
	threshold = 127
	(rows, cols) = image.shape
	for item in points:
		(row, col) = (int(item[0][0]), int(item[0][1]))
	
		if(row < 0 or row > (rows-1) or col < 0 or col > (cols-1)):
			continue
	
		if((row-1) >= 0 and (col-1) >= 0 and image[row-1][col-1] > threshold ):
			image[row-1][col-1] = min_intensity
		elif((row-1) >= 0 and (col-1) >= 0):
			image[row-1][col-1] = max_intensity

		if((row-1) >= 0 and image[row-1][col] > threshold ):
                        image[row-1][col] = min_intensity
                elif( (row-1) >= 0):
                        image[row-1][col] = max_intensity
		
		if((row-1) >= 0 and (col+1) <= (cols-1) and image[row-1][col+1] > threshold ):
                        image[row-1][col+1] = min_intensity
                elif((row-1) >= 0 and (col+1) <= (cols-1)):
                        image[row-1][col+1] = max_intensity
		
		if((col+1) <= (cols-1) and image[row][col+1] > threshold ):
                        image[row][col+1] = min_intensity
                elif((col+1) <= (cols-1)):
                        image[row][col+1] = max_intensity
		
		if( (row+1) <= (rows-1) and (col+1) <= (cols-1) and image[row+1][col+1] > threshold):
                        image[row+1][col+1] = min_intensity
                elif((row+1) <= (rows-1) and (col+1) <= (cols-1)):
                        image[row+1][col+1] = max_intensity	

		if((row+1) <= (rows-1) and image[row+1][col] > threshold):
                        image[row+1][col] = min_intensity
                elif((row+1) <= (rows-1)):
                        image[row+1][col] = max_intensity
		
		if((row+1) <= (rows-1) and (col-1) >= 0 and image[row+1][col-1] > threshold):
                        image[row+1][col-1] = min_intensity
                elif((row+1) <= (rows-1) and (col-1) >= 0):
                        image[row+1][col-1] = max_intensity
		

		if((col-1) >= 0 and image[row][col-1] > threshold):
                        image[row][col-1] = min_intensity
                elif((col-1) >= 0):
                        image[row][col-1] = max_intensity		

	cv2.imwrite(filename, image)


def detect_edges(img_org, filename):
	img_edg = cv2.Canny(img_org, 100, 200)
	cv2.imwrite(filename, img_edg)
	return img_edg

def get_edge_points(img_org, filename):
	img_edg = detect_edges(img_org, "b_w_" + filename)
	edge_points = []
	row_index = 0
	for row_array in img_edg:
		col_index = 0
		for pixel in row_array:
			if(pixel == max_intensity):
				edge_points.append([[row_index, col_index]])
			col_index = col_index + 1
		row_index = row_index + 1
	visualize_points(img_org.copy(), edge_points, filename)
	return numpy.array(edge_points)	

def main():
	store_translated_image("cameraman.jpg")
	reverse_translate()
	exit(0)	
	
	# params for ShiTomasi corner detection
	feature_params = dict( 	maxCorners = 100,
				qualityLevel = 0.3,
				minDistance = 7,
				blockSize = 7 )
	
	# Parameters for lucas kanade optical flow
	lk_params = dict( 	winSize  = (15,15),
				maxLevel = 2,
				criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

	prev_frame = cv2.imread("cameraman.jpg", 0)
	cur_frame = cv2.imread("translated.jpg", 0)
	
	prev_points = get_edge_points(prev_frame.copy(), "points_prev_frame.jpg")
	cur_points = get_edge_points(cur_frame.copy(), "points_cur_frame.jpg")
	prev_points = prev_points.astype(numpy.float32)
	cur_points = cur_points.astype(numpy.float32)	
	
	print("prev_points : " + str(prev_points.shape))
	print("cur_points : " + str(cur_points.shape))

	tot_prev_points = prev_points.shape[0]
	tot_cur_points = cur_points.shape[0]
	min_points = min(tot_prev_points, tot_cur_points)

	if(tot_prev_points != min_points):
		diff = tot_prev_points - min_points
		start_index = int(diff / 2)
		prev_points = prev_points[start_index:(start_index+min_points)]
	elif(total_cur_points != min_points):
		diff = tot_cur_points - min_points
		start_index = int(diff/2)
		cur_points = cur_points[start_index:(start_index+min_points)]

	print("rev_prev_points : " + str(prev_points.shape))
        print("rev_cur_points : " + str(cur_points.shape))

	p0 = cv2.goodFeaturesToTrack(prev_frame, mask = None, **feature_params)
	p1, status, err = cv2.calcOpticalFlowPyrLK(prev_frame, cur_frame, p0, None, **lk_params)
	
	visualize_points(prev_frame.copy(), p0, "optical_flow_prev_frame.jpg")
	visualize_points(cur_frame.copy(), p1, "optical_flow_cur_frame.jpg")

	p0, p1 = remove_missed_points(status, p0, p1)

	trans_mat = cv2.findHomography(p1, p0)
	trans_mat = trans_mat[0]

	transformed_frame = numpy.empty(cur_frame.shape)
	transformed_frame = cv2.warpPerspective(src=cur_frame, dst=transformed_frame, M=trans_mat, dsize=(cur_frame.shape[1], cur_frame.shape[0]))

	cv2.imwrite("predicted.jpg", transformed_frame)

start_time = time.time()
main()
end_time = time.time()
elapsed_time = end_time-start_time
print("Processed in " + str(elapsed_time) + " seconds!")
