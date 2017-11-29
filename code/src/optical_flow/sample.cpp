// ref -> https://stackoverflow.com/questions/21622608/video-stabilization-using-opencv

// compile with this - g++ -I/usr/local/include/opencv -I/usr/local/include/opencv2 -L/usr/local/lib/ -g -o binary  sample.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_videoio -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_stitching

#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include <stdio.h>
#include <ctype.h>

using namespace cv;
using namespace std;

Mat points_prev, points_cur, status;

void remove_missed_points()
{
	vector<int> status_vect, points_prev_vect, points_cur_vect;	
	int cur_index = 0, new_index = 0;
	int tot_points = points_prev.rows;
	vector<int> points_prev_new_vect, points_cur_new_vect;
	Mat points_prev_new, points_cur_new;

	status_vect.assign(status.datastart, status.dataend);
	points_prev_vect.assign(points_prev.datastart, points_prev.dataend);
	points_cur_vect.assign(points_cur.datastart, points_cur.dataend);

	while( cur_index < tot_points )
	{	
		if( status_vect[cur_index] == 1 )		
		{
			cout << "-------------------------------------------------------------------\n";
			points_prev_new_vect[new_index] = points_prev_vect[cur_index];
			points_cur_new_vect[new_index] = points_cur_vect[cur_index]; 
			new_index = new_index + 1;
		}

		cur_index = cur_index + 1;
	}

	
	
	cout << "points_prev_new_vect size : " << points_prev_new_vect.size() << endl;
	cout << "points_prev_new size : " << points_prev_new.size()  << endl;
	
	points_prev = points_prev_new.clone();
	points_cur = points_cur_new.clone();
}

/*        for cur_stat, cur_prev_points, cur_cur_points in zip(status, prev_points, cur_points):
                if(cur_stat[0] == 1):
                        prev_points_filt.append(cur_prev_points)
                        cur_points_filt.append(cur_cur_points)

        prev_points_filt = numpy.array(prev_points_filt)
        cur_points_filt = numpy.array(cur_points_filt)
        return prev_points_filt, cur_points_filt
*/
void start()
{
	bool ret_stat;
	Mat prev_frame, cur_frame, prev_frame_gray, cur_frame_gray, trans_frame, trans_frame_gray, trans_mat;
	VideoCapture capture;
	Size winSize(31,31);
	vector<float> error;
	TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);

	capture.open(0);

	if( !capture.isOpened() )
	{
		cout << "cannot capture the video!\n";
		return;
	}
	
	capture >> prev_frame;
	cvtColor(prev_frame, prev_frame_gray, COLOR_BGR2GRAY);
	
	while(1)
	{
		capture >> cur_frame;

		if( cur_frame.empty() )
		{
			cout << "empty frame!" << endl;
			break;
		}
	
		cvtColor(cur_frame, cur_frame_gray, COLOR_BGR2GRAY);
		
		goodFeaturesToTrack(prev_frame_gray, points_prev, 500, 0.01, 10, Mat(), 3, 3, 0, 0.04);
		calcOpticalFlowPyrLK(prev_frame_gray, cur_frame_gray, points_prev, points_cur, status, error, winSize, 3, termcrit, 0, 0.001);
		cout << "Before removal : " << points_prev.size() << endl;
		remove_missed_points();
		cout << "After removal : " << points_prev.size() << endl;
		trans_mat = findHomography(points_prev, points_cur);
		warpPerspective(cur_frame, trans_frame, trans_mat, cur_frame.size(), INTER_NEAREST|WARP_INVERSE_MAP, BORDER_CONSTANT);

		imshow("Original footage", cur_frame);
		imshow("Stabilized footage", trans_frame);

		cvtColor(trans_frame, trans_frame_gray, COLOR_BGR2GRAY);
		prev_frame_gray = trans_frame_gray.clone();
		waitKey(40);
	}
}

int main()
{
	start();
	return 0;
}
