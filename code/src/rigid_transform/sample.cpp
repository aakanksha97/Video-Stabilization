/* 

ref 	-> https://stackoverflow.com/questions/21622608/video-stabilization-using-opencv
	-> https://github.com/opencv/opencv/blob/master/samples/cpp/lkdemo.cpp

compile with this - g++ -I/usr/local/include/opencv -I/usr/local/include/opencv2 -L/usr/local/lib/ -g -o binary  sample.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_videoio -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_stitching

*/

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main()
{
	bool ret_stat;
	Mat prev_frame, cur_frame, prev_frame_gray, cur_frame_gray, trans_frame, trans_frame_gray, trans_mat;
	VideoCapture capture;

	capture.open(0);

	if( !capture.isOpened() )
	{
		cout << "cannot capture the video!\n";
		return 0;
	}
	
	capture >> prev_frame;
	cvtColor(prev_frame, prev_frame_gray, COLOR_BGR2GRAY);
	
	while(1)
	{
		capture >> cur_frame;
		cvtColor(cur_frame, cur_frame_gray, COLOR_BGR2GRAY);
		
		trans_mat = estimateRigidTransform(prev_frame_gray, cur_frame_gray, false);
		warpAffine(cur_frame, trans_frame, trans_mat, cur_frame.size(), WARP_INVERSE_MAP);

		imshow("Original footage", cur_frame);
		imshow("Stabilized footage", trans_frame);

		cvtColor(trans_frame, trans_frame_gray, COLOR_BGR2GRAY);
		trans_frame_gray.copyTo(prev_frame_gray);
		waitKey(40);
	}
	
	return 0;
}
