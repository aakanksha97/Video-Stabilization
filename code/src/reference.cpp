// compile with this - g++ -I/usr/local/include/opencv -I/usr/local/include/opencv2 -L/usr/local/lib/ -g -o binary  reference.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_videoio -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_stitching

#include "opencv2/videoio.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include <stdio.h>

using namespace cv;
using namespace std;

int main(int ac, char** av)
{	
	Mat src, prev, curr, rigid_mat, dst;
	VideoCapture cap(0);

	while (1)
	{
		bool bSuccess = cap.read(src);

		if (!bSuccess) //if not success, break loop
		{
			cout << "Cannot read the frame from video file" << endl;
			break;
		}

		cvtColor(src, curr, CV_BGR2GRAY);

		if (prev.empty())
		{
			prev = curr.clone();
		}

		rigid_mat = estimateRigidTransform(prev, curr, false);

		warpAffine(src, dst, rigid_mat, src.size(), INTER_NEAREST|WARP_INVERSE_MAP, BORDER_CONSTANT);


		// ---------------------------------------------------------------------------//

		imshow("input", src);
		imshow("output", dst);

		Mat dst_gray;
		cvtColor(dst, dst_gray, CV_BGR2GRAY);
		prev = dst_gray.clone();

		waitKey(30);
	}

	
	cv::destroyAllWindows();
	return 0;
}
