
/*
	https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
*/

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <cstdlib>
#include <iomanip>


using namespace std;
using namespace cv;

Mat cMatrix640 = (Mat_<double>(3, 3) << 5.3226273868525448e+02, 0, 3.2590522394049350e+02, 0, 5.3226273868525448e+02, 2.6946997900677803e+02, 0, 0, 1);
Mat distortion640 = (Mat_<double>(1, 5) << 0, -6.1539772782054671e-02, 0, 0, 1.7618036793466491e-02);

//Mat rvecs, tvecs;
vector<Vec3d> rvecs, tvecs;

Mat frame;
Mat frame_copy;
VideoCapture capture;

vector<int> ids;
vector<std::vector<cv::Point2f>> corners;

Mat rotMat(3, 3, CV_64F);
Mat Euler(3, 3, CV_64F);

Vec3f Eul;

// Checks if a matrix is a valid rotation matrix.
bool isRotationMatrix(Mat& R) {

	Mat Rt;
	transpose(R, Rt);
	Mat shouldBeIdentity = Rt * R;
	Mat I = Mat::eye(3, 3, shouldBeIdentity.type());

	return  norm(I, shouldBeIdentity) < 1e-6;

}

// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
Vec3f rotationMatrixToEulerAngles(Mat& R) {

	assert(isRotationMatrix(R));

	float sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));

	bool singular = sy < 1e-6; // If

	float x, y, z;
	if (!singular)
	{
		x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
		y = atan2(-R.at<double>(2, 0), sy);
		z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
	}
	else
	{
		x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
		y = atan2(-R.at<double>(2, 0), sy);
		z = 0;
	}
	return Vec3f(x, y, z);
}


int main(int argc, char* argv[]) {

	Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_ARUCO_ORIGINAL);

	capture.open(0);

	if (!capture.isOpened()) {
		cout << "--(!)Error opening video capture\n";
		return -1;
	}

	while(1) 
	{
		capture.read(frame);

		if (frame.empty()) {
			cout << "--(!) No captured frame -- Break!\n";
			break;
		}
		frame.copyTo(frame_copy);

		cv::aruco::detectMarkers(frame, dictionary, corners, ids);
		cv::aruco::drawDetectedMarkers(frame_copy, corners, ids);

		cv::aruco::estimatePoseSingleMarkers(corners, 29, cMatrix640, distortion640, rvecs, tvecs);

		//cout << rvecs << "\n";

		if (rvecs.size() > 0 && tvecs.size() > 0) {
			Rodrigues(rvecs[0], rotMat);
		}

		if (isRotationMatrix(rotMat)) {
			Eul = rotationMatrixToEulerAngles(rotMat);

			cout << setprecision(3);

			if (rvecs.size() != 0) {
				//cout << setw(7) << Eul[0]*180/3.1415926 << setw(7) << Eul[1] * 180 / 3.1415926 << setw(7) << Eul[2] * 180 / 3.1415926 << "\n";
				cout << Eul[1] * 180 / 3.1415926 << "\n";
			}
		}

		imshow("Marker", frame_copy);

		if (waitKey(25) == 27)
			break;

	}
	capture.release();
	destroyAllWindows();
	return 0;
}

