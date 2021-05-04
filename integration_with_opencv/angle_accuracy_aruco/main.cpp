
/*
	https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
*/

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
//#include <cstdlib>
#include <iomanip>
#include <clocale>
#include <string>
#include <vector>
#include <chrono>

#include "../angle_accuracy_net/Statistics.h"

#define VIDEO_BASE_PATH "E:/University/12sem/ВКРМ/Нейронки/angle/"

using namespace std;
using namespace std::chrono;
using namespace cv;

int real_angle = 0;

Mat cMatrixFullHD = (Mat_<double>(3, 3) << 1.8319233710098095e+04, 0, 9.5950000000000000e+02,
	0, 1.8319233710098095e+04, 5.3950000000000000e+02,
	0, 0, 1);

Mat distortionFullHD = (Mat_<double>(1, 5) << 2.5318626977076864e+01, -1.2398369018684898e+04, 0, 0, 2.4953338626272022e+06);

vector<string> videos = { "90.mp4", "85.mp4", "80.mp4", "75.mp4", "70.mp4", "65.mp4", "60.mp4", "55.mp4", "50.mp4", "45.mp4", "40.mp4", "35.mp4", "30.mp4" };

Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_ARUCO_ORIGINAL);

VideoCapture capture;

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


int do_alalysis(string vid) {

	vector<Vec3d> rvecs, tvecs;
	Vec3f Eul;
	vector<int> ids;
	vector<std::vector<cv::Point2f>> corners;

	Mat frame, frame_copy;
	Mat rotMat(3, 3, CV_64F);
	Mat Euler(3, 3, CV_64F);

	Statistics myStats;

	string path = VIDEO_BASE_PATH + vid;
	capture.open(path);

	if (!capture.isOpened()) {
		cout << "--(!)Error opening video capture\n";
		return -1;
	}

	while (1)
	{
		capture.read(frame);

		if (frame.empty()) {
			cout << "--(!) No captured frame -- Break!\n";
			break;
		}
		frame.copyTo(frame_copy);

		cv::aruco::detectMarkers(frame, dictionary, corners, ids);
		cv::aruco::drawDetectedMarkers(frame_copy, corners, ids);
		cv::aruco::estimatePoseSingleMarkers(corners, 29, cMatrixFullHD, distortionFullHD, rvecs, tvecs);

		//cout << rvecs << "\n";

		if (rvecs.size() > 0 && tvecs.size() > 0) {
			Rodrigues(rvecs[0], rotMat);
		}

		if (isRotationMatrix(rotMat)) {
			Eul = rotationMatrixToEulerAngles(rotMat);

			cout << setprecision(3);

			if (rvecs.size() != 0) {
				//cout << setw(7) << Eul[0]*180/3.1415926 << setw(7) << Eul[1] * 180 / 3.1415926 << setw(7) << Eul[2] * 180 / 3.1415926 << "\n";
				double m_angle = Eul[1] * 180 / M_PI;
				//cout << m_angle << "\n";
				myStats.add(m_angle);
			}
		}

		resize(frame_copy, frame_copy, Size(), 0.5, 0.5, cv::INTER_LINEAR);
		imshow("Marker", frame_copy);

		if (waitKey(1) == 27)
			break;
	}
	capture.release();
	cout << "\n  Real angle: " << real_angle << "\n";
	real_angle += 5;
	myStats.print_stats(path);
}




int main(int argc, char* argv[]) {

	setlocale(0, "");
	system("chcp 1251");

	auto t1 = high_resolution_clock::now();
	for (string single_video : videos) {
		do_alalysis(single_video);
	}
	auto t2 = high_resolution_clock::now();

	cout << "Total time in seconds: " << duration_cast<seconds>(t2 - t1).count() << "\n";
	capture.release();
	destroyAllWindows();
	return 0;
}

