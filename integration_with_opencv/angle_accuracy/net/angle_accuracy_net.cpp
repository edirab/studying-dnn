// angle_accuracy : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//
#include <string>
#include <vector>
#include <clocale>
#include <chrono>
//#include "FPS.h"
#include "Marker.h"
#include "AUV.h"
#include "functions.h"
#include "Statistics.h"

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

//#define VIDEO_PATH "E:/University/10sem/nirs/haar_3_4_6/pyramid_test.mp4"
#define VIDEO_PATH "E:/University/12sem/ВКРМ/Нейронки/angle/75.mp4"
#define VIDEO_BASE_PATH "E:/University/12sem/ВКРМ/Нейронки/angle/for_yolo_and_cascade/"

using namespace std;
using namespace cv;
using namespace std::chrono;

int real_angle = 0;
AUV auv;
VideoCapture capture;

vector<string> videos = { "90.mp4", "85.mp4", "80.mp4", "75.mp4", "70.mp4", "65.mp4", "60.mp4", "55.mp4", "50.mp4", "45.mp4", "40.mp4", "35.mp4", "30.mp4" };


int do_alalysis(string vid) {

	Statistics myStats;
	Mat frame;
	string path = VIDEO_BASE_PATH + vid;
	capture.open(path);

	if (!capture.isOpened()) {
		cout << "--(!)Error opening video capture\n";
		return -1;
	}

	while (1) {
		capture.read(frame);

		if (frame.empty()) {
			break;
		}

		auv.get_orientation(frame);
		myStats.add(auv.get_Euler_1());

		//cerr << setprecision(3) << auv.get_Euler_1() << "\n";

		resize(frame, frame, Size(), 0.5, 0.5, cv::INTER_LINEAR);
		imshow("Orientation", frame);

		if (waitKey(1) == 27)
			break;

	}
	capture.release();

	cout << "\n  Real angle: " << real_angle << "\n";
	real_angle += 5;
	myStats.print_stats(path);
}

int main(int argc, const char** argv) {

	setlocale(0, "");
	system("chcp 1251");

	auto t1 = high_resolution_clock::now();
	//for (string single_video : videos) {
	//	do_alalysis(single_video);
	//}
	do_alalysis("30.mp4");
	auto t2 = high_resolution_clock::now();

	cout << "Total time in seconds: " << duration_cast<seconds>(t2 - t1).count() << "\n";
	destroyAllWindows();
	return 0;
}
