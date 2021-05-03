// angle_accuracy : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//
#include <string>
#include <vector>
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

using namespace std;
using namespace cv;

Statistics myStats;
AUV auv;
VideoCapture capture;

vector<string> videos = { "90.mp4", "85.mp4", "80.mp4", "75.mp4", "70.mp4", "65.mp4", "60.mp4", "55.mp4", "50.mp4", "45.mp4", "40.mp4", "35.mp4", "30.mp4" };

int main(int argc, const char** argv) {

	Mat frame;
	string path = VIDEO_PATH;
	capture.open(path);

	if (!capture.isOpened()) {
		cout << "--(!)Error opening video capture\n";
		return -1;
	}

	while (1) {
		capture.read(frame);

		if (frame.empty()) {
			cout << "--(!) No captured frame -- Break!\n";
			break;
		}

		auv.get_orientation(frame);
		myStats.add(auv.get_Euler_1());

		cout << setprecision(3) << auv.get_Euler_1() << "\n";

		resize(frame, frame, Size(), 0.5, 0.5, cv::INTER_LINEAR);
		imshow("Orientation", frame);

		if (waitKey(1) == 27)
			break;

	}
	capture.release();
	destroyAllWindows();

	myStats.print_stats(path);
	return 0;
}
