// angle_accuracy : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

//#include "FPS.h"
#include "Marker.h"
#include "AUV.h"
#include "functions.h"


#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

//#define VIDEO_PATH "E:/University/10sem/nirs/haar_3_4_6/pyramid_test.mp4"
#define VIDEO_PATH "E:/University/12sem/ВКРМ/Нейронки/angle/85.mp4"

using namespace std;
using namespace cv;

int false_positive_counter = 0;

int main(int argc, const char** argv) {

	bool use_video = false;
	bool debug_on_image = false;

	int camera_device = 0;
	int frameno = 0;
	Mat frame;

	VideoCapture capture;

	/*
		Инициализация камеры
	*/
	if (!use_video && !debug_on_image) {
		capture.open(camera_device);
	}
	else if (!debug_on_image) {
		capture.open(VIDEO_PATH);
	}

	if (!capture.isOpened() && !debug_on_image) {
		cout << "--(!)Error opening video capture\n";
		return -1;
	}

	Size S = Size((int)capture.get(CAP_PROP_FRAME_WIDTH), (int)capture.get(CAP_PROP_FRAME_HEIGHT));

	//string abs_path = "E:/University/10sem/nirs/haar_3_4_6/preparing navigation/videos/pyramid_test_demo.mp4";
	//VideoWriter video(abs_path, CV_FOURCC('M', 'J', 'P', 'G'), 30, Size(1280, 720));
	//VideoWriter video(abs_path, CV_FOURCC('M', 'P', '4', 'V'), 30, Size(1280, 720));

	if (debug_on_image) {
		//frame = imread("E:/University/10sem/nirs/haar_3_4_6/preparing navigation/test/2.jpg");
		frame = imread("E:/University/10sem/nirs/haar_3_4_6/comparioson/09.jpg");
	}

	AUV auv;

	do {

		if (!debug_on_image) {
			capture.read(frame);
			//frame.copyTo(frame_2);
		}

		if (frame.empty()) {
			cout << "--(!) No captured frame -- Break!\n";
			break;
		}

		auv.get_orientation(frame);

		resize(frame, frame, Size(), 0.5, 0.5, cv::INTER_LINEAR);
		imshow("Orientation", frame);

		if (waitKey(1) == 27)
			break;

	} while (!debug_on_image);

	if (debug_on_image)
		waitKey(0);

	capture.release();
	//video.release();

	destroyAllWindows();
	return 0;
}
