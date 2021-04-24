// yolo_v4_cuda.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
// https://gist.github.com/YashasSamaga/e2b19a6807a13046e399f4bc3cca3a49

#include <iostream>
#include <iomanip>
#include <chrono>

#include <opencv2/core.hpp>
#include "opencv2/objdetect.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


using std::cout;
using std::setw;
using std::vector;

using cv::Mat;
using cv::Rect;
using cv::Scalar;

int frame_counter = 0;

/*
	- dock
	- w.c. marker2
	- b.c. marker1

	Маркер:
	- кадров, где обнаружено меньше положенного
	- кадров, где обнаружено столько, сколько нужно
	- кадров, где обнаружено больше положенного объектов

	0 объектов, 1 объект, 2 объекта и более
*/

#define MAX_OBJECTS 15
int marker_stats[2][MAX_OBJECTS];

cv::Mat frame, frame_gray, frame_resized;

#define PNK Scalar(255, 0, 255)
#define YEL Scalar(0, 255, 255)

#define VIDEO_W_CIRCLE "D:/studying-dnn/integration_with_opencv/measure_accuracy/white_circle.mp4"
#define VIDEO_B_CIRCLE "D:/studying-dnn/integration_with_opencv/measure_accuracy/black_circle.mp4"

#define bc_cascade_path "E:/University/10sem/nirs/haar_3_4_6/preparing navigation/haar_navigation_m1_v3/cascade.xml"
#define wc_cascade_path "E:/University/10sem/nirs/haar_3_4_6/preparing navigation/haar_navigation_m2_v1/cascade.xml"


/*
	Отобразить объекты из вектора на изображении
*/
void draw_objects(Mat& frame, vector<Rect> objects, cv::Scalar color) {

	for (size_t i = 0; i < objects.size(); i++) {
		rectangle(frame, objects[i], color);
	}
}


int main()
{

	cv::CascadeClassifier marker_bc_cascade, marker_wc_cascade;

	//-- 1. Load the cascades
	if (!marker_bc_cascade.load(bc_cascade_path)) {
		cout << "--(!)Error loading first cascade\n";
	}
	else if (!marker_wc_cascade.load(wc_cascade_path)) {
		cout << "--(!)Error loading second cascade\n";
	}


	cv::VideoCapture source(VIDEO_W_CIRCLE);
	//cv::VideoCapture source(0);

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	while (1)
	{
		source >> frame;
		if (frame.empty())
		{
			//cv::waitKey();
			break;
		}


		cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
		equalizeHist(frame_gray, frame_gray);

		vector<Rect> markers_bc, markers_wc;

		marker_bc_cascade.detectMultiScale(frame_gray, markers_bc);
		marker_wc_cascade.detectMultiScale(frame_gray, markers_wc);

		draw_objects(frame, markers_bc, YEL);
		draw_objects(frame, markers_wc, PNK);


		cv::namedWindow("output");
		cv::resize(frame, frame_resized, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
		cv::imshow("output", frame_resized);

		/*
		Total frames: 900

			0 0 0
			383 293 224
			1 66 833
		*/
		cout << "Frame #" << frame_counter++ << " " << markers_bc.size() << " " << markers_wc.size() << "\n";

		if (markers_wc.size() < MAX_OBJECTS) {
			marker_stats[0][markers_wc.size()]++;
		}
		else {
			marker_stats[0][MAX_OBJECTS - 1]++;
		}

		if (markers_bc.size() < MAX_OBJECTS) {
			marker_stats[1][markers_bc.size()]++;
		}
		else {
			marker_stats[1][MAX_OBJECTS - 1]++;
		}

		//if (markers_wc.size() == 0) {
		//	marker_stats[1][0]++;
		//} else if (markers_wc.size() == 1) {
		//	marker_stats[1][1]++;
		//}
		//else {
		//	marker_stats[1][2]++;
		//}

		//if (markers_bc.size() == 0) {
		//	marker_stats[2][0]++;
		//}
		//else if (markers_bc.size() == 1) {
		//	marker_stats[2][1]++;
		//}
		//else {
		//	marker_stats[2][2]++;
		//}

		char c = cv::waitKey(1);
		if (c == 27) {
			break;
		}
	} // end while(1)

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	cout << "Total frames: " << frame_counter << "\n";

	std::cout << "\tTime difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
	std::cout << "\tTime difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
	std::cout << "\tTime difference = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl;

	/* 
		Выводим индексы для красоты и наглядности
	
	*/
	cout << "objs: ";
	for (int i = 0; i < MAX_OBJECTS; i++)
		cout << setw(4) << i;
	cout << "\n";

	cout << "w.c.: ";
	for (int elem : marker_stats[0])
		cout << setw(4) << elem;
	cout << "\n";

	cout << "b.c.: ";
	for (int elem : marker_stats[1])
		cout << setw(4) << elem;

	//for (int i = 0; i < 2; i++) {
	//	for (int elem : marker_stats[i])
	//		cout << elem << " ";
	//	cout << "\n";
	//}

	return 0;
}