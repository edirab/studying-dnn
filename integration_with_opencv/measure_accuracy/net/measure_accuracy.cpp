// yolo_v4_cuda.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
// https://gist.github.com/YashasSamaga/e2b19a6807a13046e399f4bc3cca3a49

#include <iostream>
#include <iomanip>
#include <clocale>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "Experiment.h"

using std::cout;
using std::setw;

int frame_counter = 0;

std::string file_path = VIDEO_W_CIRCLE;

/*
	Маркер: 
	- кадров, где обнаружено меньше положенного
	- кадров, где обнаружено столько, сколько нужно
	- кадров, где обнаружено больше положенного объектов

	0 объектов, 1 объект, 2 объекта и более
*/

#define MAX_OBJECTS 15
int marker_stats[3][MAX_OBJECTS];


cv::Mat frame_resized;

int main()
{
	std::setlocale(0, "");
	Experiment myExp;

	cv::VideoCapture source(file_path);
	//cv::VideoCapture source(0);
	
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	while (1)
	{
		source >> myExp.frame;
		if (myExp.frame.empty())
		{
			break;
		}


		myExp.Process();
		myExp.Draw();
		myExp.AnnotateStats();

		cv::namedWindow("output");
		cv::resize(myExp.frame, frame_resized, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
		cv::imshow("output", frame_resized);

		/*
			Frame #0
					dock:
						1 (751, 253, 451, 280)
					w.c.:
						1 (865, 500, 58, 64)
						4 (1045, 496, 51, 65)
						6 (1401, 313, 188, 99)
					b.c.:
						4 (1033, 320, 62, 49)
						1 (861, 330, 54, 50)

			Total frames: 915

		*/
		cout << "\nFrame #" << frame_counter++ << "\n";

		for (int i = 0; i < NUM_CLASSES; i++) {
			cout << setw(7) << myExp.class_names[i] << ": \n";

			for (int elem : myExp.indices[i]) {
				cout << "\t" << elem << " ";
				cv::Rect rect = myExp.boxes[i][elem];
				cout << "(" << rect.x << ", " << rect.y << ", " << rect.width << ", " << rect.height << ")\n";
			}
		}

		//
		for (int i = 0; i < NUM_CLASSES; i++) {
			if (myExp.indices[i].size() < MAX_OBJECTS) {
				marker_stats[i][ myExp.indices[i].size() ]++;
			}
			else {
				marker_stats[i][MAX_OBJECTS - 1]++;
			}
		}


		char c = cv::waitKey(1);
		if (c == 27) {
			break;
		}

	} // end while(1)
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	cout << "Total frames: " << frame_counter << " in " << file_path << "\n";

	std::cout << "\tTime difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
	std::cout << "\tTime difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
	std::cout << "\tTime difference = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl;

	/*
		Выводим индексы для красоты и наглядности
	*/
	cout << "objs: [";
	for (int i = 0; i < MAX_OBJECTS; i++)
		cout << setw(4) << i << ",";
	cout << " ]\n";

	cout << "dock: [";
	for (int elem : marker_stats[0])
		cout << setw(4) << elem << ",";
	cout << " ]\n";

	cout << "w.c.: [";
	for (int elem : marker_stats[1])
		cout << setw(4) << elem << ",";
	cout << " ]\n";

	cout << "b.c.: [";
	for (int elem : marker_stats[2])
		cout << setw(4) << elem << ",";
	cout << " ]\n";

	return 0;
}