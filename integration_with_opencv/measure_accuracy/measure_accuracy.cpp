// yolo_v4_cuda.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
// https://gist.github.com/YashasSamaga/e2b19a6807a13046e399f4bc3cca3a49

#include <iostream>
#include <iomanip>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "Experiment.h"

using std::cout;
using std::setw;

int frame_counter = 0;

/*
	Маркер: 
	- кадров, где обнаружено меньше положенного
	- кадров, где обнаружено столько, сколько нужно
	- кадров, где обнаружено больше положенного объектов

	0 объектов, 1 объект, 2 объекта и более
*/

int marker_stats[3][3];

int dock_counter = 0;
int two_bcs = 0;
int two_wcs = 0;

cv::Mat frame_resized;

int main()
{
	Experiment myExp;

	cv::VideoCapture source(VIDEO_B_CIRCLE);
	//cv::VideoCapture source(0);
	
	while (1)
	{
		source >> myExp.frame;
		if (myExp.frame.empty())
		{
			cv::waitKey();
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


		for (int i = 0; i < NUM_CLASSES; i++) {
		
			// ноль объектов | ровно 1 объект | 2 и более
			
			if (myExp.indices[i].size() == 0) {
				marker_stats[i][0]++;
			}
			else if (myExp.indices[i].size() == 1) {
				marker_stats[i][1]++;
			}
			else {
				marker_stats[i][2]++;
			}
		}

		if (myExp.indices[0].size() == 1) {
			dock_counter++;
		}
		if (myExp.indices[1].size() == 2) {
			two_wcs++;
		}
		if (myExp.indices[2].size() == 2) {
			two_bcs++;
		}


		char c = cv::waitKey(1);

		if (c == 27) {
			break;
		}

	} // end while(1)



	cout << "Total frames: " << frame_counter << "\n";
	cout << "\tFound pairs of white circles: " << two_wcs << " " << float(two_wcs / frame_counter) << "\n";
	cout << "\tFound pairs of black circles: " << two_bcs << " " << float(two_bcs/frame_counter) << "\n";
	

	for (int i = 0; i < 3; i++) {
		for (int elem : marker_stats[i])
			cout << elem << " ";
		cout << "\n";
	}

	return 0;
}