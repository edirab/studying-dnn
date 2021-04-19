// yolo_v4_cuda.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
// https://gist.github.com/YashasSamaga/e2b19a6807a13046e399f4bc3cca3a49

#include <iostream>

#include <iostream>
#include <queue>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "Experiment.h"



int main()
{
	Experiment myExp;

	//cv::VideoCapture source("demo.mp4");
	cv::VideoCapture source(0);
	
	while (cv::waitKey(1) < 1)
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
		cv::imshow("output", myExp.frame);
	}
	return 0;
}