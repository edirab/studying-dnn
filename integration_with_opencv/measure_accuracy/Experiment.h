#pragma once

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


#define CLASSES "D:/studying-dnn/neural-nets/yolo_v4/docking_station/data/classes.names"
#define WEIGHTS "D:/studying-dnn/neural-nets/yolo_v4/docking_station/backup/yolov4_custom_train_best.weights"
#define CFG_FILE "D:/studying-dnn/neural-nets/yolo_v4/docking_station/cfg/yolov4_custom_test.cfg"

#define VIDEO_WITH "D:/studying-dnn/integration_with_opencv/measure_accuracy/with.mp4"
#define VIDEO_WITHOUT "D:/studying-dnn/integration_with_opencv/measure_accuracy/without.mp4"

#define VIDEO_W_CIRCLE "D:/studying-dnn/integration_with_opencv/measure_accuracy/white_circle.mp4"
#define VIDEO_B_CIRCLE "D:/studying-dnn/integration_with_opencv/measure_accuracy/black_circle.mp4"

constexpr float CONFIDENCE_THRESHOLD = 0.8;
constexpr float NMS_THRESHOLD = 0.05;
constexpr int NUM_CLASSES = 3;

// colors for bounding boxes
const cv::Scalar colors[] = {
	{0, 255, 255}, // YELLOW
	{255, 255, 0}, // CYAN
	{0, 255, 0}, // GREEN
	{255, 0, 255} // PINK
};

const auto NUM_COLORS = sizeof(colors) / sizeof(colors[0]);

class Experiment
{


public:

	std::vector<std::string> class_names;

	/*
		Это массив (фиксированного размера) векторов (!)
	*/
	std::vector<int> indices[NUM_CLASSES];
	std::vector<cv::Rect> boxes[NUM_CLASSES];
	std::vector<float> scores[NUM_CLASSES];

	std::vector<cv::Mat> detections;
	std::vector<cv::String> output_names;

	std::chrono::steady_clock::time_point total_start;
	std::chrono::steady_clock::time_point total_end;
	std::chrono::steady_clock::time_point dnn_start;
	std::chrono::steady_clock::time_point dnn_end;


	cv::Mat frame, blob;

	Experiment();

	void Draw();
	void Process();
	void AnnotateStats();

	void HoughValid();

private:

	cv::dnn::Net net;


	void clearArrays();
	void readClassNames();
};

