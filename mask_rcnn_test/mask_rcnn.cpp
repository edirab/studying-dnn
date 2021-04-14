// Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
// https://learnopencv.com/deep-learning-based-object-detection-and-instance-segmentation-using-mask-r-cnn-in-opencv-python-c/

#include <iostream>

#include <iostream>
#include <queue>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <istream>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using std::vector;
using std::string;
using std::ifstream;

using cv::Mat;
using cv::Rect;
using cv::Point;
using cv::String;
using cv::Size;
using cv::Scalar;

// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float maskThreshold = 0.3; // Mask threshold

// colors for bounding boxes
vector<Scalar> colors;
vector<string> classes;


// Draw the predicted bounding box, colorize and show the mask on the image
void drawBox(Mat& frame, int classId, float conf, Rect box, Mat& objectMask)
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), Scalar(255, 178, 50), 3);

	//Get the label for the class name and its confidence
	string label = cv::format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;
	}

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	box.y = std::max(box.y, labelSize.height);
	rectangle(frame, Point(box.x, box.y - round(1.5 * labelSize.height)), Point(box.x + round(1.5 * labelSize.width), box.y + baseLine), Scalar(255, 255, 255), cv::FILLED);
	putText(frame, label, Point(box.x, box.y), cv::FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);

	Scalar color = colors[classId % colors.size()];
	// Comment the above line and uncomment the two lines below to generate different instance colors
	//int colorInd = rand() % colors.size();
	//Scalar color = colors[colorInd];


	/* 
		Resize the mask, threshold, color and apply it on the image
		Где-то в этих 4-х строках ниже проблема
	*/
	resize(objectMask, objectMask, Size(box.width, box.height));
	Mat mask = (objectMask > maskThreshold);

	std::cout << "w & h: " << box.width << " " << box.height << " x & y:" << box.x << " " << box.y << 
		" SuM: " << box.x + box.width << " " << box.y + box.height << "\n";

	// Вот здесь выходит ошибка
	Mat tempMat = frame(box);

	try {

		if (box.empty()) {
			std::cout << "box is empty \n";
		}

		Mat coloredRoi = (0.3 * color + 0.7 * frame(box));

		if (coloredRoi.empty()) {
			std::cout << "coloredRoi is empty \n";
		}

		coloredRoi.convertTo(coloredRoi, CV_8UC3);
		// Draw the contours on the image
		vector<Mat> contours;
		Mat hierarchy;
		mask.convertTo(mask, CV_8U);
		findContours(mask, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
		drawContours(coloredRoi, contours, -1, color, 5, cv::LINE_8, hierarchy, 100);

		if (!mask.empty() && !box.empty()) {
			coloredRoi.copyTo(frame(box), mask);
		}

	}
	catch(const cv::Exception & ex) {
		std::cout << "Exception in lower part \n";
	}
	//cv::imshow("Inside drawBox", coloredRoi);
}


// For each frame, extract the bounding box and mask for each detected object
void postprocess(Mat& frame, const vector<Mat>& outs)
{
	Mat outDetections = outs[0];
	Mat outMasks = outs[1];

	// Output size of masks is NxCxHxW where
	// N - number of detected boxes
	// C - number of classes (excluding background)
	// HxW - segmentation shape
	const int numDetections = outDetections.size[2];
	const int numClasses = outMasks.size[1];

	outDetections = outDetections.reshape(1, outDetections.total() / 7);
	for (int i = 0; i < numDetections; ++i)
	{
		float score = outDetections.at<float>(i, 2);
		if (score > confThreshold)
		{
			// Extract the bounding box
			int classId = static_cast<int>(outDetections.at<float>(i, 1));
			int left = static_cast<int>(frame.cols * outDetections.at<float>(i, 3));
			int top = static_cast<int>(frame.rows * outDetections.at<float>(i, 4));
			int right = static_cast<int>(frame.cols * outDetections.at<float>(i, 5));
			int bottom = static_cast<int>(frame.rows * outDetections.at<float>(i, 6));

			left = std::max(0, std::min(left, frame.cols - 1));
			top = std::max(0, std::min(top, frame.rows - 1));
			right = std::max(0, std::min(right, frame.cols - 1));
			bottom = std::max(0, std::min(bottom, frame.rows - 1));
			Rect box = Rect(left, top, right - left + 1, bottom - top + 1);

			// Extract the mask for the object
			Mat objectMask(outMasks.size[2], outMasks.size[3], CV_32F, outMasks.ptr<float>(i, classId));

			// Draw bounding box, colorize and show the mask on the image

			try {
				drawBox(frame, classId, score, box, objectMask);
			}
			catch (cv::Exception & e) {
				std::cout << "Exception while calling drawBox func \n";
			}

		}
	}

	//cv::imshow("Frame", frame);
}


int main()
{

	// Load names of classes
	string classesFile = "D:/ML/mask_rcnn/mask-rcnn-coco/object_detection_classes_coco.txt";
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) classes.push_back(line);

	// Load the colors
	string colorsFile = "D:/ML/mask_rcnn/mask-rcnn-coco/colors.txt";
	ifstream colorFptr(colorsFile.c_str());
	while (getline(colorFptr, line)) {
		char* pEnd;
		double r, g, b;
		r = strtod(line.c_str(), &pEnd);
		g = strtod(pEnd, NULL);
		b = strtod(pEnd, NULL);
		colors.push_back(Scalar(r, g, b, 255.0));
	}

	//cv::VideoCapture source("demo.mp4");
	cv::VideoCapture source(0);

	//auto net = cv::dnn::readNetFromDarknet("E:/University/12sem/yolov4.cfg", "E:/University/12sem/yolov4.weights");
	auto net = cv::dnn::readNetFromTensorflow("D:/ML/mask_rcnn/mask-rcnn-coco/frozen_inference_graph.pb", 
												"D:/ML/mask_rcnn/mask-rcnn-coco/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt");
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	 //net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	 //net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	auto output_names = net.getUnconnectedOutLayersNames();

	cv::Mat frame, blob;
	std::vector<cv::Mat> detections;
	while (cv::waitKey(1) < 1)
	{
		source >> frame;
		if (frame.empty())
		{
			cv::waitKey();
			break;
		}

		auto total_start = std::chrono::steady_clock::now();

		cv::dnn::blobFromImage(frame, blob, 1.0, Size(frame.cols, frame.rows), Scalar(), true, false);
		net.setInput(blob);


		std::vector<cv::String> outNames(2);
		outNames[0] = "detection_out_final";
		outNames[1] = "detection_masks";
		vector<cv::Mat> outs;
		

		auto dnn_start = std::chrono::steady_clock::now();
		net.forward(outs, outNames);
		auto dnn_end = std::chrono::steady_clock::now();

		// Extract the bounding box and mask for each of the detected objects
		//try {
			postprocess(frame, outs);
		//}
		//catch (const cv::Exception & ex) {
		//	std::cout << "Exception while calling postprocess func \n";
		//}

		// Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
		vector<double> layersTimes;
		double freq = cv::getTickFrequency() / 1000;
		double t = net.getPerfProfile(layersTimes) / freq;
		string label = cv::format("Mask-RCNN : Inference time for a frame : %.2f ms", t);
		putText(frame, label, Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));

		// Write the frame with the detection boxes
		Mat detectedFrame;
		frame.convertTo(detectedFrame, CV_8U);

		imshow("Mask R-CNN Inference", frame);
	}

	return 0;
}