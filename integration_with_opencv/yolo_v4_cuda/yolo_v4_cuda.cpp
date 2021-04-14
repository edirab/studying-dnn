/* yolo_v4_cuda.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.

	ЭТО ОСНОВНОЙ ПРОЕКТ

	https://gist.github.com/YashasSamaga/e2b19a6807a13046e399f4bc3cca3a49
*/

#define IMX_477_FULLHD "nvarguscamerasrc exposuretimerange=\"300000 300000\" exposurecompensation=0   gainrange=\"16 16\" ! "\
"video/x-raw(memory:NVMM), "\
"width=1920,height=1080,framerate=60/1 ! "\
"nvvidconv flip-method=2 ! "\
"videoconvert ! "\
"video/x-raw, format=(string)BGR ! appsink"

#include <iostream>

#include <iostream>
#include <queue>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

/*
	0. Original COCO Dataset (80 classes)
	1. Waiter (1 class)
	2. Docking (3 classes)
*/

#define CHOSEN_NET  2

int yolo_num_classes[] = { 80,  1,  3 };

#if defined(_MSC_VER) || defined(WIN32)  || defined(_WIN32) || defined(__WIN32__) \
|| defined(WIN64)    || defined(_WIN64) || defined(__WIN64__) 

/*
	Пути к файлу с метками классов
*/
std::string yolo_classes_names[] = {
	"E:/University/12sem/object_detection_classes_yolov3.txt", 
	"D:/ML/yolo_v4/waiter/data/classes.names", 
	"D:/ML/yolo_v4/docking_station/data/classes.names"
};

#else

std::string yolo_classes_names[] = {
	"E:/University/12sem/object_detection_classes_yolov3.txt", 
	"/home/eugene/studying-dnn/yolo_v4/waiter/data/classes.names", 
	"/home/eugene/studying-dnn/yolo_v4/docking_station/data/classes.names"
};

#endif

/*
	Пути к весам
*/
#if defined(_MSC_VER) || defined(WIN32)  || defined(_WIN32) || defined(__WIN32__) \
|| defined(WIN64)    || defined(_WIN64) || defined(__WIN64__) 
std::string yolo_weights[] = { 
	"E:/University/12sem/yolov4.weights", 
	"D:/ML/yolo_v4/waiter/backup/yolov4_custom_train_best.weights",
	"D:/ML/yolo_v4/docking_station/backup/yolov4_custom_train_best.weights"
};

#else
std::string yolo_weights[] = {
	"E:/University/12sem/yolov4.weights",
	"/home/eugene/studying-dnn/yolo_v4/waiter/backup/yolov4_custom_train_best.weights",
	"/home/eugene/studying-dnn/yolo_v4/docking_station/backup/yolov4_custom_train_best.weights"
};
#endif

/*
	Конфиг файлы
*/
#if defined(_MSC_VER) || defined(WIN32)  || defined(_WIN32) || defined(__WIN32__) \
|| defined(WIN64)    || defined(_WIN64) || defined(__WIN64__) 
std::string yolo_cfgs[] = {
	"E:/University/12sem/yolov4.cfg", 
	"D:/ML/yolo_v4/waiter/cfg/yolov4_custom_test.cfg", 
	"D:/ML/yolo_v4/docking_station/cfg/yolov4_custom_test.cfg"
};
#else
std::string yolo_cfgs[] = {
	"E:/University/12sem/yolov4.cfg",
	"/home/eugene/studying-dnn/yolo_v4/waiter/cfg/yolov4_custom_test.cfg",
	"/home/eugene/studying-dnn/yolo_v4/docking_station/cfg/yolov4_custom_test.cfg"
};
#endif


constexpr float CONFIDENCE_THRESHOLD = 0.5;
constexpr float NMS_THRESHOLD = 0.1;
constexpr int NUM_CLASSES = 3;

// colors for bounding boxes
const cv::Scalar colors[] = {
	{0, 255, 255},
	{255, 255, 0},
	{0, 255, 0},
	{255, 0, 0}
};
const auto NUM_COLORS = sizeof(colors) / sizeof(colors[0]);

int main()
{
	std::vector<std::string> class_names;
	{
		//std::ifstream class_file("E:/University/12sem/object_detection_classes_coco.txt");
		//std::ifstream class_file("E:/University/12sem/object_detection_classes_yolov3.txt");
		std::ifstream class_file(yolo_classes_names[CHOSEN_NET]);
		if (!class_file)
		{
			std::cerr << "failed to open "<< yolo_classes_names[CHOSEN_NET] << "\n";
			return 0;
		}

		std::string line;
		while (std::getline(class_file, line))
			class_names.push_back(line);
	}

	bool _debug_on_video = false;
	cv::VideoCapture source;

#if defined(_MSC_VER) || defined(WIN32)  || defined(_WIN32) || defined(__WIN32__) \
|| defined(WIN64)    || defined(_WIN64) || defined(__WIN64__) 

	if (_debug_on_video) {
		source.open("demo.mp4");
	}
	else {
		source.open(0);
		//cap.set(CAP_PROP_FRAME_WIDTH, 1280);
		//cap.set(CAP_PROP_FRAME_HEIGHT, 720);
	}
#else
	source.open(IMX_477_FULLHD);
#endif

	auto net = cv::dnn::readNetFromDarknet(yolo_cfgs[CHOSEN_NET], yolo_weights[CHOSEN_NET]);
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
		cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(640, 480), cv::Scalar(), true, false, CV_32F);
		//cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(480, 360), cv::Scalar(), true, true, CV_32F);
		net.setInput(blob);

		auto dnn_start = std::chrono::steady_clock::now();
		net.forward(detections, output_names);
		auto dnn_end = std::chrono::steady_clock::now();

		std::vector<int> indices[NUM_CLASSES];
		std::vector<cv::Rect> boxes[NUM_CLASSES];
		std::vector<float> scores[NUM_CLASSES];

		for (auto& output : detections)
		{
			const auto num_boxes = output.rows;
			for (int i = 0; i < num_boxes; i++)
			{
				auto x = output.at<float>(i, 0) * frame.cols;
				auto y = output.at<float>(i, 1) * frame.rows;
				auto width = output.at<float>(i, 2) * frame.cols;
				auto height = output.at<float>(i, 3) * frame.rows;
				cv::Rect rect(x - width / 2, y - height / 2, width, height);

				for (int c = 0; c < NUM_CLASSES; c++)
				{
					auto confidence = *output.ptr<float>(i, 5 + c);
					if (confidence >= CONFIDENCE_THRESHOLD)
					{
						boxes[c].push_back(rect);
						scores[c].push_back(confidence);
					}
				}
			}
		}

		for (int c = 0; c < NUM_CLASSES; c++)
			cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);

		for (int c = 0; c < NUM_CLASSES; c++)
		{
			for (size_t i = 0; i < indices[c].size(); ++i)
			{
				const auto color = colors[c % NUM_COLORS];

				auto idx = indices[c][i];
				const auto& rect = boxes[c][idx];
				cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);

				std::ostringstream label_ss;
				label_ss << class_names[c] << ": " << std::fixed << std::setprecision(2) << scores[c][idx];
				auto label = label_ss.str();

				int baseline;
				auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
				cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
				cv::putText(frame, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
			}
		}

		auto total_end = std::chrono::steady_clock::now();

		float inference_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(dnn_end - dnn_start).count();
		float total_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
		std::ostringstream stats_ss;
		stats_ss << std::fixed << std::setprecision(2);
		stats_ss << "Inference FPS: " << inference_fps << ", Total FPS: " << total_fps;
		auto stats = stats_ss.str();

		int baseline;
		auto stats_bg_sz = cv::getTextSize(stats.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
		cv::rectangle(frame, cv::Point(0, 0), cv::Point(stats_bg_sz.width, stats_bg_sz.height + 10), cv::Scalar(0, 0, 0), cv::FILLED);
		cv::putText(frame, stats.c_str(), cv::Point(0, stats_bg_sz.height + 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));

		cv::namedWindow("output");

		if (frame.cols >= 720) {
			cv::resize(frame, frame, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
		}

		cv::imshow("output", frame);
	}

	return 0;
}