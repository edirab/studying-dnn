#include "Experiment.h"

Experiment::Experiment() {

	this->readClassNames();

	net = cv::dnn::readNetFromDarknet(CFG_FILE, WEIGHTS);
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	//net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	//net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	output_names = net.getUnconnectedOutLayersNames();
}


void Experiment::readClassNames() {

	std::ifstream class_file(CLASSES);
	if (!class_file)
	{
		std::cerr << "failed to open classes.txt\n";
		return;
	}

	std::string line;
	while (std::getline(class_file, line))
		class_names.push_back(line);
	return;
}

void Experiment::clearArrays() {

	for (int i = 0; i < NUM_CLASSES; i++) {

		indices[i].clear();
		boxes[i].clear();
		scores[i].clear();
	}
}


void Experiment::Process() {

	clearArrays();

	total_start = std::chrono::steady_clock::now();
	cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(640, 480), cv::Scalar(), true, false, CV_32F);
	//cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(480, 360), cv::Scalar(), true, true, CV_32F);
	net.setInput(blob);

	dnn_start = std::chrono::steady_clock::now();
	net.forward(detections, output_names);
	dnn_end = std::chrono::steady_clock::now();

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

	//std::cout << indices->size() << " " << boxes->size() << " " << scores->size() << "\n";
}


void Experiment::Draw() {

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
}


void Experiment::AnnotateStats() {
	total_end = std::chrono::steady_clock::now();

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
}

