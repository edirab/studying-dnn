#include "AUV.h"


AUV::AUV() {

	m1.resize(2);
	m2.resize(2);

	
	//this->blobDetector = SimpleBlobDetector::create(params);

	this->readClassNames();

	net = cv::dnn::readNetFromDarknet(CFG_FILE, WEIGHTS);
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	output_names = net.getUnconnectedOutLayersNames();

	/*
	Simplified solution

	double focal_length = frame_gray.cols; // Approximate focal length.
	Point2d center = cv::Point2d(frame_gray.cols / 2, frame_gray.rows / 2);
	cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x, 0, focal_length, center.y, 0, 0, 1);
	*/

	cMatrix640 = (Mat_<double>(3, 3) << 5.3226273868525448e+02, 0, 3.2590522394049350e+02, 0, 5.3226273868525448e+02, 2.6946997900677803e+02, 0, 0, 1);
	cMatrix1280 = (Mat_<double>(3, 3) << 8.6155235630774325e+02, 0, 6.2961522415048103e+02 ,0, 8.6155235630774325e+02, 3.9881978167213623e+02,0, 0, 1);

	distortion640 = (Mat_<double>(1, 5) << 0, -6.1539772782054671e-02, 0, 0, 1.7618036793466491e-02);
	distortion1280 = (Mat_<double>(1, 5) << 0, -6.5524123635067169e-02, 0, 0, 0 );

	cMatrixFullHD = (Mat_<double>(3, 3) << 1.8319233710098095e+04, 0, 9.5950000000000000e+02,
											0, 1.8319233710098095e+04, 5.3950000000000000e+02,
											0, 0, 1);

	distortionFullHD = (Mat_<double>(1, 5) << 2.5318626977076864e+01, -1.2398369018684898e+04, 0, 0, 2.4953338626272022e+06);

	// Задание координат маркеров
	model_points.push_back(cv::Point3d(-50, 50, 0));  // left up corner
	model_points.push_back(cv::Point3d(50, 50, 0));   // right up corner
	model_points.push_back(cv::Point3d(50, -50, 0));  // left up corner
	model_points.push_back(cv::Point3d(-50, -50, 0)); // left down corner

	fout.open("E:/University/10sem/nirs/haar_3_4_6/haar_navigation/filtering_output.dat");

	if (!fout) {
		cout << "Error opening fout \n";
	}
}


AUV ::~AUV() {

	fout.flush();
	fout.close();
}


void AUV::rotate_over_normal(Mat& frame) {

	static double delta_x = 0;
	static double delta_y = 0;
	static double alpha = 0;

	if (m2.size() == 2) {

		//PointA
		delta_x = abs(m2[0].x - m2[1].x);
		delta_y = abs(m2[0].y - m2[1].y);

		//не известно, в каком порядке детектируются точки: сначала левая, а потом правая, или наоборот
		// Если ОДНА точка левее и выше, то считаем поворот по ч.с. со знаком "+"
		if ((m2[0].x < m2[1].x && m2[0].y > m2[1].y) || (m2[1].x < m2[0].x && m2[1].y > m2[0].y)) {
			alpha = atan(delta_y / delta_x);
		}
		else {
			alpha = -atan(delta_y / delta_x);
		}
	}

	if (abs(alpha) >= 0 && abs(alpha) < 2 * 3.1415926535) {
		this->d_roll = alpha * 180 / 3.1415926535;
	}

	ostringstream strstream;
	strstream << setprecision(2);
	strstream << "rotation: ";
	strstream << this->d_roll;
	strstream << " deg";

	//string str = "Rotation over " + to_string(degs) + "degs";

	String text(strstream.str());
	int text_y = int(frame.rows * 0.6);
	putText(frame, text, Point(100, text_y), 0, 1, BLK, 2);

	return;
}


void AUV::arrange_markers(Mat& frame, bool debug) {

	//assert(m1.size() == 2 && m2.size() == 2);

	if ((m1.size() == 2 && m2.size() == 2)) 
	{
		bool for_comparison = true;

		if (for_comparison)
		{
			if (m1[0].x > m1[1].x) {
				swap(m1[0], m1[1]);
			}
			if (m2[0].x > m2[1].x) {
				swap(m2[0], m2[1]);
			}
		}
		else 
		{
			if (this->d_roll > 0) {

				//if (m1[0].x > m1[1].x && m1[0].y < m1[1].y) {
				if (m1[0].y < m1[1].y) {
					swap(m1[0], m1[1]);
				}
				if (m2[0].y < m2[1].y) {
					swap(m2[0], m2[1]);
				}
			}
			else if (this->d_roll <= 0) {

				if (m1[0].y > m1[1].y) {
					swap(m1[0], m1[1]);
				}
				if (m2[0].y > m2[1].y) {
					swap(m2[0], m2[1]);
				}
			}
		}
		Scalar COLOR;
		if (frame.channels() == 1) {
			COLOR = WHT;
		}
		else {
			COLOR = RED;
		}

		if (debug) {
			putText(frame, String("11"), Point(m1[0].x + 10, m1[0].y - 10), 1, 1, COLOR);
			putText(frame, String("12"), Point(m1[1].x + 10, m1[1].y - 10), 1, 1, COLOR);
			putText(frame, String("21"), Point(m2[0].x + 10, m2[0].y - 10), 1, 1, COLOR);
			putText(frame, String("22"), Point(m2[1].x + 10, m2[1].y - 10), 1, 1, COLOR);

			draw_configuration(frame, this->m1, this->m2);
		}
	}
}


void AUV::calculate_distance(Mat& frame, bool debug) {
	
	if (m1.size() == 2) {
		//Point a, b;
		upper = sqrt(pow(abs(m1[0].x - m1[1].x), 2) + pow(abs(m1[0].y - m1[1].y), 2));
	}

	if (m2.size() == 2) {
		lower = sqrt(pow(abs(m2[0].x - m2[1].x), 2) + pow(abs(m2[0].y - m2[1].y), 2));
	}

	if (debug) {
		cout << "upper = " << upper << " lower = " << lower << "\n";
	}

	int w = frame.cols;
	float scale = float(w) / 640;

	double average = (upper + lower) / 2;
	double calculated_distance = 50 * 100 / (average / scale);

	if (calculated_distance > 0 && calculated_distance <= 200)
		this->dist = calculated_distance;

	ostringstream strstream;
	//strstream << setprecision(0);
	strstream << "d, cm: ";
	strstream << setw(3) << int(dist);// << " " << setw(3) << int(lower);

	String text(strstream.str());
	//putText(frame, text, Point(10, 400), 0, 1, Scalar(255, 255, 255), 2);
	int text_y = int(frame.rows * 0.5);
	putText(frame, text, Point(100, text_y), 0, 1, BLK, 2);	
}


void AUV::line_equation(double& k, double& b, bool mainDiag) {
	// y = k*x + b
	Eigen::Matrix2d A;
	Eigen::Vector2d B;
	Eigen::Vector2d X;

	if (m1.size() == 2 && m2.size() == 2) {

		if (mainDiag) {

			A.row(0) << m1[0].x, 1;
			A.row(1) << m2[1].x, 1;

			B << m1[0].y, m2[1].y;
		}
		else {
			A.row(0) << m1[1].x, 1;
			A.row(1) << m2[0].x, 1;

			B << m1[1].y, m2[0].y;
		}
		//cout << A << "\n\n";
		//cout << B << "\n\n";

		X = A.lu().solve(B);

		k = X[0];
		b = X[1];
	}
}


void AUV::calculate_deltas(Mat& frame, bool debug) {

	static double k1 = 0;
	static double k2 = 0;
	static double b1 = 0;
	static double b2 = 0;

	if (m1.size() == 2) {
		// y = k*x + b
		this->line_equation(k1, b1, true);
		this->line_equation(k2, b2, false);

		cout << k1 << " " << b1 << " " << m1[0].x << " " << m1[0].y << " ";
		cout << m2[1].x << " " << m2[1].y << "\n";

		cout << k2 << " " << b2 << " " << m1[1].x << " " << m1[1].y << " ";
		cout << m2[0].x << " " << m2[0].y << "\n";
	}

	return;
}

void AUV::filter_objects_2(vector<Rect> objects, Mat& currentFrame, Mat& frame_gray, markerType m_type, Mat AUV_sees, bool debug = false) {

	vector<Marker> markers_;
	vector<Marker> hough_valid;
	Mat roi_mat_gray;
	Mat roi_mat_color;

	sort(objects.begin(), objects.end(), compar);
	//print_objects(objects);

	if (m_type == markerType::black_circle) {
		blobDetector_params.blobColor = 80;
	}
	else {
		blobDetector_params.blobColor = 255;
	}
	this->blobDetector = SimpleBlobDetector::create(blobDetector_params);

	if (debug) {
		cout << "objects.size() = " << objects.size() << "\n";

		for (size_t i = 0; i < objects.size(); ++i)
		{
			const auto color = colors[0];
			const auto& rect = objects[i];
			cv::rectangle(currentFrame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);
		}
	}

	// всё делается за один проход
	for (int i = 0; i < objects.size(); i++) {

		vector<Vec3f> circles;
		roi_mat_gray = frame_gray(objects[i]);
		roi_mat_color = currentFrame(objects[i]);
		medianBlur(roi_mat_gray, roi_mat_gray, 5);

		/*
		Ищем все кружочки внутри одного ROI
		*/
		vector<KeyPoint> keypoints;
		blobDetector->detect(roi_mat_gray, keypoints);
		drawKeypoints(roi_mat_color, keypoints, roi_mat_color, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		imshow("Key points", roi_mat_color);
		waitKey(0);

		/*
			Если каскад указал на объект и детектор Хаффа нашёл кружочек, то скорее всего, это то что нужно
		*/
		if (circles.size() == 1) {

			Marker temp_m(objects[i].x + int(circles[0][0]), objects[i].y + int(circles[0][1]), m_type, objects[i]);
			hough_valid.push_back(temp_m);
		}
		/*
		В одном roi_mat_gray кружочков больше одного. Что странно
		Этот блок практически ничего не делает. За всё тестовое видео сработал 4 раза
		*/
		else if (circles.size() > 1) {

			Mat t;
			if (m_type == markerType::black_circle) {
				t = Marker::get_template_t1(roi_mat_gray.rows, roi_mat_gray.cols);
				threshold(roi_mat_gray, roi_mat_gray, 200, 255, 0);
				absdiff(roi_mat_gray, t, roi_mat_gray);
				int nonZero = countNonZero(roi_mat_gray);

				//if (nonZero < 0.1 * frame_gray.cols) {
				//	hough_valid.push_back(objects[i]);
				//}
			}
			else {
				t = Marker::get_template_t2(roi_mat_gray.rows, roi_mat_gray.cols);
				threshold(roi_mat_gray, roi_mat_gray, 200, 255, 0);
				absdiff(roi_mat_gray, t, roi_mat_gray);
				int nonZero = countNonZero(roi_mat_gray);
				//if (nonZero < 0.15 * frame_gray.cols) {
				//	hough_valid.push_back(objects[i]);
				//}
			}
		}
		/*
		Ситуация: каскад утверждает, что есть объект, но детектор Хаффа кружочка не нашёл
		вот здесь и можно проверить маской
		*/
		else {
			Mat t;
			

			if (m_type == markerType::black_circle) {
				t = Marker::get_template_t1(roi_mat_gray.rows, roi_mat_gray.cols);
				threshold(roi_mat_gray, roi_mat_gray, 60, 255, THRESH_BINARY);
				//imshow("roi_mat_gray m1 thresholded", roi_mat_gray);

				absdiff(roi_mat_gray, t, roi_mat_gray);
				int nonZero = countNonZero(roi_mat_gray);
				//cout << "m1 = " << setw(7) << nonZero << "\n";
				//fout << "m1 = " << nonZero << "\n";
			}
			else {
				t = Marker::get_template_t2(roi_mat_gray.rows, roi_mat_gray.cols);
				threshold(roi_mat_gray, roi_mat_gray, 200, 255, THRESH_BINARY);
				//imshow("roi_mat_gray m2 thresholded", roi_mat_gray);

				absdiff(roi_mat_gray, t, roi_mat_gray);
				int nonZero = countNonZero(roi_mat_gray);
				//cout << "m2 = " << setw(7) << nonZero << "\n";
				//fout << "m2 = " << nonZero << "\n";
			}
		}
		// конец проверки маской
	}

	/*
	
	*/
	if (hough_valid.size() > 2) {
		for (size_t i = 0; i < hough_valid.size() - 1; i++) {

			int max_width = max(hough_valid[i].roi.width, hough_valid[i + 1].roi.width);
			int min_width = min(hough_valid[i].roi.width, hough_valid[i + 1].roi.width);

			int delta = max_width - min_width;
			//delta = abs(delta);
			double diff = double(delta) / double(max_width);

			if (debug)
				cout << "delta = " << delta << " diff = " << diff << "\n";

			if (diff < 0.25) {

				markers_.push_back(hough_valid[i]);
				markers_.push_back(hough_valid[i + 1]);
				break;
			}
		}
	}
	else if (hough_valid.size() == 2){

		if (m_type == markerType::black_circle) {
			m1[0] = hough_valid[0];
			m1[1] = hough_valid[1];
		}
		else {
			m2[0] = hough_valid[0];
			m2[1] = hough_valid[1];
		}
	}
	else {
		//pass
	}
	return;
}


Vec3f AUV::rotationMatrixToEulerAngles(Mat& R) {

	assert(isRotationMatrix(R));

	float sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));

	bool singular = sy < 1e-6; // If

	float x, y, z;
	if (!singular)
	{
		x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
		y = atan2(-R.at<double>(2, 0), sy);
		z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
	}
	else
	{
		x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
		y = atan2(-R.at<double>(2, 0), sy);
		z = 0;
	}
	return Vec3f(x, y, z);
}


void AUV::estimatePos(Mat &frame, bool draw_perp) {

	if (this->m1.size() == 2 && this->m2.size() == 2) {

		vector<Point2d> corners = {
									Point2d(m1[0].x, m1[0].y),
									Point2d(m1[1].x, m1[1].y),
									Point2d(m2[1].x, m2[1].y),
									Point2d(m2[0].x, m2[0].y)
		};

		//estimatePoseSingleMarkers(corners, markerLen, cMatrix640, distortion640, Rvec, Tvec);

		// Solve for pose
		solvePnP(model_points, corners, cMatrixFullHD, distortionFullHD, this->Rvec, this->Tvec);
		//solvePnP(model_points, corners, camera_matrix, distortion640, Rvec, Tvec);

		Mat rotMat(3, 3, CV_64F);
		Rodrigues(this->Rvec, rotMat);

		this->Euler_angles = rotationMatrixToEulerAngles(rotMat);
		//cout << setprecision(3);
		//cout << "In estimatePos: " << setw(7) << Euler_angles[0] * 180 / M_PI << setw(7) << Euler_angles[1] * 180 / M_PI << setw(7) << Euler_angles[2] * 180 / M_PI << "\n";
		//cout  << Euler_angles[1] * 180 / 3.1415926  << "\n";
	
		bool show_tvec = false;

		if (show_tvec) {
			cout << setprecision(5);

			for (int j = 0; j < Tvec.rows; j++) {
				cout << setw(8) << Tvec.at<double>(j, 0);
			}
			cout << "\n";
		}

		if (draw_perp) {

			vector<Point2d> perpendicular_point2D;
			vector<Point3d> perpendicular_point3D;
				perpendicular_point3D.push_back(Point3d(0, 0, 0));
				perpendicular_point3D.push_back(Point3d(0, 0, 100));

			projectPoints(perpendicular_point3D, Rvec, Tvec, cMatrix640, distortion640, perpendicular_point2D);
			line(frame, perpendicular_point2D[0], perpendicular_point2D[1], cv::Scalar(255, 0, 0), 2);
		}
	}
	else {
		cout << "Less than 4 markers\n";
		cout << m1.size() << " "  << m2.size() << "\n";
	}
}


double AUV::get_Euler_1() {
	return Euler_angles[1] * 180 / M_PI;
}


void AUV::get_orientation(Mat &frame) {

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	vector<Rect> markers_bc, markers_wc;

	for (size_t i = 0; i < indices[1].size(); ++i) {
		auto idx = indices[1][i];
		const auto& rect = boxes[1][idx];
		markers_wc.push_back(rect);
	}

	for (size_t i = 0; i < indices[2].size(); ++i) {
		auto idx = indices[2][i];
		const auto& rect = boxes[2][idx];
		markers_bc.push_back(rect);
	}

	this->Process(frame);
	this->Draw(frame);
	this->AnnotateStats(frame);

	filter_objects_2(markers_bc, frame, frame_gray, markerType::black_circle, AUV_sees, false);
	filter_objects_2(markers_wc, frame, frame_gray, markerType::white_circle, AUV_sees, false);

	//print_objects(m1, "-");

	Mat our_markers = Mat::zeros(frame.size(), CV_8UC1);

	this->rotate_over_normal(frame);
	this->arrange_markers(our_markers, true);
	this->calculate_distance(frame, false);
	////this->calculate_deltas(frame, true);

	this->estimatePos(frame, false);

	AUV_sees = Mat::zeros(frame.size(), CV_8UC1);

	for (int i = 0; i < m1.size(); i++) {
		rectangle(AUV_sees, m1[i].roi, WHT, -1);
	}
	for (int i = 0; i < m2.size(); i++) {
		rectangle(AUV_sees, m2[i].roi, WHT, -1);
	}

	////cout << m1.size() << " " << m2.size() << "\n";
	////imshow("AUV mask", AUV_sees);
	//AUV_sees = AUV_sees & frame_gray;

	//imshow("AUV sees", AUV_sees);
	//imshow("our markers", our_markers);

	draw_objects(frame, m1, YEL);
	draw_objects(frame, m2, PNK);
}


void AUV::readClassNames() {

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

void AUV::clearArrays() {

	for (int i = 0; i < NUM_CLASSES; i++) {

		indices[i].clear();
		boxes[i].clear();
		scores[i].clear();
	}
}


void AUV::Process(const Mat &frame) {

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


void AUV::Draw(Mat &frame) {

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


void AUV::AnnotateStats(Mat &frame) {
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