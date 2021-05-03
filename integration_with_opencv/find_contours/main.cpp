/*
	https://docs.opencv.org/3.4/de/d62/tutorial_bounding_rotated_ellipses.html
	https://docs.opencv.org/3.4/da/d5c/tutorial_canny_detector.html
*/

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat src_gray;
int thresh = 100;
RNG rng(12345);
void thresh_callback(int, void*);

int test(Mat &src) {


	resize(src, src, Size(), 0.4, 0.4, INTER_LINEAR);
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3));

	const char* source_window = "Source";
	namedWindow(source_window);
	imshow(source_window, src);
	const int max_thresh = 255;

	createTrackbar("Canny thresh:", source_window, &thresh, max_thresh, thresh_callback);
	thresh_callback(0, 0);
	waitKey();
}

void design_new_algo(Mat &frame) {

	vector<Rect> rois;
	rois.push_back(Rect(764, 261, 70, 114));
	rois.push_back(Rect(964, 235, 91, 130));
	rois.push_back(Rect(753, 605, 65, 121));
	rois.push_back(Rect(947, 648, 95, 156));


	cvtColor(frame, src_gray, COLOR_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3));

	Mat canny_output;
	Canny(src_gray, canny_output, 10, 250);


	for (Rect roi : rois) {
	
		vector<vector<Point> > contours;
		findContours(canny_output(roi), contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

		vector<RotatedRect> minRect(contours.size());
		vector<RotatedRect> minEllipse(contours.size());

		for (size_t i = 0; i < contours.size(); i++)
		{
			minRect[i] = minAreaRect(contours[i]);
			minRect[i].center.x += roi.x;
			minRect[i].center.y += roi.y;

			if (contours[i].size() > 5)
			{
				minEllipse[i] = fitEllipse(contours[i]);
				minEllipse[i].center.x += roi.x;
				minEllipse[i].center.y += roi.y;
			}
		}

		for (size_t i = 0; i < contours.size(); i++)
		{
			Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
			// contour
			//drawContours(frame, contours, (int)i, color);
			// ellipse
			ellipse(frame, minEllipse[i], color, 3);
			// rotated rectangle
			Point2f rect_points[4];
			minRect[i].points(rect_points);
			for (int j = 0; j < 4; j++)
			{
				line(frame, rect_points[j], rect_points[(j + 1) % 4], color);
			}
		}

	}


	for (Rect r : rois) {
		rectangle(frame, r, Scalar(255, 0, 0), 3);
	}



	resize(frame, frame, Size(), 0.7, 0.7);
	resize(canny_output, canny_output, Size(), 0.5, 0.5);

	imshow("Frame", frame);
	imshow("Canny", canny_output);
	waitKey(0);
}

int main(int argc, char** argv)
{
	CommandLineParser parser(argc, argv, "{@input | E:/University/12sem/ВКРМ/Нейронки/angle/imgs/30.jpg | input image}");
	Mat src = imread(samples::findFile(parser.get<String>("@input")));

	if (src.empty())
	{
		cout << "Could not open or find the image!\n" << endl;
		cout << "Usage: " << argv[0] << " <Input image>" << endl;
		return -1;
	}

	//test(src);
	design_new_algo(src);
	return 0;
}


void thresh_callback(int, void*)
{
	Mat canny_output;
	Canny(src_gray, canny_output, thresh, thresh * 2);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
	for (size_t i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		drawContours(drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0);
	}
	imshow("Contours", drawing);
}