#pragma once
#define _USE_MATH_DEFINES
#include <math.h>

#include <Eigen/LU>
#include <fstream>

#include <opencv2/opencv.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"

//#include <opencv2/aruco.hpp>

#include "Marker.h"
#include "functions.h"

#define CASCADE_BLACK_CIRCLE "E:/University/10sem/nirs/haar_3_4_6/preparing navigation/haar_navigation_m1_v3/cascade.xml"
#define CASCADE_WHITE_CIRCLE "E:/University/10sem/nirs/haar_3_4_6/preparing navigation/haar_navigation_m2_v1/cascade.xml"


using namespace cv;
//using namespace cv::cuda;
//using namespace aruco;
using namespace std;

class AUV {

	double detection_coeff; // px * coeff = cm

	CascadeClassifier marker_type_1, marker_type_2;

	int marker1Counter = 0, marker2Counter = 0;
	int marker1Counter_prev = 0, marker2Counter_prev = 0;

	Mat frame_gray;

	// Матрицы камеры. Получены при калибровке
	Mat cMatrix640;
	Mat distortion640;
	Mat cMatrix1280;
	Mat distortion1280;

	Mat cMatrixFullHD;
	Mat distortionFullHD;

	// Координаты маркеров донной станции в глобальной системе координат
	vector<Point3d> model_points;

	Mat Rvec;
	Mat Tvec;

	Vec3f Euler_angles;

	float markerLen = 100; // здесь именно расстояние между нашими маркерами, а не сторона одного маркера

public:
	Mat AUV_sees;

	vector<Marker> m1;
	vector<Marker> m2;

	double delta_x;
	double delta_y;
	double delta_z;

	double upper;
	double lower;
	double dist; //aka наклонная дальность

	double d_yaw;
	double d_pitch;
	double d_roll;

	// debug purposes
	ofstream fout;
	//Mat AUV_sees;

	AUV();
	~AUV ();


	/*
		Calculates rotation matrix to euler angles
		The result is the same as MATLAB except the order
		of the euler angles ( x and z are swapped ).
	*/
	Vec3f rotationMatrixToEulerAngles(Mat& R);

	double get_Euler_1();


	void get_orientation(Mat& frame);

	/*
		Определяет наклон камеры по крену (ну почти)
		в диапазоне от -90 до +90 градусов
	*/
	void rotate_over_normal(Mat& frame);
	void calculate_distance(Mat& frame, bool debug);

	void calculate_deltas(Mat& frame, bool debug);
	void line_equation(double &k, double &b, bool mainDiag);

	void filter_objects_2(vector<Rect> objects, Mat& currentFrame, Mat& frame_gray, markerType m_type, Mat AUV_sees, bool debug);
	void arrange_markers(Mat& frame, bool debug);

	/*
		Заполняет Rvec && Tvec, вызваая solvePnP
		Выполняет преобразование Родригеса
		Рисует перпендикуляр (при необходимости)
	*/
	void estimatePos(Mat& frame, bool draw_perp);
};

