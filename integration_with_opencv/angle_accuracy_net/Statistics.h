#pragma once
#define _USE_MATH_DEFINES

#include <math.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <algorithm>

using std::string;
using std::vector;
using std::cout;
using std::setw;
using std::ofstream;

class Statistics
{

public:
	Statistics();
	~Statistics();

	void print_stats(string path);

	void add(double val);

	double get_median();
	double get_average();
	double get_std_dev();

	double get_median_filtered();
	double get_average_filtered();
	double get_std_filtered();

private:
	ofstream out_file;

	vector <double> values;
	vector <double> values_non_negative;

	double median{0};
	double average{0};
	double std_dev{0};

	double median_filtered{0};
	double average_filtered{0};
	double std_filtered{0};
};

