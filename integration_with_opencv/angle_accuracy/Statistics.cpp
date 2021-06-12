#include "Statistics.h"


Statistics::Statistics() {

	// 5 seconds of video * framerate
	values.reserve(5 * 30);
	values_non_negative.reserve(5 * 30);

	out_file.open("C:/aruco.txt");
}

Statistics::~Statistics() {

	if (out_file.is_open())
		out_file.close();

}

void Statistics::add(double val) {

	values.push_back(val);

	if (val >= 0) {
		values_non_negative.push_back(val);
	}

	if (out_file.is_open()) out_file << std::to_string(val) << "\n";
}

double Statistics::get_median() {

	double m = 0;
	std::sort(values.begin(), values.end());

	if (values.size()) {

		if (values.size() % 2 == 1) {
			m = values[values.size() / 2];
		}
		else {
			double m1 = values[values.size() / 2 - 1];
			double m2 = values[values.size() / 2];
			m = (m1 + m2) / 2;
		}
	}
	this->median = m;
	return m;
}


double Statistics::get_median_filtered() {
	double m = 0;
	std::sort(values_non_negative.begin(), values_non_negative.end());

	if (values_non_negative.size()) {

		if (values_non_negative.size() % 2 == 1) {
			m = values_non_negative[values_non_negative.size() / 2];
		}
		else {
			double m1 = values_non_negative[values_non_negative.size() / 2 - 1];
			double m2 = values_non_negative[values_non_negative.size() / 2];
			m = (m1 + m2) / 2;
		}
	}
	this->median_filtered = m;
	return m;
}


double Statistics::get_average() {

	this->average = 0;

	if (values.size() > 0) {
		for (double elem : values) { this->average += elem; }
		this->average /= values.size();
	}
	return this->average;
}


double Statistics::get_average_filtered() {

	this->average_filtered = 0;

	if (values_non_negative.size() > 0) {
		for (double elem : values_non_negative) { this->average_filtered += elem; }
		this->average_filtered /= values_non_negative.size();
	}
	return this->average_filtered;
}


double Statistics::get_std_dev() {

	double std_dev = 0;
	int N = values.size();
	if (N > 1) {
	
		for (double elem : values) {
			std_dev += ((elem - this->average) * (elem - this->average));
		}
		std_dev /= (N - 1);
		std_dev = std::sqrt(std_dev);
		this->std_dev = std_dev;
		
	}
	return this->std_dev;
}


double Statistics::get_std_filtered() {
	double std_dev = 0;
	int N = values_non_negative.size();
	if (N > 1) {

		for (double elem : values_non_negative) {
			std_dev += ((elem - this->average_filtered) * (elem - this->average_filtered));
		}
		std_dev /= (N - 1);
		std_dev = std::sqrt(std_dev);
		this->std_filtered = std_dev;

	}
	return this->std_filtered;
}


void Statistics::print_stats(string path) {

	constexpr int w1 = 28;
	constexpr int w2 = 7;

	cout << "  Stats for \"" << path << "\" \n";
	cout << std::setprecision(4);

	cout << setw(w1) << "Len values: "          << setw(w2) << values.size() << "\n";
	//cout << setw(w1) << "Len values filtered: " << setw(w2) << values_non_negative.size() << "\n";

	cout << setw(w1) << "Average: "           << setw(w2) << get_average() << "\n";
	//cout << setw(w1) << "Average filtered: "  << setw(w2) << get_average_filtered() << "\n";

	cout << setw(w1) << "Median: "          << setw(w2) << get_median() << "\n";
	//cout << setw(w1) << "Median filtered: " << setw(w2) << get_median_filtered() << "\n";

	cout << setw(w1) << "Std dev: "          << setw(w2) << get_std_dev() << "\n";
	//cout << setw(w1) << "Std dev filtered: " << setw(w2) << get_std_filtered() << "\n";

	return;
}