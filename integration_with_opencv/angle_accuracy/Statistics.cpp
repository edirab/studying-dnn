#include "Statistics.h"


Statistics::Statistics() {

	// 5 seconds of video * framerate
	values.reserve(5 * 30);
	values_non_negative.reserve(5 * 30);
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
	for (double elem : values) { this->average += elem; }
	this->average /= values.size();
	return this->average;
}

double Statistics::get_average_filtered() {

	this->average_filtered = 0;
	for (double elem : values_non_negative) { average_filtered += elem; }
	average_filtered /= values_non_negative.size();
	return average_filtered;
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

	constexpr int w1 = 35;
	constexpr int w2 = 7;

	cout << "Stats for" << path << "\n";
	cout << std::setprecision(4);

	cout << setw(w1) << "\tLen values: "          << setw(w2) << values.size() << "\n";
	cout << setw(w1) << "\tLen values filtered: " << setw(w2) << values_non_negative.size() << "\n";

	cout << setw(w1) << "\tAverage: "           << setw(w2) << get_average() << "\n";
	cout << setw(w1) << "\tAverage filtered: "  << setw(w2) << get_average_filtered() << "\n";

	cout << setw(w1) << "\tMedian: "          << setw(w2) << get_median() << "\n";
	cout << setw(w1) << "\tMedian filtered: " << setw(w2) << get_median_filtered() << "\n";

	cout << setw(w1) << "\tStd dev: "          << setw(w2) << get_std_dev() << "\n";
	cout << setw(w1) << "\tStd dev filtered: " << setw(w2) << get_std_filtered() << "\n";

	return;
}