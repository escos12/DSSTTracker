
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <chrono>

#include "opencv2/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "trackerCSRT.hpp"

String seqense = "bike1";

std::string ZeroPadNumber(int num, int pad)
{
	std::ostringstream ss;
	ss << std::setw(pad) << std::setfill( '0' ) << num;
	return ss.str();
}

std::vector<cv::Rect2d> parseCoorf(String filename)
{
	std::vector<cv::Rect2d> coordinates;
	std::ifstream read(filename);

	// read until you reach the end of the file
	for (std::string line; std::getline(read, line); )
	{
		std::stringstream ss(line);
		std::string token;

		std::vector<int> vals;
		while(std::getline(ss, token, ','))
		{
			vals.push_back(std::stoi( token ));
		}
		if (vals.size() == 4)
		{
			cv::Rect2d rect(vals[0], vals[1], vals[2], vals[3]);
			std::cout << rect << std::endl; //and output it
			coordinates.push_back(rect);
		}
	}
	read.close();

	return coordinates;
}

int main(int argc, char **argv)
{
	std::vector<cv::Rect2d> pos =  parseCoorf("../DSSTTracker/"+ seqense + "/" + seqense + ".txt");
	Rect2d obj = pos[0];
	//Rect2d obj(703,361,64,96);
	TrackerCSRT tracker;
	int counter = 1;
	cv::Mat in = cv::imread("../DSSTTracker/"+ seqense + "/" + ZeroPadNumber(counter, 6) + ".jpg", 1);
	//cv::Mat in(768, 1024, CV_8UC3, Scalar(255,0,0));
	tracker.initImpl(in, obj);


	while(true)
	{
		//counter++;
		//in = cv::imread("../DSSTTracker/"+ seqense + "/" + ZeroPadNumber(counter, 6) + ".jpg", 1);
		cv::Mat input = in.clone();

		auto start = std::chrono::high_resolution_clock::now();
		std::vector<int> times = tracker.updateImpl(input, obj);
		auto finish = std::chrono::high_resolution_clock::now();
		int tm = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();

		putText(input, "FPS:" + std::to_string(1000/tm), cv::Point(50, 45), 1, 1, cv::Scalar(150, 235, 80), 1, LINE_8, false);
		putText(input, "elapsed time all:" + std::to_string(tm) + "ms", cv::Point(50, 60), 1, 1, cv::Scalar(150, 235, 80), 1, LINE_8, false);

		std::cout << "FPS:" + std::to_string(1000/tm) << "  \t elapsed time all:" + std::to_string(tm) + "ms ";
		std::cout << std::endl;

		rectangle(input, obj, Scalar(0, 0, 255), 2, 8, 0);
		rectangle(input, pos[counter -1], Scalar(0, 255, 255), 1, 8, 0);
		cv::imshow("input", input);
		char k = cv::waitKey(1);
		if (k == 27)
			break;
	}

	return 0;
}
