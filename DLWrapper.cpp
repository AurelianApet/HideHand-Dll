// DLWrapper.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <exception>

#include "opencv2/core/cuda.hpp"
using namespace cv;
using namespace std;

class Histogram1D {
private:
	int histSize[1];
	float hranges[2];
	const float* ranges[1];
	int channels[1];
public:
	Histogram1D() {
		histSize[0] = 256;
		hranges[0] = 0.0;
		hranges[1] = 255.0;
		ranges[0] = hranges;
		channels[0] = 0;
	}


	cv::MatND getHistogram(const cv::Mat &image) {
		cv::MatND hist;
		cv::calcHist(&image,
			1,
			channels,
			cv::Mat(),
			hist,
			1,
			histSize,
			ranges
		);
		return hist;
	}


	cv::Mat getHistogramImage(const cv::Mat &image) {

		cv::MatND hist = getHistogram(image);


		double maxVal = 0;
		double minVal = 0;
		cv::minMaxLoc(hist, &minVal, &maxVal, 0, 0);


		cv::Mat histImg(histSize[0], histSize[0], CV_8U, cv::Scalar(255));

		int hpt = static_cast<int>(0.9*histSize[0]);

		for (int h = 0; h<histSize[0]; h++) {
			float binVal = hist.at<float>(h);
			int intensity = static_cast<int>(binVal*hpt / maxVal);
			cv::line(histImg, cv::Point(h, histSize[0]), cv::Point(h, histSize[0] - intensity), cv::Scalar::all(0));
		}
		return histImg;
	}
};


extern "C" {
	_declspec(dllexport) bool isHideByHand(unsigned char* data, int height, int width)
	{
		Mat constructed (height, width, CV_8UC4, data);
		try {
			if (!constructed.data)
				return false;

			Histogram1D h;
			cv::MatND histo = h.getHistogram(constructed);

			float sum = 0;
			for (int i = 0; i < 256; i++) {
				sum += histo.at<float>(i);
			}
			float target = 0;
			for (int i = 0; i < 150; i++) {
				target += histo.at<float>(i);
			}

			if (target / sum > 0.7) {
				return true;					 
			}
			else {
				return false;
			}
		}
		catch (Exception ex)
		{
			return false;
		}
	}

	_declspec(dllexport) float GetHideByHand(unsigned char* data, int height, int width)
	{
		Mat constructed(height, width, CV_8UC4, data);
		try {
			if (!constructed.data)
				return false;

			Histogram1D h;
			cv::MatND histo = h.getHistogram(constructed);

			float sum = 0;
			for (int i = 0; i < 256; i++) {
				sum += histo.at<float>(i);
			}
			float target = 0;
			for (int i = 0; i < 150; i++) {
				target += histo.at<float>(i);
			}
			float value = target / sum;
			return value;
		}
		catch (Exception ex)
		{
			return 0;
		}
	}

	_declspec(dllexport) double GetHisRate(unsigned char* data, int height, int width)
	{		
		Mat imgs(height, width, CV_8UC4, data);
		Mat imgs1;
		imgs1 = imread("1.jpg", IMREAD_COLOR);
		Mat imgsHLS, imgsHLS1;
		if (imgs.data == 0 || imgs1.data == 0)
		{
			return 0;
		}

		cvtColor(imgs, imgsHLS, COLOR_BGR2HLS);
		cvtColor(imgs1, imgsHLS1, COLOR_BGR2HLS);
		Mat histogram, histogram1;
		int channel_numbers[] = { 0, 1, 2 };
		int* number_bins = new int[imgsHLS.channels()];
		for (int ch = 0; ch < imgsHLS.channels(); ch++)
		{
			number_bins[ch] = 8;
		}
		float ch_range[] = { 0.0, 255.0 };
		const float *channel_ranges[] = { ch_range, ch_range, ch_range };
		calcHist(&imgsHLS, 1, channel_numbers, Mat(), histogram, imgsHLS.channels(), number_bins, channel_ranges);
		normalize(histogram, histogram, 1.0);

		number_bins = new int[imgsHLS1.channels()];
		for (int ch = 0; ch < imgsHLS1.channels(); ch++)
		{
			number_bins[ch] = 8;
		}
		calcHist(&imgsHLS1, 1, channel_numbers, Mat(), histogram1, imgsHLS1.channels(), number_bins, channel_ranges);
		normalize(histogram1, histogram1, 1.0);

		double matching_score = compareHist(histogram, histogram1, HISTCMP_CORREL);
		return matching_score;
	}
}