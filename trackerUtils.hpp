#include "opencv2/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <thread>
#include <omp.h>

using namespace cv;

bool setCpuToThread(std::thread &th, int core);

inline int modul(int a, int b)
{
    // function calculates the module of two numbers and it takes into account also negative numbers
    return ((a % b) + b) % b;
}

inline double kernel_epan(double x)
{
    return (x <= 1) ? (2.0/3.14)*(1-x) : 0;
}

Mat gaussian_shaped_labels(const float sigma, const int w, const int h);

Mat get_hann_win(Size sz);

Mat bgr2hsv(const Mat &img);

Mat get_subwindow(const Mat &image, const Point2f center, const int w, const int h, Rect *valid_pixels);
Mat get_subwindow(const Mat &image, const Point2f center, const int w, const int h);

void get_features_rgb(const Mat &patch, const Size &output_size, std::vector<Mat> &features);
void get_features_cn(const Mat &ppatch_data, const Size &output_size, std::vector<Mat> &features);
void get_features_hog(const Mat &im, const int bin_size, std::vector<Mat> &features);

void fourier_transform_mat(const Mat &M, Mat &out);
std::vector<Mat> fourier_transform_features(const std::vector<Mat> &M);
std::vector<Mat> fourier_transform_features_(const std::vector<Mat> &M);

Mat divide_complex_matrices(const Mat &A, const Mat &B);

Mat circshift(Mat matrix, int dx, int dy);

double get_max(const Mat &m);
double get_min(const Mat &m);


float subpixel_peak(const Mat &response, const std::string &s, const Point2f &p);
