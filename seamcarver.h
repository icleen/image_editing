#ifndef SEAMCARVER_H
#define SEAMCARVER_H

#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;

class SeamCarver
{
public:
  SeamCarver() {
    seam_count = 0;
  };
  ~SeamCarver() {
    for(int y = 0; y < seam_count; y++) {
      delete[] seams[y];
    }
    delete[] seams;
  };

  cv::Mat carve(cv::Mat image, int seams_to_remove);

  cv::Mat drawSeams(cv::Mat image);

  int* clusters();

private:

  int** carve_seam(int **img, int rows, int cols);

  void delete2darray(int **arr, int rows);

  int **seams;
  int seam_count;

};

#endif // SEAMCARVER_H
