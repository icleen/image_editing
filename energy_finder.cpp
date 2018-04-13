#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <map>
#include <string.h>
#include <assert.h>
#include "dirent.h"

using namespace std;

cv::Mat carve(cv::Mat image, cv::Mat imagecolor);

double** get_cost(int **img, vector<int> bounds, int rows, int cols);
vector< vector<int> > get_paths(double **img, vector<int> bounds, int rows, int cols);
vector< vector<int> > trim_paths(vector< vector<int> > paths, vector<int> bounds, int cols);

int* get_profile(int **img, int rows, int cols);
std::vector<int> get_bounds(int **img, int rows, int cols);

cv::Mat drawPaths(cv::Mat image, vector< vector<int> > paths, int width, std::vector<int> bounds);
void draw(cv::Mat image);

int WEIGHT_MAX = 100;
int FUNCTION = 2;
float INCREMENT = 0.15;

cv::Mat carve(cv::Mat image, cv::Mat imagecolor)
{

  int rows = image.rows, cols = image.cols, x, y;
  // printf("rows: %d, cols: %d\n", rows, cols);
  int **img = new int*[rows];
  for(y = 0; y < rows; y++)
  {
    img[y] = new int[cols];
    for(x = 0; x < cols; x++)
    {
      img[y][x] = image.at<uchar>(y, x);
    }
  }

  double **cost;
  vector<int> bounds = get_bounds(img, rows, cols);
  cost = get_cost(img, bounds, rows, cols);
  vector< vector<int> > paths = get_paths(cost, bounds, rows, cols);
  // trim_paths(paths, bounds, cols);

  cv::Mat mt = drawPaths(imagecolor, paths, cols, bounds);

  for(y = 0; y < rows; y++) {
    delete[] img[y];
  }
  delete[] img;

  for(y = 0; y < rows; y++) {
    delete[] cost[y];
  }
  delete[] cost;

  return mt;

}


vector< vector<int> > trim_paths(vector< vector<int> > paths, vector<int> bounds, int cols)
{
  int path_count = bounds.size()-1;
  int path, x, y, upper, lower;
  float mean, stdev;
  vector<int> freqp;
  vector< vector<int> > newpaths;

  for(path = 0; path < path_count; ++path)
  {
    lower = bounds[path]+1;
    upper = bounds[path+1];
    freqp.resize(upper-lower, 0);
    for (x = 0; x < cols; ++x) {
      assert(paths[path][x] >= lower && paths[path][x] <= upper);
      freqp[paths[path][x]-lower] += 1;
    }

    // mean = 0.0;
    // stdev = 0.0;
    // for(y = lower; y < upper; ++y)
    // {
    //   cout << freqp[y-lower] << ", ";
    //   // mean += freqp[y-lower];
    //   if (mean < freqp[y-lower]) {
    //     mean = freqp[y-lower];
    //   }
    // }
    // // mean /= (upper-lower);
    // for(y = lower; y < upper; ++y) {
    //   stdev += pow(freqp[y-lower] - mean, 2);
    // }
    // stdev = sqrt(stdev / (upper-lower));
    // // cout << "stdev: " << stdev << endl;
    // if (stdev > 100) {
    //   // paths.erase(paths.begin()+path);
    //   // newpaths.push_back(paths[path]);
    // }
  }

  return newpaths;
}


int* get_profile(int **img, int rows, int cols)
{
  int *yprof = new int[rows];
  int x, y, sumy;
  for(y = 0; y < rows; ++y)
  {
    sumy = 0;
    for(x = 0; x < cols; ++x)
    {
      sumy += img[y][x];
    }
    yprof[y] = sumy;
  }
  return yprof;
}


vector<int> get_bounds(int **img, int rows, int cols)
{
  vector<int> bounds;
  int y;
  int *prof = get_profile(img, rows, cols);

// Zero out the first and last 10% of the image since it isn't useful to us.
// The amount that needs to be zeroes out will vary depending on the
// data set you are working with.
  int five = rows * 0.1;
  for(y = 0; y < five; ++y) {
    prof[y] = 0;
    prof[rows-1-y] = 0;
  }

  int maxy, upper, lower, num, max_bounds = 100;
  for(num = 0; num < max_bounds; ++num)
  {
// find the max value in the profile projection
    maxy = 0;
    for(y = 1; y < rows; ++y) {
      if ( prof[y] > prof[maxy] ) {
        maxy = y;
      }
    }
// This zeroes out the profile around the chosen maxy.
// This makes sure that you don't pick to maxes that are too close to
// each other. This will vary by data set
// (I used the average distance between lines)
    upper = min(rows, maxy+10);
    lower = max(maxy-10, 0);
    for(y = lower; y < upper; ++y) {
      prof[y] = 0;
    }
// add the y value to the bounds list
    if (maxy != 0) {
      bounds.push_back(maxy);
    }
  }
  delete[] prof;

  sort(bounds.begin(), bounds.end());
  return bounds;
}


double interpolation(int x, int f)
{
  if(f == 0) {
    return x * 1.0;
  }else if (f == 1) {
    return (x * 1.0) / (1 + abs(x * 1.0));
  }else if (f == 2) {
    return x * x * 1.0;
  }else if (f == 3) {
    return exp(x);
  }else if (f == 4) {
    return 1 / (1 + exp(x * -1.0));
  }
}


double** get_cost(int **img, vector<int> bounds, int rows, int cols)
{

  double lowest;
  int y, x = cols-1, f = FUNCTION;
  double **cost = new double*[rows];
  for(y = 0; y < rows; ++y) {
    cost[y] = new double[cols];
    cost[y][cols-1] = interpolation(img[y][x], f);
    if ( std::find(bounds.begin(), bounds.end(), y) != bounds.end() ) {
      cost[y][cols-1] += WEIGHT_MAX;
    }
  }
// raise the cost of the first and last rows to maximum so the seam can't go there
  y = rows-1;
  for(x = cols-2; x >= 0; --x) {
    cost[0][x] = interpolation(img[0][x], f) + cost[0][x+1] + WEIGHT_MAX;
    cost[y][x] = interpolation(img[y][x], f) + cost[y][x+1] + WEIGHT_MAX;
  }

  for(x = cols-2; x >= 0; --x)
  {
    for(y = 1; y < rows-1; ++y)
    {
      if( std::find(bounds.begin(), bounds.end(), y) != bounds.end() )
      {
        cost[y][x] = cost[y][x+1] + interpolation(255, f) + WEIGHT_MAX;
      }
      else
      {
        lowest = cost[y][x+1];
        if(cost[y-1][x+1] + WEIGHT_MAX < lowest) {
          lowest = cost[y-1][x+1] + WEIGHT_MAX;
        }if(cost[y+1][x+1] + WEIGHT_MAX < lowest) {
          lowest = cost[y+1][x+1] + WEIGHT_MAX;
        }
        cost[y][x] = lowest + interpolation(img[y][x], f);
      }
    }
  }

  return cost;

}

vector< vector<int> > get_paths(double **img, vector<int> bounds, int rows, int cols)
{
  // printf("getting forptrs; rows: %d, cols: %d\n", rows, cols);
  int path_count = bounds.size()-1;
  vector<int> path(cols, 0);
  vector< vector<int> > paths(path_count, path);
  int i, x, y, bound, lower, upper, miny;
  double lowest, highest;

  for(i = 0; i < path_count; ++i)
  {
    lower = bounds[i]+1;
    upper = bounds[i+1];
    miny = lower;
    for(y = lower+1; y < upper; ++y)
    {
      if ( img[y][0] < img[miny][0] ) {
        miny = y;
      }
    }
// cout << "lower: " << lower << ", upper: " << upper << ", miny: " << miny << endl;
    assert(miny >= lower && miny <= upper);
    paths[i][0] = miny;
  }

  for(i = 0; i < path_count; ++i)
  {
    for(x = 1; x < cols; ++x)
    {
      y = paths[i][x-1];
      // lowest = img[y][x] - weight;
      lowest = img[y][x];
      paths[i][x] = y;
      if ( y > 0 && img[y-1][x] < lowest ) {
        lowest = img[y-1][x];
        paths[i][x] = y-1;
      }
      if ( y < rows-1 && img[y+1][x] < lowest ) {
        paths[i][x] = y+1;
      }
      assert(paths[i][x] >= bounds[i] && paths[i][x] <= bounds[i+1]);
    }
  }

// Remove the paths that don't have an stdev above 100
  int sum, size;
  float mean, stdev;
  vector<int> freqp, torem;
  for(i = 0; i < path_count; ++i)
  {
    lower = bounds[i];
    upper = bounds[i+1];
    size = upper-lower+1;
    freqp.clear();
    freqp.resize(size, 0);
    sum = 0;
    for (x = 0; x < cols; ++x) {
      assert(paths[i][x] >= lower && paths[i][x] <= upper);
      assert(paths[i][x]-lower >= 0 && paths[i][x]-lower < freqp.size());
      freqp[paths[i][x]-lower] += 1;
      sum += 1;
    }
    assert(sum == cols);
    // mean = (float) sum / (upper-lower+1) * 1.0;
    // mean = 0.0;
    lowest = 1000;
    highest = 0;
    for(y = lower; y < upper+1; ++y)
    {
      if (highest < freqp[y-lower]) {
        highest = freqp[y-lower];
      }
      if (lowest > freqp[y-lower]) {
        lowest = freqp[y-lower];
      }
    }

    // stdev = 0.0;
    // for(y = lower; y < upper; ++y) {
    //   stdev += pow(freqp[y-lower] - mean, 2);
    // }
    // stdev = sqrt(stdev / size);
    // cout << "stdev: " << stdev << endl;

    if (highest - lowest < 300) {
      torem.push_back(i);
      // paths.erase(paths.begin()+i);
      // newpaths.push_back(paths[i]);
    }
  }

  // cout << "Before: " << paths.size() << endl;
  for(i = torem.size()-1; i >= 0; --i)
  {
    paths.erase(paths.begin()+torem[i]);
    // printf("%d, ", torem[i]);
  }
  // cout << "\nAfter: " << paths.size() << endl;

  return paths;
}


cv::Mat drawPaths(cv::Mat image, vector< vector<int> > paths, int width, std::vector<int> bounds)
{
  int path, x;
  for(path = 0; path < paths.size(); path++)
  {
    for(x = 0; x < width; x++)
    {
      image.at<cv::Vec3b>(paths[path][x], x) = cv::Vec3b(0, 0, 255);
    }
  }
  // for(path = 0; path < bounds.size(); path++)
  // {
  //   for(x = 0; x < width; x++)
  //   {
  //     image.at<cv::Vec3b>(bounds[path], x) = cv::Vec3b(255, 0, 0);
  //   }
  // }

  return image;

}

void draw(cv::Mat image)
{

  cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
  cv::imshow("Display Image", image);
  cv::waitKey(0);

}

void write_lines(string imgfile, string outfile)
{
  cv::Mat image;
  cv::Mat imagecolor;
  try {
    image = cv::imread( imgfile.c_str(), 0 );
    imagecolor = cv::imread( imgfile.c_str(), 1 );
  }catch(...) {
    printf("No image data: %s \n", imgfile.c_str());
    return;
  }if( !image.data ) {
      printf("No image data: %s \n", imgfile.c_str());
      return;
  }

  cout << "infile: " << imgfile << endl;

  cv::Mat image2(image.rows/4, image.cols/4, image.type());
  cv::Mat image2color(imagecolor.rows/4, imagecolor.cols/4, image.type());
  cv::resize(image, image2, image2.size(), 0, 0, cv::INTER_LINEAR);
  cv::resize(imagecolor, image2color, image2.size(), 0, 0, cv::INTER_LINEAR);

  cv::Mat img2;
  img2 = carve(image2, image2color);

  cv::Mat image3;
  cv::transpose(image2, image3);
  cv::Mat img3;
  cv::transpose(img2, img3);
  img3 = carve(image3, img3);

  cv::Mat outImg;
  cv::transpose(img3, outImg);
  // outImg = img2;

  // vector<int> compression_params;
  // compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
  // compression_params.push_back(9);
  // cout << cv::imwrite(outfile.c_str(), outImg, compression_params);
  cout << cv::imwrite(outfile.c_str(), outImg);
  cout << "outfile: " << outfile << endl;
}

int main(int argc, char** argv )
{

  if (argc < 3 || argc > 5) {
    printf("usage: EnergyFinder.out <Images_Folder> <Output_Folder> <Diagnol_Penalty> <Interpolation_Function\n");
    return -1;
  }
  if (argc > 3) {
    WEIGHT_MAX = atoi(argv[3]);
  }if (argc > 4) {
    FUNCTION = atoi(argv[4]);
  }

  DIR *dir;
  struct dirent *ent;
  string input;
  string output;
  if( (dir = opendir(argv[1])) != NULL ) {
    /* print all the files and directories within directory */
    while( (ent = readdir(dir)) != NULL ) {
      input = argv[1] + (string) ent->d_name;
      output = argv[2] + (string) ent->d_name;
      write_lines(input, output);
    }
    closedir (dir);
  }else {
    /* could not open directory */
    perror ("");
    return EXIT_FAILURE;
  }

  // write_lines(argv[1], argv[2]);

  return 0;
}
