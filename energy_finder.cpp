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

cv::Mat carve(cv::Mat image, cv::Mat imagecolor, float percent);

double** get_cost(int **img, vector<int> bounds, int rows, int cols);
vector< vector<int> > get_paths(double **img, vector<int> bounds, int rows, int cols);
vector< vector<int> > trim_paths(vector< vector<int> > paths, vector<int> bounds, int cols);

double** get_cost_ver(int **img, vector<int> bounds, int rows, int cols);
vector< vector<int> > get_paths_ver(double **img, vector<int> bounds, int rows, int cols);

int* get_profile(int **img, int rows, int cols);
std::vector<int> get_bounds(int **img, int rows, int cols, float percent);

int* get_profile_ver(int **img, int rows, int cols);
std::vector<int> get_bounds_ver(int **img, int rows, int cols, float percent);

vector<Point> match_points(vector< vector<int> > paths, vector< vector<int> > paths2);

cv::Mat drawPaths(cv::Mat image, vector< vector<int> > paths, int width, std::vector<int> bounds);
cv::Mat drawPaths_ver(cv::Mat image, vector< vector<int> > paths, int height, std::vector<int> bounds);
void draw(cv::Mat image);

int WEIGHT_MAX = 100;
int FUNCTION = 2;
float INCREMENT = 0.15;

cv::Mat carve(cv::Mat image, cv::Mat imagecolor, float percent)
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
  vector<int> bounds = get_bounds(img, rows, cols, percent);
  cost = get_cost(img, bounds, rows, cols);
  vector< vector<int> > paths = get_paths(cost, bounds, rows, cols);

  for(y = 0; y < rows; y++) {
    delete[] cost[y];
  }
  delete[] cost;

  bounds = get_bounds_ver(img, rows, cols, percent/2.0);
  cost = get_cost_ver(img, bounds, rows, cols);
  vector< vector<int> > paths2 = get_paths_ver(cost, bounds, rows, cols);

  points = match_points(paths, paths2);

  cv::Mat mt = drawPaths(imagecolor, paths, cols, bounds);
  mt = drawPaths_ver(mt, paths2, rows, bounds);

  for(y = 0; y < rows; y++) {
    delete[] img[y];
  }
  delete[] img;

  return mt;

}


vector<Point> match_points(vector< vector<int> > paths, vector< vector<int> > paths2)
{
  vector<Point> points;
  int i, j = 0, y, x;
  for(i = 0; i < paths.size(); i++)
  {
    for(x = 0; x < paths[i].size(); x++)
    {
      y = paths[i][x];
      if (paths2[j][y] == x) {
        // found a point
        points.push_back(Point(x, y));
        ++j;
      }
    }
  }
  return points;
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


vector<int> get_bounds(int **img, int rows, int cols, float percent)
{
  vector<int> bounds;
  int y;
  int *prof = get_profile(img, rows, cols);

// Zero out the first and last 10% of the image since it isn't useful to us.
// The amount that needs to be zeroes out will vary depending on the
// data set you are working with.
  int five = rows * percent;
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


double** get_cost(int **img, vector<int> bounds, int rows, int cols)
{

  double lowest;
  int y, x = cols-1, f = FUNCTION;
  double **cost = new double*[rows];
  for(y = 0; y < rows; ++y) {
    cost[y] = new double[cols];
    cost[y][x] = interpolation(img[y][x], f);
    if ( std::find(bounds.begin(), bounds.end(), y) != bounds.end() ) {
      cost[y][x] += WEIGHT_MAX;
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

  // find the paths that move around the most
  int sum, low = 0.25 * cols, high = 0.75 * cols;
  vector<int> torem;
  float limit = 0.3 * (high - low);
  // printf("limit: %f\n", limit);
  for(i = 0; i < path_count; ++i)
  {
    sum = 0;
    for (x = low; x < high; ++x) {
      if(paths[i][x] != paths[i][x+1])
        sum += 1;
    }
    // printf("path %d sum: %d\n", i, sum);
    if (sum > limit) {
      torem.push_back(i);
    }
    // if (sum > high) {
    //   high = sum;
    // }
  }

  // printf("half high: %f\n", float(high/2.0));
  // for(i = 0; i < path_count; ++i)
  // {
  //   if (sum > limit) {
  //     torem.push_back(i);
  //   }
  // }

  cout << "Before: " << paths.size() << endl;
  for(i = torem.size()-1; i >= 0; --i)
  {
    paths.erase(paths.begin()+torem[i]);
  }
  cout << "After: " << paths.size() << endl;

  return paths;
}


int* get_profile_ver(int **img, int rows, int cols)
{
  int *xprof = new int[cols];
  int x, y, sumx;
  for(x = 0; x < cols; ++x)
  {
    sumx = 0;
    for(y = 0; y < rows; ++y)
    {
      sumx += img[y][x];
    }
    xprof[x] = sumx;
  }
  return xprof;
}


vector<int> get_bounds_ver(int **img, int rows, int cols, float percent)
{
  vector<int> bounds;
  int x;
  printf("getting profile\n");
  int *prof = get_profile_ver(img, rows, cols);
  printf("got profile\n");
  int five = cols * percent;
  for(x = 0; x < five; ++x) {
    prof[x] = 0;
    prof[rows-1-x] = 0;
  }

  int maxx, upper, lower, num, max_bounds = 100;
  for(num = 0; num < max_bounds; ++num)
  {
    maxx = 0;
    for(x = 1; x < cols; ++x) {
      if ( prof[x] > prof[maxx] ) {
        maxx = x;
      }
    }
    upper = min(cols, maxx+10);
    lower = max(maxx-10, 0);
    for(x = lower; x < upper; ++x) {
      prof[x] = 0;
    }
    if (maxx != 0) {
      bounds.push_back(maxx);
    }
  }
  delete[] prof;

  sort(bounds.begin(), bounds.end());
  return bounds;
}


double** get_cost_ver(int **img, vector<int> bounds, int rows, int cols)
{

  double lowest;
  int y, x = cols-1, f = FUNCTION;
  double **cost = new double*[rows];
  for(y = 0; y < rows; ++y) {
    cost[y] = new double[cols];
  }
  y = rows-1;
  for(x = 0; x < cols; ++x) {
    cost[y][x] = interpolation(img[y][x], f);
    if ( std::find(bounds.begin(), bounds.end(), x) != bounds.end() ) {
      cost[y][x] += WEIGHT_MAX;
    }
  }
// raise the cost of the first and last cols to maximum so the seam can't go there
  x = cols-1;
  for(y = rows-2; y >= 0; --y) {
    cost[y][0] = interpolation(img[0][x], f) + cost[y+1][0] + WEIGHT_MAX;
    cost[y][x] = interpolation(img[y][x], f) + cost[y+1][x] + WEIGHT_MAX;
  }

  for(y = rows-2; y >= 0; --y)
  {
    for(x = 1; x < cols-1; ++x)
    {
      if( std::find(bounds.begin(), bounds.end(), x) != bounds.end() )
      {
        cost[y][x] = cost[y+1][x] + interpolation(255, f) + WEIGHT_MAX;
      }
      else
      {
        lowest = cost[y+1][x];
        if(cost[y+1][x-1] + WEIGHT_MAX < lowest) {
          lowest = cost[y+1][x-1] + WEIGHT_MAX;
        }if(cost[y+1][x+1] + WEIGHT_MAX < lowest) {
          lowest = cost[y+1][x+1] + WEIGHT_MAX;
        }
        cost[y][x] = lowest + interpolation(img[y][x], f);
      }
    }
  }

  return cost;

}

vector< vector<int> > get_paths_ver(double **img, vector<int> bounds, int rows, int cols)
{
  // printf("getting forptrs; rows: %d, cols: %d\n", rows, cols);
  int path_count = bounds.size()-1;
  vector<int> path(rows, 0);
  vector< vector<int> > paths(path_count, path);
  int i, x, y, bound, lower, upper, minx;
  double lowest, highest;

  for(i = 0; i < path_count; ++i)
  {
    lower = bounds[i]+1;
    upper = bounds[i+1];
    minx = lower;
    for(x = lower+1; x < upper; ++x)
    {
      if ( img[0][x] < img[0][minx] ) {
        minx = x;
      }
    }
// cout << "lower: " << lower << ", upper: " << upper << ", minx: " << minx << endl;
    assert(minx >= lower && minx <= upper);
    paths[i][0] = minx;
  }

  for(i = 0; i < path_count; ++i)
  {
    for(y = 1; y < rows; ++y)
    {
      x = paths[i][y-1];
      lowest = img[y][x];
      paths[i][y] = x;
      if ( x > 0 && img[y][x-1] < lowest ) {
        lowest = img[y][x-1];
        paths[i][y] = x-1;
      }
      if ( x < cols-1 && img[y][x+1] < lowest ) {
        paths[i][y] = x+1;
      }
      assert(paths[i][y] >= bounds[i] && paths[i][y] <= bounds[i+1]);
    }
  }

  // find the paths that move around the most
  int sum, low = 0.25 * rows, high = 0.75 * rows;
  vector<int> torem;
  float limit = 0.3 * (high - low);
  // printf("limit: %f\n", limit);
  for(i = 0; i < path_count; ++i)
  {
    sum = 0;
    for (y = low; y < high; ++y) {
      if(paths[i][y] != paths[i][y+1])
        sum += 1;
    }
    // printf("path %d sum: %d\n", i, sum);
    if (sum > limit) {
      torem.push_back(i);
    }
    // if (sum > high) {
    //   high = sum;
    // }
  }

  // printf("half high: %f\n", float(high/2.0));
  // for(i = 0; i < path_count; ++i)
  // {
  //   if (sum > limit) {
  //     torem.push_back(i);
  //   }
  // }

  cout << "Before: " << paths.size() << endl;
  for(i = torem.size()-1; i >= 0; --i)
  {
    paths.erase(paths.begin()+torem[i]);
  }
  cout << "After: " << paths.size() << endl;

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

cv::Mat drawPaths_ver(cv::Mat image, vector< vector<int> > paths, int height, std::vector<int> bounds)
{
  int path, y;
  for(path = 0; path < paths.size(); path++)
  {
    for(y = 0; y < height; y++)
    {
      image.at<cv::Vec3b>(y, paths[path][y]) = cv::Vec3b(0, 0, 255);
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

  float xpercent = 0.15, ypercent = 0.1;

  cout << "infile: " << imgfile << endl;

  cv::Mat image2(image.rows/4, image.cols/4, image.type());
  cv::Mat image2color(imagecolor.rows/4, imagecolor.cols/4, image.type());
  cv::resize(image, image2, image2.size(), 0, 0, cv::INTER_LINEAR);
  cv::resize(imagecolor, image2color, image2.size(), 0, 0, cv::INTER_LINEAR);

  cv::Mat img2;
  img2 = carve(image2, image2color, xpercent);

  // cv::Mat image3;
  // cv::transpose(image2, image3);
  // cv::Mat img3;
  // cv::transpose(img2, img3);
  // img3 = carve(image3, img3, ypercent);
  //
  // cv::Mat outImg;
  // cv::transpose(img3, outImg);

  // vector<int> compression_params;
  // compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
  // compression_params.push_back(9);
  // cout << cv::imwrite(outfile.c_str(), outImg, compression_params);
  // cout << cv::imwrite(outfile.c_str(), outImg);
  cout << cv::imwrite(outfile.c_str(), img2);
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
