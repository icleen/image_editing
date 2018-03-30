#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <map>
#include <string.h>
#include "dirent.h"

using namespace std;

cv::Mat carve(cv::Mat image, cv::Mat imagecolor);
double** get_cost(int **img, int rows, int cols);
double** get_costfor(int **img, int rows, int cols);
int** get_backptrs(double **img, int **paths, int rows, int cols, int &path_count);
int** get_forptrs(double **img, int rows, int cols);
cv::Mat drawPaths(cv::Mat image, int** paths, int path_count);
cv::Mat drawPaths(cv::Mat image, int** paths, int path_count, int height);
cv::Mat drawPaths(cv::Mat image, std::vector< std::vector<int> > paths);
void draw(cv::Mat image);
std::vector< std::vector<int> > findMostFreq(int** paths, int path_count, int cols);

int WEIGHT_MAX = 20;
int FUNCTION = 0;
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

  double **cost = get_cost(img, rows, cols);
  int **forptrs = get_forptrs(cost, rows, cols);
  int path_count = rows;

  // for(y = 0; y < rows; y++) {
  //   delete[] cost[y];
  // }
  // delete[] cost;
  // cost = get_costfor(img, rows, cols);
  // int **backptrs = get_backptrs(cost, forptrs, rows, cols, path_count);
  // cv::Mat mt = drawPaths(imagecolor, backptrs, path_count, cols);

  cv::Mat mt = drawPaths(imagecolor, forptrs, path_count, cols);

  for(y = 0; y < rows; y++) {
    delete[] img[y];
  }
  delete[] img;

  for(y = 0; y < rows; y++) {
    delete[] cost[y];
  }
  delete[] cost;

  for(y = 0; y < rows; y++) {
    delete[] forptrs[y];
  }
  delete[] forptrs;

  // for(y = 0; y < path_count; y++) {
  //   delete[] backptrs[y];
  // }
  // delete[] backptrs;

  return mt;

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

double** get_cost(int **img, int rows, int cols)
{

  double lowest;
  int five = rows * 0.05;
  int y, x = cols-1, f = FUNCTION;
  double **cost = new double*[rows];
  for(y = 0; y < rows; ++y) {
    cost[y] = new double[cols];
    cost[y][cols-1] = interpolation(img[y][x], f);
    if (y % five == 0) {
      cost[y][cols-1] += WEIGHT_MAX;
    }
  }
  y = rows-1;
  for(x = cols-2; x >= 0; --x) {
    cost[0][x] = interpolation(img[0][x], f) + cost[0][x+1] + WEIGHT_MAX;
    cost[y][x] = interpolation(img[y][x], f) + cost[y][x+1] + WEIGHT_MAX;
  }

  int section = 1;

  for(x = cols-2; x >= 0; --x)
  {
    for(y = 1; y < rows-1; ++y)
    {
      if (y % five == 0)
      {
        cost[y][x] = cost[y][x+1] + interpolation(255, f) + WEIGHT_MAX;
      }
      else
      {
        lowest = cost[y][x+1];
        if(cost[y-1][x+1] + WEIGHT_MAX < lowest) {
          lowest = cost[y-1][x+1];
        }if(cost[y+1][x+1] + WEIGHT_MAX < lowest) {
          lowest = cost[y+1][x+1];
        }
        cost[y][x] = lowest + interpolation(img[y][x], f);
      }
    }
  }

  x = cols/2;
  cout << "Cost Matrix:\n";
  for(y = 0; y < rows; ++y) {
    cout << ", " << cost[y][x];
  }
  cout << endl;

  return cost;

}

double** get_costfor(int **img, int rows, int cols)
{

  double lowest;
  int y, x = 0, f = FUNCTION;
  double **cost = new double*[rows];
  for(y = 0; y < rows; ++y) {
    cost[y] = new double[cols];
    cost[y][0] = interpolation(img[y][x], f);
  }
  y = rows-1;
  for(x = 1; x < cols; ++x) {
    cost[0][x] = interpolation(img[0][x], f) + cost[0][x-1] + WEIGHT_MAX;
    cost[y][x] = interpolation(img[y][x], f) + cost[y][x-1] + WEIGHT_MAX;
  }

  for(x = 1; x < cols; ++x)
  {
    for(y = 1; y < rows-1; ++y)
    {
      lowest = cost[y][x-1];
      if(cost[y-1][x-1] + WEIGHT_MAX < lowest) {
        lowest = cost[y-1][x-1];
      }if(cost[y+1][x-1] + WEIGHT_MAX < lowest) {
        lowest = cost[y+1][x-1];
      }
      cost[y][x] = lowest + interpolation(img[y][x], f);
      // cost[y][x] = interpolation(img[y][x], f);
    }
  }

  return cost;

}

int** get_forptrs(double **img, int rows, int cols)
{
  // printf("getting forptrs; rows: %d, cols: %d\n", rows, cols);
  double lowest = 0;
  int path = 0, path_count = rows, x = 0, y = 0;
  // float weight = 0.0, start = cols * 0.05;
  int **forptrs = new int*[path_count];
  for(path = 0; path < path_count; path++) {
    forptrs[path] = new int[cols];
    forptrs[path][x] = path;
  }

  for(x = 1; x < cols; x++)
  {
    // if (x > start && weight < WEIGHT_MAX) // ignores the first 5% of the image
    //   weight += INCREMENT;
    for(path = 0; path < rows; path++)
    {
      y = forptrs[path][x-1];
      // lowest = img[y][x] - weight;
      lowest = img[y][x];
      forptrs[path][x] = y;
      if ( y > 0 && img[y-1][x] < lowest ) {
        lowest = img[y-1][x];
        forptrs[path][x] = y-1;
      }
      if ( y < rows-1 && img[y+1][x] < lowest ) {
        forptrs[path][x] = y+1;
      }
    }
  }
  return forptrs;
}

int** get_backptrs(double **img, int **paths, int rows, int cols, int &path_count)
{
  int x = cols-1, y = 0, path = 0, lowest = 0;
  std::map<int, int> uniqs;
  for(path = 0; path < rows; path++) {
    uniqs[paths[path][x]] = 0;
  }
  for(path = 0; path < rows; path++) {
    uniqs[paths[path][x]] += 1;
  }

  for(std::map<int,int>::iterator iter = uniqs.begin(); iter != uniqs.end(); ++iter) {
    // printf("%d, ", iter->second);
    if (iter->second < 1) {
      uniqs.erase(iter);
    }
  }
  // cout << endl;

  path_count = uniqs.size();
  int **backptrs = new int*[path_count];
  std::map<int,int>::iterator iter = uniqs.begin();
  cout << "Number of unique paths: " << path_count << endl;
  for(path = 0; path < path_count; ++path)
  {
    backptrs[path] = new int[cols];
    backptrs[path][cols-1] = iter->first;
    for(x = cols-2; x >= 0; --x)
    {
      y = backptrs[path][x+1];
      lowest = img[y][x] - (WEIGHT_MAX);
      backptrs[path][x] = y;
      if ( y > 0 && img[y-1][x] < lowest ) {
        lowest = img[y-1][x];
        backptrs[path][x] = y-1;
      }
      if ( y < rows-1 && img[y+1][x] < lowest ) {
        backptrs[path][x] = y+1;
      }
    }
    ++iter;
  }
  // cout << endl;

  return backptrs;
}

std::vector< std::vector<int> > findMostFreq(int** paths, int path_count, int cols)
{

  int **img = new int*[path_count+2];
  int path, x, y;
  for(y = 0; y < path_count+2; y++) {
    img[y] = new int[cols] {};
  }

  for(path = 0; path < path_count; path++)
  {
    for(x = cols-1; x > 0; x--)
    {
      img[paths[path][x]][x-1] += 1;
    }
  }

  std::vector< std::vector<int> > freq(cols-1);

  for(x = 0; x < cols-1; x++)
  {
    for(y = 0; y < path_count+2; y++)
    {
      if(img[y][x] > 2) {
        freq[x].push_back(y);
      }
    }
  }
  return freq;

}

cv::Mat drawPaths(cv::Mat image, std::vector< std::vector<int> > paths)
{

  int cols = paths[0].size();
  for(int x = 0; x < paths.size(); x++)
  {
    for(int y = 0; y < paths[x].size(); y++)
    {
      image.at<cv::Vec3b>(paths[x][y], x) = cv::Vec3b(0, 0, 255);
      // image.at<uchar>(paths[x][y], x) = uchar(255);
    }
  }
  return image;

}


cv::Mat drawPaths(cv::Mat image, int** paths, int path_count, int width)
{
  int five = path_count * 0.05;
  // printf("rows: %d, cols: %d\n", image.rows, image.cols);
  for(int path = 0; path < path_count; path++)
  {
    // cout << "path: " << path << ":\n";
    image.at<cv::Vec3b>(path, 0) = cv::Vec3b(0, 0, 255);
    for(int x = 0; x < width; x++)
    {
      // cout << "y: " << paths[path][x] << ", x: " << x << ", ";
      image.at<cv::Vec3b>(paths[path][x], x) = cv::Vec3b(0, 0, 255);
      if (path % five == 0) {
        image.at<cv::Vec3b>(path, x) = cv::Vec3b(255, 0, 0);
      }
    }
    // cout << "\n";
  }
  return image;

}


cv::Mat drawPaths(cv::Mat image, int** paths, int path_count)
{

  // printf("rows: %d, cols: %d\n", image.rows, image.cols);
  for(int path = 0; path < path_count; path++)
  {
    // cout << "path: " << path << ":\n";
    image.at<cv::Vec3b>(path, 0) = cv::Vec3b(0, 0, 255);
    for(int x = 1; x < image.cols-1; x++)
    {
      // cout << "y: " << paths[path][x] << ", x: " << x << ", ";
      image.at<cv::Vec3b>(paths[path][x-1], x) = cv::Vec3b(0, 0, 255);
    }
    // cout << "\n";
  }
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
    printf("No image data \n");
    return;
  }if( !image.data ) {
      printf("No image data \n");
      return;
  }

  cout << "infile: " << imgfile << endl;

  cv::Mat image2(image.rows/4, image.cols/4, image.type());
  cv::Mat image2color(imagecolor.rows/4, imagecolor.cols/4, image.type());
  cv::resize(image, image2, image2.size(), 0, 0, cv::INTER_LINEAR);
  cv::resize(imagecolor, image2color, image2.size(), 0, 0, cv::INTER_LINEAR);

  cv::Mat img2;
  img2 = carve(image2, image2color);

  // cv::Mat image3;
  // cv::transpose(image2, image3);
  // cv::Mat img3;
  // cv::transpose(img2, img3);
  // img3 = carve(image3, img3);

  cv::Mat outImg;
  // cv::transpose(img3, outImg);
  outImg = img2;

  vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(9);
  cv::imwrite(outfile.c_str(), outImg, compression_params);
  cout << "outfile: " << outfile << endl;
}

int main(int argc, char** argv )
{

  // printf("argc: %d\n", argc);
  if (argc == 4) {
    WEIGHT_MAX = atoi(argv[3]);
  }else if (argc == 5) {
    WEIGHT_MAX = atoi(argv[3]);
    FUNCTION = atoi(argv[4]);
  }else if (argc < 3 || argc > 5) {
    printf("usage: EnergyFinder.out <Images_Folder> <Output_Folder> <Diagnol_Penalty>\n");
    return -1;
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
