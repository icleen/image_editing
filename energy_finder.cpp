#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <map>

using namespace std;

cv::Mat carve(cv::Mat image, cv::Mat imagecolor);
int** get_backptrs(int **img, int **paths, int rows, int cols, int &path_count);
int** get_forptrs(int **img, int rows, int cols);
cv::Mat drawPaths(cv::Mat image, int** paths, int path_count);
cv::Mat drawPaths(cv::Mat image, int** paths, int path_count, int height);
cv::Mat drawPaths(cv::Mat image, std::vector< std::vector<int> > paths);
void draw(cv::Mat image);
std::vector< std::vector<int> > findMostFreq(int** paths, int path_count, int cols);

int WEIGHT_MAX = 20;
float INCREMENT = 0.15;

cv::Mat carve(cv::Mat image, cv::Mat imagecolor)
{

  int rows = image.rows, cols = image.cols, x, y;
  printf("rows: %d, cols: %d\n", rows, cols);
  int **img = new int*[rows];
  for(y = 0; y < rows; y++)
  {
    img[y] = new int[cols];
    for(x = 0; x < cols; x++)
    {
      img[y][x] = image.at<uchar>(y, x);
    }
  }

  int **forptrs = get_forptrs(img, rows, cols);
  int path_count = rows;
  int **backptrs = get_backptrs(img, forptrs, rows, cols, path_count);
  printf("drawing paths\n");
  cv::Mat mt = drawPaths(imagecolor, backptrs, path_count, cols);
  printf("paths drawn\n");

  for(y = 0; y < rows; y++) {
    delete[] img[y];
  }
  delete[] img;

  for(y = 0; y < rows; y++) {
    delete[] forptrs[y];
  }
  delete[] forptrs;

  for(y = 0; y < path_count; y++) {
    delete[] backptrs[y];
  }
  delete[] backptrs;

  return mt;

}

int** get_forptrs(int **img, int rows, int cols)
{
  printf("getting forptrs; rows: %d, cols: %d\n", rows, cols);
  int lowest = 0, path = 0, path_count = rows, x = 0, y = 0;
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
      lowest = img[y][x] - WEIGHT_MAX;
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

int** get_backptrs(int **img, int **paths, int rows, int cols, int &path_count)
{
  int x = cols-1, y = 0, path = 0, lowest = 0;
  std::map<int, int> uniqs;
  for(path = 0; path < rows; path++)
  {
    if(uniqs.find(paths[path][x]) != uniqs.end()) {
      uniqs[paths[path][x]] = 1;
    }else {
      uniqs[paths[path][x]] += 1;
    }
  }

  path_count = uniqs.size();
  int **backptrs = new int*[path_count];
  std::map<int,int>::iterator iter = uniqs.begin();
  cout << "lasts:" << '\n';
  for(path = 0; path < path_count; ++path)
  {
    backptrs[path] = new int[cols];
    backptrs[path][cols-1] = iter->first;
    cout << ", " << iter->first;
    for(x = cols-2; x >= 0; --x)
    {
      y = backptrs[path][x+1];
      lowest = img[y][x] - WEIGHT_MAX;
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
  cout << endl;

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

  // printf("rows: %d, cols: %d\n", image.rows, image.cols);
  for(int path = 0; path < path_count; path++)
  {
    // cout << "path: " << path << ":\n";
    image.at<cv::Vec3b>(path, 0) = cv::Vec3b(0, 0, 255);
    for(int x = 0; x < width; x++)
    {
      // cout << "y: " << paths[path][x] << ", x: " << x << ", ";
      image.at<cv::Vec3b>(paths[path][x], x) = cv::Vec3b(0, 0, 255);
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

int main(int argc, char** argv )
{

  // printf("argc: %d\n", argc);
  if (argc == 4) {
    WEIGHT_MAX = atoi(argv[3]);
  }else if (argc < 3 || argc > 4) {
    printf("usage: seamcarver.out <Image_Path> <Output_Path> <Diagnol_Penalty>\n");
    return -1;
  }
  cv::Mat image;
  cv::Mat imagecolor;
  image = cv::imread( argv[1], 0 );
  imagecolor = cv::imread( argv[1], 1 );
  if ( !image.data )
  {
      printf("No image data \n");
      return -1;
  }

  cv::Mat image2(image.rows/4, image.cols/4, image.type());
  cv::Mat image2color(imagecolor.rows/4, imagecolor.cols/4, image.type());
  // cv::Mat image2(100, 120, image.type());
  cv::resize(image, image2, image2.size(), 0, 0, cv::INTER_LINEAR);
  cv::resize(imagecolor, image2color, image2.size(), 0, 0, cv::INTER_LINEAR);

  printf("rows: %d, cols: %d\n", image2.rows, image2.cols);

  cv::Mat img2;
  printf("Carving\n");
  img2 = carve(image2, image2color);

  printf("done carving\n");
  printf("rows: %d, cols: %d\n", img2.rows, img2.cols);


  vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(9);
  cv::imwrite(argv[2], img2, compression_params);
  // cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
  // cv::imshow("Display Image", img2);
  // cv::waitKey(0);

  return 0;
}
