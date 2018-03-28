#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

using namespace std;

cv::Mat carve(cv::Mat image, cv::Mat imagecolor);
int** get_backptrs(int **img, int rows, int cols);
cv::Mat drawPaths(cv::Mat image, int** paths, int path_count);
cv::Mat drawPaths(cv::Mat image, std::vector< std::vector<int> > paths);
void draw(cv::Mat image);
std::vector< std::vector<int> > findMostFreq(int** paths, int path_count, int cols);


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

  int **ptrs = get_backptrs(img, rows, cols);
  printf("finding most frequent\n");
  std::vector< std::vector<int> > freq = findMostFreq(ptrs, rows, cols);
  printf("drawing paths\n");
  cv::Mat mt = drawPaths(imagecolor, freq);
  printf("paths drawn\n");

  for(y = 0; y < rows; y++) {
    delete[] img[y];
  }
  delete[] img;

  for(y = 0; y < rows; y++) {
    delete[] ptrs[y];
  }
  delete[] ptrs;

  return mt;

}

int** get_forptrs(int **img, int rows, int cols)
{
  // printf("getting forptrs; rows: %d, cols: %d\n", rows, cols);
  // int lowest = 0, x = 0, y = 0;
  // // int **backptrs = new int*[rows];
  // int **forptrs = new int*[cols-1];
  // for(x = 0; x < cols-1; x++) {
  //   forptrs[x] = new int[rows];
  // }
  //
  // for(x = 0; x < cols-1; x++)
  // {
  //   for(y = 0; y < rows; y++)
  //   {
  //     lowest = cost[y][x+1];
  //     forptrs[y][x] = y;
  //     if ( y > 0 && cost[y-1][x+1] < lowest ) {
  //       lowest = cost[y-1][x+1];
  //       forptrs[y][x] = y-1;
  //     }
  //     if ( y < rows-1 && cost[y+1][x+1] < lowest ) {
  //       lowest = cost[y+1][x+1];
  //       forptrs[y][x] = y+1;
  //     }
  //   }
  // }
  // return forptrs;
  return NULL;
}

int** get_backptrs(int **img, int rows, int cols)
{
  printf("carving seams; rows: %d, cols: %d\n", rows, cols);
  int lowest = 0, x = 0, y = 0;
  int **backptrs = new int*[rows];
  int **cost = new int*[rows];
  // set the cost of the first col to the initiprintf("rows: %d, cols: %d\n", rows, cols);al pixel values
  for(y = 0; y < rows; y++) {
    backptrs[y] = new int[cols] {};
    cost[y] = new int[cols] {};
    cost[y][x] = img[y][x];
  }
  y = 0;
  for(x = 1; x < cols; x++) {
    cost[y][x] = img[y][x] + cost[y][x-1];
    cost[rows-1][x] = img[rows-1][x] + cost[rows-1][x-1];
  }

  // printf("creating cost matrix\n");
  for(x = 1; x < cols; x++)
  {
    for(y = 1; y < rows-1; y++)
    {
// find the lowest cost to get to the current pixel
      lowest = cost[y][x-1];
      backptrs[y][x] = y;
      if (cost[y-1][x-1] < lowest) {
        lowest = cost[y-1][x-1];
        backptrs[y][x] = y-1;
      }
      if (cost[y+1][x-1] < lowest) {
        lowest = cost[y+1][x-1];
        backptrs[y][x] = y+1;
      }
      cost[y][x] = lowest + img[y][x];
    }
  }

  for(y = 0; y < rows; y++) {
    delete[] cost[y];
  }
  delete[] cost;

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


cv::Mat drawPaths(cv::Mat image, int** paths, int path_count)
{

  // printf("rows: %d, cols: %d\n", image.rows, image.cols);
  for(int path = 0; path < path_count; path++)
  {
    cout << "path: " << path << ":\n";
    image.at<cv::Vec3b>(path, 0) = cv::Vec3b(0, 0, 255);
    for(int x = 1; x < image.cols-1; x++)
    {
      cout << "y: " << paths[path][x] << ", x: " << x << ", ";
      image.at<cv::Vec3b>(paths[path][x-1], x) = cv::Vec3b(0, 0, 255);
    }
    cout << "\n";
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
    if ( argc != 2 )
    {
        printf("usage: seamcarver.out <Image_Path>\n");
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
    cv::imwrite("overdrawn.png", img2, compression_params);
    // cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    // cv::imshow("Display Image", img2);
    // cv::waitKey(0);

    return 0;
}
