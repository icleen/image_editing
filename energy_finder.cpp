#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;

cv::Mat carve(cv::Mat image);
int** get_backptrs(int **img, int rows, int cols);
cv::Mat drawPaths(cv::Mat image, int** paths, int path_count);
void draw(cv::Mat image);


cv::Mat carve(cv::Mat image)
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
  cv::Mat mt = drawPaths(image, ptrs, rows);

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

  // cout << "Cost Matrix:\n";
  // for(y = 0; y < rows; y++)
  // {
  //   for(x = 0; x < cols; x++)
  //   {
  //     cout << cost[y][x] << ",";
  //   }
  //   cout << endl;
  // }
  // cout << endl;

  for(y = 0; y < rows; y++) {
    delete[] cost[y];
  }
  delete[] cost;

  return backptrs;
}


cv::Mat drawPaths(cv::Mat image, int** paths, int path_count)
{

  printf("rows: %d, cols: %d\n", image.rows, image.cols);
  for(int path = 0; path < path_count; path++)
  {
    cout << "path: " << path << ":";
    for(int x = 0; x < image.cols; x++)
    {
      cout << "y: " << paths[path][x] << ", x: " << x << ", ";
      image.at<cv::Vec3b>(paths[path][x], x) = cv::Vec3b(0, 0, 255);
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
    image = cv::imread( argv[1], cv::IMREAD_COLOR );
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

    cv::Mat image2(image.rows/2, image.cols/2, image.type());
    // cv::Mat image2(100, 120, image.type());
    cv:resize(image, image2, image2.size(), 0, 0, cv::INTER_LINEAR);

    printf("rows: %d, cols: %d\n", image2.rows, image2.cols);

    cv::Mat img2;
    printf("Carving\n");
    img2 = carve(image2);

    printf("done carving\n");
    printf("rows: %d, cols: %d\n", img2.rows, img2.cols);

    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", img2);
    cv::waitKey(0);

    return 0;
}
