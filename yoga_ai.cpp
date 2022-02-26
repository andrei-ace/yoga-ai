#include <iostream>
#include <memory>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/openpose.hpp>


using namespace std;
using Result = vitis::ai::OpenPoseResult::PosePoint;


static cv::Mat process_result(cv::Mat& image, vitis::ai::OpenPoseResult results,
                              bool is_jpeg) {
  vector<vector<int>> limbSeq = {{0, 1},  {1, 2},   {2, 3},  {3, 4}, {1, 5},
                                 {5, 6},  {6, 7},   {1, 8},  {8, 9}, {9, 10},
                                 {1, 11}, {11, 12}, {12, 13}};

  for (size_t k = 1; k < results.poses.size(); ++k) {
    for (size_t i = 0; i < results.poses[k].size(); ++i) {
      if (results.poses[k][i].type == 1) {
        cv::circle(image, results.poses[k][i].point, 5, cv::Scalar(0, 255, 0),-1);
      }
    }
    for (size_t i = 0; i < limbSeq.size(); ++i) {
      Result a = results.poses[k][limbSeq[i][0]];
      Result b = results.poses[k][limbSeq[i][1]];
      if (a.type == 1 && b.type == 1) {
        cv::line(image, a.point, b.point, cv::Scalar(255, 0, 0), 3, 4);
      }
    }
  }
  return image;
}

int main(int argc, char *argv[])
{
    auto image = cv::imread("yoga.jpg");
    if (image.empty())
    {
        std::cerr << "cannot load image" << std::endl;
        abort();
    }
    auto det = vitis::ai::OpenPose::create("openpose_pruned_0_3");
    int width = det->getInputWidth();
    int height = det->getInputHeight();

    auto results = det->run(image);
    image = process_result(image, results, true);
    for (size_t k = 1; k < results.poses.size(); ++k) {
        stringstream temp;
        temp << "[";
        for (size_t i = 0; i < results.poses[k].size(); ++i) {
            if(results.poses[k][i].type != 1) {
                std::cerr << "not all anchor points valid" << std::endl;
                abort();
            }
            if (i<results.poses[k].size()-1) {
                temp << results.poses[k][i].point<<", ";
            } 
            else {
                temp << results.poses[k][i].point;
            }
        }
        temp << "]";
        std::cout << temp.str() << std::endl;
    }
    
    bool check = imwrite("result_yoga.jpg", image);
    if (check == false) {
        std::cerr << "cannot save image" << std::endl;
        abort();
    }
}