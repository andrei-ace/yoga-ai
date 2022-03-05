#include <iostream>
#include <memory>
#include <vector>
#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/openpose.hpp>

using namespace std;
using namespace cv;
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


int main()
{
    VideoCapture cap(-1);

    // Check if camera opened successfully
    if (!cap.isOpened())
    {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    auto det = vitis::ai::OpenPose::create("openpose_pruned_0_3");
    int width = det->getInputWidth();
    int height = det->getInputHeight();


    while (1)
    {

        Mat frame;
        // Capture frame-by-frame
        cap >> frame;

        // If the frame is empty, break immediately
        if (frame.empty())
            break;

        auto results = det->run(frame);
        frame = process_result(frame, results, true);    

        // Display the resulting frame
        imshow("Frame", frame);

        // Press  ESC on keyboard to exit
        char c = (char)waitKey(25);
        if (c == 27)
            break;
    }

    // When everything done, release the video capture object
    cap.release();

    // Closes all the frames
    destroyAllWindows();

    return 0;
}
