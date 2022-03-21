#ifndef POSE_DETECT_H
#define POSE_DETECT_H

#include <vector>
#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/openpose.hpp>

using namespace std;
using namespace cv;
using namespace vitis::ai;
using namespace vart;

extern float CAMERA_TO_WORLD[4][4];
extern Mat CAMERA_TO_WORLD_MAT;

void dpuOutputIn2FP32(int8_t *outputAddr, float *buffer, int size, float output_scale);

class PoseDetect
{
private:
    unique_ptr<Runner> runner;
    unique_ptr<OpenPose> det;

public:
    PoseDetect(char *model);
    vector<Mat> predict2D_from_3D(vector<OpenPoseResult> results_2d);
    OpenPoseResult predict2D(const Mat &image);
    vector<OpenPoseResult> predict2D(const vector<Mat> &images);
};

#endif