#define _USE_MATH_DEFINES
#include <iostream>
#include <stdio.h>
#include <memory>
#include <vector>
#include <cmath>
#include <chrono>
#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/openpose.hpp>
#include "common.h"
#include "pose.hpp"

using namespace std;
using namespace cv;
using namespace vitis::ai;
using namespace vart;
using namespace xir;

float CAMERA_TO_WORLD[4][4] = {
    {0, 0, -1, 0},
    {-1, 0, 0, 0},
    {0, 1, 0, 0},
    {0, 0, 0, 1}};

Mat CAMERA_TO_WORLD_MAT = Mat(4, 4, CV_32FC1, CAMERA_TO_WORLD);

void dpuOutputIn2FP32(int8_t *outputAddr, float *buffer, int size,
                      float output_scale)
{
    for (int idx = 0; idx < size; idx++)
    {
        buffer[idx] = outputAddr[idx] * output_scale;
    }
}

vector<cv::Mat> PoseDetect::predict2D_from_3D(vector<OpenPoseResult> results_2d)
{
    vector<Mat> results3D;
    vector<Mat> validBodies;
    vector<vector<Mat>> batchedValidBodies;

    TensorShape inshapes[1];
    TensorShape outshapes[1];

    GraphInfo shapes;
    shapes.inTensorList = inshapes;
    shapes.outTensorList = outshapes;
    getTensorShape(runner.get(), &shapes, 1, 1);

    auto inTensors = cloneTensorBuffer(runner->get_input_tensors());
    auto outTensors = cloneTensorBuffer(runner->get_output_tensors());

    int batchSize = inTensors[0]->get_shape().at(0);
    int width = inshapes[0].width;
    int height = inshapes[0].height;
    int inSize = inshapes[0].size;
    int outSize = outshapes[0].size;

    auto input_scale = get_input_scale(runner->get_input_tensors()[0]);
    auto output_scale = get_output_scale(runner->get_output_tensors()[0]);

    for (size_t r = 0; r < results_2d.size(); ++r)
    {
        OpenPoseResult result = results_2d.at(r);
        for (size_t k = 1; k < result.poses.size(); ++k)
        {
            vector<Point3f> bodyVec;
            for (size_t i = 0; i < result.poses[k].size(); ++i)
            {
                if (result.poses[k][i].type == 1)
                {
                    Point2f point = result.poses[k][i].point;
                    Point3f point3D(point.x, point.y, 1.);
                    bodyVec.push_back(point3D);
                }
            }
            if (bodyVec.size() == 14)
            {
                Point3f hip = (bodyVec.at(8) + bodyVec.at(11)) / 2;
                Point3f diff = hip - bodyVec.at(0);
                // ignore third coordinate
                float scale_hip_head = 1. / cv::sqrt(diff.x * diff.x + diff.y * diff.y);
                float image_to_camera_mat[3][3] = {
                    {-1, 0, 0},
                    {0, -1, 0},
                    {0, 0, 1}};
                Mat image_to_camera = Mat(3, 3, CV_32FC1, image_to_camera_mat);
                float scale_mat[3][3] = {
                    {scale_hip_head, 0, 0},
                    {0, scale_hip_head, 0},
                    {0, 0, 1}};
                Mat scale = Mat(3, 3, CV_32FC1, scale_mat);
                float transpose_mat[3][3] = {
                    {1, 0, hip.x},
                    {0, 1, hip.y},
                    {0, 0, 1}};
                Mat center = Mat(3, 3, CV_32FC1, transpose_mat);
                Mat body = Mat(14, 3, CV_32FC1, bodyVec.data());
                Mat transform = scale * center * image_to_camera;
                Mat bodyNormalized = (transform * body.t()).t();
                validBodies.push_back(bodyNormalized);
            }
        }
    }
    size_t quotient = validBodies.size() / batchSize;
    size_t reminder = validBodies.size() % batchSize;
    for (size_t i = 0; i < (quotient + (reminder > 0 ? 1 : 0)); i++)
    {
        vector<Mat> batch;
        if (i < quotient)
        {
            batch = vector<Mat>(validBodies.begin() + i * batchSize, validBodies.begin() + ((i + 1) * batchSize));
        }
        else
        {
            batch = vector<Mat>(validBodies.begin() + i * batchSize, validBodies.begin() + i * batchSize + reminder);
        }
        batchedValidBodies.push_back(batch);
    }

    for (size_t i = 0; i < batchedValidBodies.size(); ++i)
    {
        vector<Mat> batch = batchedValidBodies.at(i);
        int8_t *datain = new int8_t[inSize * batchSize];
        int8_t *dataresult = new int8_t[outSize * batchSize];
        for (size_t j = 0; j < batch.size(); ++j)
        {
            Mat bodyNormalized = batch.at(j);
            for (size_t n = 0; n < inSize; ++n)
            {
                if (n % 2 == 0)
                {
                    // x
                    datain[j * inSize + n] = bodyNormalized.at<float>(n / 2, 0) * input_scale;
                }
                else
                {
                    // y
                    datain[j * inSize + n] = bodyNormalized.at<float>(n / 2, 1) * input_scale;
                }
            }
        }

        vector<unique_ptr<vart::TensorBuffer>> inputs, outputs;
        inputs.push_back(make_unique<CpuFlatTensorBuffer>(datain, inTensors[0].get()));
        outputs.push_back(make_unique<CpuFlatTensorBuffer>(dataresult, outTensors[0].get()));
        vector<vart::TensorBuffer *> inputsPtr, outputsPtr;
        inputsPtr.push_back(inputs[0].get());
        outputsPtr.push_back(outputs[0].get());

        auto job_id = runner->execute_async(inputsPtr, outputsPtr);
        runner->wait(job_id.first, -1);
        vector<float> results(outSize * batchSize);
        dpuOutputIn2FP32(dataresult, results.data(), outSize, output_scale);
        for (size_t j = 0; j < batch.size(); ++j)
        {
            Mat bodyNormalized = batch.at(j);
            float open_pose_body[14][4];
            for (size_t n = 0; n < outSize; ++n)
            {
                open_pose_body[n][0] = bodyNormalized.at<float>(n, 0);
                open_pose_body[n][1] = bodyNormalized.at<float>(n, 1);
                open_pose_body[n][2] = results.at(j * outSize + n);
                open_pose_body[n][3] = 1.;
            }
            Mat bodyMat = Mat(14, 4, CV_32FC1, open_pose_body);
            Mat bodyMat_world = (CAMERA_TO_WORLD_MAT * bodyMat.t()).t();
            results3D.push_back(bodyMat_world);
        }
        delete[] datain;
        delete[] dataresult;
    }

    return results3D;
}

OpenPoseResult PoseDetect::predict2D(const Mat &image)
{
    return det->run(image);
}

vector<OpenPoseResult> PoseDetect::predict2D(const vector<Mat> &images)
{
    return det->run(images);
}

PoseDetect::PoseDetect(char *model)
{
    det = OpenPose::create("openpose_pruned_0_3");
    auto graph = Graph::deserialize(model);
    auto subgraph = get_dpu_subgraph(graph.get());
    runner = Runner::create_runner(subgraph[0], "run");
}