#define _USE_MATH_DEFINES
#include <iostream>
#include <stdio.h>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <vector>
#include <cmath>
#include <chrono>
#include "common.h"
#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/openpose.hpp>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;
using namespace std;
using namespace std::chrono;
using namespace cv;
using namespace xir;
using namespace vart;
using namespace vitis::ai;

using Result = vitis::ai::OpenPoseResult::PosePoint;

int OPENPOSE_BATCH_SIZE = 8;
float CAMERA_TO_WORLD[4][4] = {
    {0, 0, -1, 0},
    {-1, 0, 0, 0},
    {0, 1, 0, 0},
    {0, 0, 0, 1}};

string PLOT_IMAGE_NAME = "tmp_plot.png";
Mat CAMERA_TO_WORLD_MAT = Mat(4, 4, CV_32FC1, CAMERA_TO_WORLD);
size_t COLS = 800;
size_t ROWS = 600;

void dpuOutputIn2FP32(int8_t* outputAddr, float* buffer, int size,
                      float output_scale) {
  for (int idx = 0; idx < size; idx++) {
    buffer[idx] = outputAddr[idx] * output_scale;
  }
}

void draw3DPlot(cv::Mat body, string filename)
{
    plt::figure_size(COLS, ROWS);
    cv::Mat anchor = (body.row(8) + body.row(11)) / 2;
    body.push_back(anchor);
    unsigned int start[] = {14, 8, 9, 14, 11, 12, 14, 1, 1, 2, 3, 1, 5, 6};
    unsigned int end[] = {8, 9, 10, 11, 12, 13, 1, 0, 2, 3, 4, 5, 6, 7};
    unsigned int colors[] = {1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0};

    string rcolor = "blue";
    string lcolor = "red";
    vector<vector<float>> x;
    vector<vector<float>> y;
    vector<vector<float>> z;
    for (auto i = 0; i < sizeof(start) / sizeof(start[0]); ++i)
    {
        map<string, string> keywords;
        if (colors[i])
        {
            keywords.insert(std::pair<std::string, std::string>("c", lcolor));
        }
        else
        {
            keywords.insert(std::pair<std::string, std::string>("c", rcolor));
        }
        // Need to save the vectors for plotting to work
        vector<float> xx{body.at<float>(start[i], 0), body.at<float>(end[i], 0)};
        vector<float> yy{body.at<float>(start[i], 1), body.at<float>(end[i], 1)};
        vector<float> zz{body.at<float>(start[i], 2), body.at<float>(end[i], 2)};
        x.push_back(xx);
        y.push_back(yy);
        z.push_back(zz);
        plt::plot3(x.at(i), y.at(i), z.at(i), keywords, 1);
    }

    float x_root = body.at<float>(14, 0);
    float y_root = body.at<float>(14, 1);
    float z_root = body.at<float>(14, 2);

    plt::set_xlim3d(-1.2, 1.2, 1);
    plt::set_ylim3d(-1.2, 1.2, 1);
    plt::set_zlim3d(-1.2, 1.2, 1);

    plt::xlabel("x");
    plt::ylabel("y");
    plt::set_zlabel("z");
    plt::save(filename);
    plt::close();
}

vector<Mat> run2D_to_3D(unique_ptr<Runner> &runner, vector<OpenPoseResult> results_2d)
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
                open_pose_body[n][2] = results.at(j*outSize + n);
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

cv::Mat process_result(cv::Mat &image, vitis::ai::OpenPoseResult results,
                       bool is_jpeg)
{
    vector<vector<int>> limbSeq = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {1, 5}, {5, 6}, {6, 7}, {1, 8}, {8, 9}, {9, 10}, {1, 11}, {11, 12}, {12, 13}};

    for (size_t k = 1; k < results.poses.size(); ++k)
    {
        for (size_t i = 0; i < results.poses[k].size(); ++i)
        {
            if (results.poses[k][i].type == 1)
            {
                cv::circle(image, results.poses[k][i].point, 5, cv::Scalar(0, 255, 0), -1);
            }
        }
        for (size_t i = 0; i < limbSeq.size(); ++i)
        {
            Result a = results.poses[k][limbSeq[i][0]];
            Result b = results.poses[k][limbSeq[i][1]];
            if (a.type == 1 && b.type == 1)
            {
                cv::line(image, a.point, b.point, cv::Scalar(255, 0, 0), 3, 4);
            }
        }
    }
    return image;
}

void usage(const char *progname)
{
    std::cout << "usage : " << progname << "<model> <img_url>"
              << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc <= 2)
    {
        usage(argv[1]);
        exit(1);
    }

    auto image_file_name = std::string{argv[2]};

    auto image = cv::imread(image_file_name);
    if (image.empty())
    {
        std::cerr << "cannot load image" << std::endl;
        abort();
    }
    auto det = vitis::ai::OpenPose::create("openpose_pruned_0_3");
    int width = det->getInputWidth();
    int height = det->getInputHeight();

    OpenPoseResult results = det->run(image);
    image = process_result(image, results, true);

    auto out_file =
        image_file_name.substr(0, image_file_name.size() - 4) + "_result.jpg";

    bool check = imwrite(out_file, image);
    if (check == false)
    {
        std::cerr << "cannot save image" << std::endl;
        abort();
    }

    if (results.poses.size() <= 1)
    {
        std::cerr << "no pose found" << std::endl;
        abort();
    }

    auto graph = Graph::deserialize(argv[1]);
    auto subgraph = get_dpu_subgraph(graph.get());
    auto runner = Runner::create_runner(subgraph[0], "run");

    vector<OpenPoseResult> results_2d;
    results_2d.push_back(results);
    vector<Mat> results_3d = run2D_to_3D(ref(runner), results_2d);
    if (!results_3d.empty())
    {
        auto out_file = image_file_name.substr(0, image_file_name.size() - 4) + "_plot.jpg";
        Mat body = results_3d.front();
        draw3DPlot(body, out_file);
    }
}