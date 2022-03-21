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
#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/openpose.hpp>
#include "common.h"
#include "pose.hpp"
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
size_t COLS = 800;
size_t ROWS = 600;

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

cv::Mat process_result(cv::Mat &image, vitis::ai::OpenPoseResult results, bool is_jpeg)
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
    cout << "usage : " << progname << "<model> <img_url>" << endl;
}

int main(int argc, char *argv[])
{
    if (argc <= 2)
    {
        usage(argv[1]);
        exit(1);
    }
    
    unique_ptr<PoseDetect> poseDetect(new PoseDetect(argv[1]));

    auto image_file_name = std::string{argv[2]};

    auto image = cv::imread(image_file_name);
    if (image.empty())
    {
        std::cerr << "cannot load image" << std::endl;
        abort();
    }

    OpenPoseResult results = poseDetect->predict2D(image);
    image = process_result(image, results, true);

    auto out_file = image_file_name.substr(0, image_file_name.size() - 4) + "_result.jpg";
    cout << "generated " << out_file << endl;

    bool check = imwrite(out_file, image);
    if (check == false)
    {
        cerr << "cannot save image" << endl;
        abort();
    }

    if (results.poses.size() <= 1)
    {
        cerr << "no pose found" << endl;
        abort();
    }

    vector<OpenPoseResult> results_2d;
    results_2d.push_back(results);
    vector<Mat> results_3d = poseDetect->predict2D_from_3D(results_2d);
    for (size_t i = 0; i < results_3d.size(); i++)
    {
        out_file = image_file_name.substr(0, image_file_name.size() - 4) + "_" + to_string(i + 1) + "_plot.jpg";
        cout << "generated " << out_file << endl;
        Mat body = results_3d.at(i);
        draw3DPlot(body, out_file);
    }
}