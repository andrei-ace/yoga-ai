#include <string>
#include <iostream>
#include <filesystem>
#include <memory>
#include <vector>
#include <cmath>
#include <glog/logging.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/openpose.hpp>

namespace fs = std::filesystem;

using Result = vitis::ai::OpenPoseResult::PosePoint;

static cv::Mat process_result(cv::Mat &image, vitis::ai::OpenPoseResult results,
                              bool is_jpeg)
{
    std::vector<std::vector<int>> limbSeq = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {1, 5}, {5, 6}, {6, 7}, {1, 8}, {8, 9}, {9, 10}, {1, 11}, {11, 12}, {12, 13}};

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

int main(int argc, char *argv[])
{
    if (argc <= 1)
    {
        std::cout << "usage : " << argv[0] << " <img_dir>" << std::endl;
        exit(1);
    }
    std::string path = argv[1];
    std::vector<std::string> image_files;
    for (const auto &entry : fs::directory_iterator(path))
    {
        if (!entry.is_directory()) {
            image_files.push_back(entry.path());
        }
    }
    if (image_files.empty())
    {
        std::cerr << "no input file" << std::endl;
        exit(1);
    }

    auto model = vitis::ai::OpenPose::create("openpose_pruned_0_3");

    auto batch = model->get_input_batch();

    for (auto batch_index = 0u; batch_index < ceil((float)image_files.size() / batch); ++batch_index)
    {
        std::vector<std::string> batch_files(batch);
        std::vector<cv::Mat> images(batch);
        for (auto index = 0u; index < batch; ++index)
        {
            const auto &file = image_files[(batch_index*batch + index) % image_files.size()];
            batch_files[index] = file;
            images[index] = cv::imread(file);
            CHECK(!images[index].empty()) << "cannot read image from " << file;
        }
        auto results = model->run(images);
        for (auto i = 0u; i < results.size(); i++)
        {
            // LOG(INFO) << "batch: " << batch_index << "     image: " << batch_files[i];
            auto image = process_result(images[i], results[i], true);
            auto out_file = batch_files[i].substr(0, batch_files[i].size() - 4) +
                           "_result.jpg";
            cv::imwrite(out_file, image);
            LOG(INFO) << "batch: " << batch_index << "     image: " << out_file;
        }
    }

    // for (auto index = 0u; index < batch; ++index)
    // {
    //     const auto &file = image_files[index % image_files.size()];
    //     batch_files[index] = file;
    //     images[index] = cv::imread(file);
    //     CHECK(!images[index].empty()) << "cannot read image from " << file;
    // }

    // auto results = model->run(images);

    // assert(results.size() == batch);
    // for (auto i = 0u; i < results.size(); i++)
    // {
    //     LOG(INFO) << "batch: " << i << "     image: " << batch_files[i];
    //     auto image = process_result(images[i], results[i], true);
    //     auto out_file = std::to_string(i) + "_" +
    //                     batch_files[i].substr(0, batch_files[i].size() - 4) +
    //                     "_result.jpg";
    //     cv::imwrite(out_file, image);
    //     std::cout << std::endl;
    // }
}