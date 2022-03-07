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

int main(int argc, char *argv[])
{
    if (argc <= 1)
    {
        std::cout << "usage : " << argv[0] << " <img_dir>" << std::endl;
        exit(1);
    }
    std::string path = argv[1];
    std::vector<std::string> image_files;
    std::vector<std::string> points;
    for (const auto &entry : fs::directory_iterator(path))
    {
        if (!entry.is_directory())
        {
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
            const auto &file = image_files[(batch_index * batch + index) % image_files.size()];
            batch_files[index] = file;
            images[index] = cv::imread(file);
            CHECK(!images[index].empty()) << "cannot read image from " << file;
        }
        auto results = model->run(images);

        for (auto i = 0u; i < results.size(); i++)
        {
            for (size_t j = 1; j < results[i].poses.size(); ++j)
            {
                bool valid = false;
                std::stringstream json;
                for (auto k = 0; k < results[i].poses[j].size(); ++k)
                {
                    if (results[i].poses[j][k].type != 1)
                        break;
                    if (k < results[i].poses[j].size() - 1)
                    {
                        json << results[i].poses[j][k].point << ", ";
                    }
                    else
                    {
                        json << results[i].poses[j][k].point;
                        valid = true;
                    }
                }
                if (valid)
                {
                    points.push_back("[" + json.str() + "]");
                }
            }
        }
    }

    std::cout << "[";
    for (auto i = 0u; i < points.size(); ++i)
    {
        if (i < points.size() - 1)
            std::cout << points[i] + "," << std::endl;
        else
            std::cout << points[i] << std::endl;
    }
    std::cout << "]";
}