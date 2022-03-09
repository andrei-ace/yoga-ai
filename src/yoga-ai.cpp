#define _USE_MATH_DEFINES
#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include "common.h"
#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/openpose.hpp>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;
using namespace std;
using namespace cv;
using namespace xir;
using namespace vart;
using Result = vitis::ai::OpenPoseResult::PosePoint;
GraphInfo shapes;

float CAMERA_TO_WORLD[4][4] = {
    { 0,-1, 0, 0},
    { 0, 0, 1, 0},
    {-1, 0, 0, 0},
    { 0, 0, 0, 1}};

float CAMERA_TO_WORLD_TRANSPOSED[4][4] = {
    { 0, 0,-1, 0},
    {-1, 0, 0, 0},
    { 0, 1, 0, 0},
    { 0, 0, 0, 1}};

Mat CAMERA_TO_WORLD_MAT = Mat(4,4,CV_32FC1,CAMERA_TO_WORLD_TRANSPOSED); 

static cv::Mat process_result(cv::Mat &image, vitis::ai::OpenPoseResult results,
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

int main(int argc, char *argv[])
{

    if (argc != 2)
    {
        cout << "Usage of yoga-ai: ./yoga-ai [model_file]" << endl;
        return -1;
    }

    // auto det = vitis::ai::OpenPose::create("openpose_pruned_0_3");
    // int width = det->getInputWidth();
    // int height = det->getInputHeight();

    auto graph = Graph::deserialize(argv[1]);
    auto subgraph = get_dpu_subgraph(graph.get());
    auto runner = Runner::create_runner(subgraph[0], "run");

    auto inputTensors = runner->get_input_tensors();
    auto outputTensors = runner->get_output_tensors();

    /*get in/out tensor shape*/
    int inputCnt = inputTensors.size();
    int outputCnt = outputTensors.size();
    TensorShape inshapes[inputCnt];
    TensorShape outshapes[outputCnt];
    shapes.inTensorList = inshapes;
    shapes.outTensorList = outshapes;
    getTensorShape(runner.get(), &shapes, inputCnt, outputCnt);

    /* get in/out tensors and dims*/
    auto out_dims = outputTensors[0]->get_shape();
    auto in_dims = inputTensors[0]->get_shape();

    auto input_scale = get_input_scale(inputTensors[0]);
    auto output_scale = get_output_scale(outputTensors[0]);

    /*get shape info*/
    int outSize = shapes.outTensorList[0].size;
    int inSize = shapes.inTensorList[0].size;

    int batchSize = in_dims[0];

    cout << "batch:" << batchSize << " size:" << inSize << endl;

    // int8_t *data = new int8_t[inSize * batchSize];
    // for(unsigned int n = 0; n < inSize; ++n) {
    //     data[n] = (int8_t)(0.0 * input_scale);
    // }

    float body[] = {
        1.3474549608053594, 0.46383189124773727, 0.9423380184089432, 0.33466260481431975, 0.7309730050851595, 0.5225448395215646, 0.7133599208340091, 1.0274705930594976, 0.819042427495901, 1.4619440095928273, 0.9071118499066415, 0.0058716949428818666, 0.9071118499066415, -0.45208450020698665, 0.9951772711623961, -0.8865579167403164, -0.13210413361611195, 0.052841253330945914, -0.44915165360178655, -0.47557127997851234, -0.766199173587462, -1.0274705930594967, 0.13210413361611106, -0.052841253330945914, 0.6076774141721173, -0.5812537866404042, 0.9951772711623961, -0.9570142548999065
    };

    int8_t *data = new int8_t[batchSize * outSize];
    for (unsigned int n = 0; n < inSize; ++n)
    {
        data[n] = body[n] * input_scale;
    }

    int8_t *FCResult = new int8_t[batchSize * outSize];

    std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;

    std::vector<vart::TensorBuffer *> inputsPtr, outputsPtr;
    std::vector<std::shared_ptr<xir::Tensor>> batchTensors;

    /* in/out tensor refactory for batch inout/output */
    batchTensors.push_back(std::shared_ptr<xir::Tensor>(
        xir::Tensor::create(inputTensors[0]->get_name(), in_dims,
                            xir::DataType{xir::DataType::XINT, 8u})));
    inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
        data, batchTensors.back().get()));
    batchTensors.push_back(std::shared_ptr<xir::Tensor>(
        xir::Tensor::create(outputTensors[0]->get_name(), out_dims,
                            xir::DataType{xir::DataType::XINT, 8u})));
    outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
        FCResult, batchTensors.back().get()));

    /*tensor buffer input/output */
    inputsPtr.clear();
    outputsPtr.clear();
    inputsPtr.push_back(inputs[0].get());
    outputsPtr.push_back(outputs[0].get());

    auto job_id = runner->execute_async(inputsPtr, outputsPtr);
    runner->wait(job_id.first, -1);

    float open_pose_body[14][4];
    for (unsigned int n = 0; n < outSize; ++n)
    {
        open_pose_body[n][0] = body[n*2];
        open_pose_body[n][1] = body[n*2+1];
        open_pose_body[n][2] = (float)(FCResult[n] * output_scale);
        open_pose_body[n][3] = 1.;
    }
    delete[] FCResult;
    delete[] data;

    Mat bodyMat = Mat(14,4,CV_32FC1,open_pose_body);
    Mat bodyMat_world = CAMERA_TO_WORLD_MAT * bodyMat.t();

    cout << bodyMat << endl;
    cout << endl;
    cout << bodyMat_world.t() << endl;

    // std::map<std::string, std::string> keywords;
    // keywords.insert(std::pair<std::string, std::string>("label", "parametric curve") );

    // plt::plot(x, y, keywords);
    // plt::xlabel("x label");
    // plt::ylabel("y label");
    // // plt::set_zlabel("z label"); // set_zlabel rather than just zlabel, in accordance with the Axes3D method
    // plt::legend();
    // plt::save("imshow.png");;

    // VideoCapture cap(-1);

    // // Check if camera opened successfully
    // if (!cap.isOpened())
    // {
    //     cout << "Error opening video stream or file" << endl;
    //     return -1;
    // }

    // while (1)
    // {

    //     Mat frame;
    //     // Capture frame-by-frame
    //     cap >> frame;

    //     // If the frame is empty, break immediately
    //     if (frame.empty())
    //         break;

    //     auto results = det->run(frame);
    //     frame = process_result(frame, results, true);

    //     for (size_t k = 1; k < results.poses.size(); ++k)
    //     {
    //         for (size_t i = 0; i < results.poses[k].size(); ++i)
    //         {
    //             // if (results.poses[k][i].type == 1)
    //             // {
    //             //     cout << results.poses[k][i].point << endl;
    //             // }
    //         }
    //     }

    //     // std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;

    //     // inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(imageInputs, batchTensors.back().get()));
    //     // std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;

    //     // inputsPtr.push_back(inputs[0].get());
    //     // auto job_id = runner->execute_async(inputsPtr, outputsPtr);
    //     // runner->wait(job_id.first, -1);

    //     // Display the resulting frame
    //     imshow("Yoga-AI", frame);

    //     // Press  ESC on keyboard to exit
    //     char c = (char)waitKey(25);
    //     if (c == 27)
    //         break;
    // }

    // // When everything done, release the video capture object
    // cap.release();

    // // Closes all the frames
    // destroyAllWindows();

    return 0;
}
