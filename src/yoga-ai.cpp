#include <iostream>
#include <memory>
#include <vector>
#include "common.h"
#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/openpose.hpp>

using namespace std;
using namespace cv;
using namespace xir;
using namespace vart;
using Result = vitis::ai::OpenPoseResult::PosePoint;
GraphInfo shapes;

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
        0.9602672482891879, 0.9755428071569374, 0.8031329873221553, 0.5957998025133866, 0.602350142057307, 0.5957998025133866, 0.2444330358254263, 0.896974070410659, -0.06983548610863854, 1.1981483383079317, 0.9428071720649567, 0.5172342782926325, 1.1872386016276208, 0.8053142921531125, 1.4142099549660538, 1.1588655761975546, 0.03491854618570045, -0.0982085115387048, 0.02619011433634655, -0.8183956960878085, 0.1309473591562098, -1.460033419043779, -0.03491854618570045, 0.0982085115387048, -0.4975928972374444, 0.5696112944398017, -0.23570139145054858, 1.1195828140871775
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
    for (unsigned int n = 0; n < outSize; ++n)
    {
        cout << (float)(FCResult[n] * output_scale) << ", ";
    }
    cout << endl;

    delete[] FCResult;
    delete[] data;

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
