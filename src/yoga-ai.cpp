#define _USE_MATH_DEFINES
#include <iostream>
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
using Result = vitis::ai::OpenPoseResult::PosePoint;
GraphInfo shapes;

float CAMERA_TO_WORLD[4][4] = {
    {0, 0, -1, 0},
    {-1, 0, 0, 0},
    {0, 1, 0, 0},
    {0, 0, 0, 1}};

string PLOT_IMAGE_NAME = "tmp_plot.png";
Mat CAMERA_TO_WORLD_MAT = Mat(4, 4, CV_32FC1, CAMERA_TO_WORLD);

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

static cv::Mat display_fps(cv::Mat &image, size_t fps)
{
    cv::putText(image,
                to_string(fps) + " fps",
                cv::Point(20, 20),              // Coordinates (Bottom-left corner of the text string in the image)
                cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
                1.0,                            // Scale. 2.0 = 2x bigger
                cv::Scalar(0, 0, 0),
                1,            // Line Thickness (Optional)
                cv::LINE_AA); // Anti-alias (Optional, see version note)
    return image;
}

void draw3DPlot(cv::Mat body, unsigned int rows, unsigned int cols)
{
    plt::figure_size(cols, rows);    
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
    plt::save(PLOT_IMAGE_NAME);
    plt::close();
}

int main(int argc, char *argv[])
{

    if (argc != 2)
    {
        cout << "Usage of yoga-ai: ./yoga-ai [model_file]" << endl;
        return -1;
    }

    remove(PLOT_IMAGE_NAME.c_str());

    auto det = vitis::ai::OpenPose::create("openpose_pruned_0_3");
    int width = det->getInputWidth();
    int height = det->getInputHeight();

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

    std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;
    std::vector<vart::TensorBuffer *> inputsPtr, outputsPtr;
    std::vector<std::shared_ptr<xir::Tensor>> batchTensors;

    VideoCapture cap(-1);
    time_t start, end;
    // Check if camera opened successfully
    if (!cap.isOpened())
    {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    size_t frame_counter = 0;
    time_point<steady_clock> begin_time = steady_clock::now(), new_time;
    size_t fps = 0;
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
        frame = display_fps(frame, fps);

        vector<Point3f> bodyVec;
        for (size_t k = 1; k < results.poses.size(); ++k)
        {
            for (size_t i = 0; i < results.poses[k].size(); ++i)
            {
                if (results.poses[k][i].type == 1)
                {
                    Point2f point = results.poses[k][i].point;
                    Point3f point3D(point.x, point.y, 1.);
                    bodyVec.push_back(point3D);
                }
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
            int8_t *data = new int8_t[batchSize * inSize];
            for (size_t i = 0; i < inSize; ++i)
            {
                if (i % 2)
                {
                    // x
                    data[i] = bodyNormalized.at<float>(i / 2, 0) * input_scale;
                }
                else
                {
                    // y
                    data[i] = bodyNormalized.at<float>(i / 2, 1) * input_scale;
                }
            }

            int8_t *FCResult = new int8_t[batchSize * outSize];
            inputs.clear();
            outputs.clear();
            inputsPtr.clear();
            outputsPtr.clear();
            batchTensors.clear();

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
                open_pose_body[n][0] = bodyNormalized.at<float>(n, 0);
                open_pose_body[n][1] = bodyNormalized.at<float>(n, 1);
                open_pose_body[n][2] = (float)(FCResult[n] * output_scale);
                open_pose_body[n][3] = 1.;
            }
            delete[] FCResult;
            delete[] data;

            Mat bodyMat = Mat(14, 4, CV_32FC1, open_pose_body);
            Mat bodyMat_world = (CAMERA_TO_WORLD_MAT * bodyMat.t()).t();
            draw3DPlot(bodyMat_world, frame.rows, frame.cols);
        }

        Mat plot = imread(PLOT_IMAGE_NAME);
        int rows = frame.rows;
        int cols = frame.cols * 2;

        // Create a black image
        Mat3b res(rows, cols, Vec3b(0, 0, 0));

        // Copy images in correct position
        frame.copyTo(res(Rect(0, 0, frame.cols, frame.rows)));
        if (!plot.empty())
        {
            plot.copyTo(res(Rect(frame.cols, 0, plot.cols, plot.rows)));
        }

        // Display the resulting frame
        imshow("Yoga-AI", res);

        // Press  ESC on keyboard to exit
        char c = (char)waitKey(1);
        if (c == 27)
            break;
        frame_counter++;
        new_time = steady_clock::now();
        if (new_time - begin_time >= seconds{1})
        {
            fps = frame_counter;
            frame_counter = 0;
            begin_time = new_time;
        }
    }

    // When everything done, release the video capture object
    cap.release();

    // Closes all the frames
    destroyAllWindows();
    remove(PLOT_IMAGE_NAME.c_str());

    return 0;
}
