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

mutex frame_queue_mutex;
mutex result_queue_mutex;
mutex result_vector_3D_mutex;
mutex save_image_mutex;
condition_variable result_vector_3D_cv;
queue<Mat> frame_queue;
queue<Mat> result_queue;
vector<Mat> result_vector_3D;
atomic<bool> run_program(true);

int OPENPOSE_BATCH_SIZE = 8;
float CAMERA_TO_WORLD[4][4] = {
    {0, 0, -1, 0},
    {-1, 0, 0, 0},
    {0, 1, 0, 0},
    {0, 0, 0, 1}};

string PLOT_IMAGE_NAME = "tmp_plot.png";
Mat CAMERA_TO_WORLD_MAT = Mat(4, 4, CV_32FC1, CAMERA_TO_WORLD);

using Result = OpenPoseResult::PosePoint;

void dpuOutputIn2FP32(int8_t* outputAddr, float* buffer, int size,
                      float output_scale) {
  for (int idx = 0; idx < size; idx++) {
    buffer[idx] = outputAddr[idx] * output_scale;
  }
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
    save_image_mutex.lock();
    plt::save(PLOT_IMAGE_NAME);
    plt::close();
    save_image_mutex.unlock();
}

Mat process_result(cv::Mat &image, OpenPoseResult results)
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

Mat display_fps(cv::Mat &image, size_t fps)
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

void run_3D_pose(unique_ptr<OpenPose> &det, unique_ptr<Runner> &runner)
{
    vector<Mat> frames;
    vector<OpenPoseResult> results_2d;
    vector<Mat> results_3d;
    while (run_program)
    {
        frames.clear();
        frame_queue_mutex.lock();
        for (size_t i = 0; i < OPENPOSE_BATCH_SIZE && !frame_queue.empty(); ++i)
        {
            frames.push_back(frame_queue.front());
            frame_queue.pop();
        }
        frame_queue_mutex.unlock();

        results_2d.clear();
        results_3d.clear();

        if (!frames.empty())
        {
            vector<OpenPoseResult> results_2d_temp = det->run(frames);
            results_2d.insert(results_2d.begin(), begin(results_2d_temp), end(results_2d_temp));

            vector<Mat> results_3d_temp = run2D_to_3D(runner, results_2d);
            results_3d.insert(results_3d.begin(), begin(results_3d_temp), end(results_3d_temp));

            for (size_t i = 0; i < results_2d.size(); ++i)
            {
                Mat frame = frames.at(i);
                OpenPoseResult result2D = results_2d.at(i);
                frame = process_result(frame, result2D);
                result_queue_mutex.lock();
                result_queue.push(frame);
                result_queue_mutex.unlock();
            }

            result_vector_3D_mutex.lock();
            for (size_t i = 0; i < results_3d.size(); ++i)
            {
                result_vector_3D.push_back(results_3d.at(i));
            }
            result_vector_3D_mutex.unlock();
            result_vector_3D_cv.notify_all();
        }
    }
}

void run_plotting(int rows, int cols)
{
    while (run_program)
    {
        Mat body;
        {
            unique_lock<std::mutex> lk(result_vector_3D_mutex);
            result_vector_3D_cv.wait(lk, []
                                     { return run_program && !result_vector_3D.empty(); });
            if (!result_vector_3D.empty())
            {
                body = result_vector_3D.back();
                result_vector_3D.clear();
            }
        }
        if (!body.empty())
        {
            draw3DPlot(body, rows, cols);
        }
    }
}

int main(int argc, char *argv[])
{

    if (argc != 2)
    {
        cout << "Usage of yoga-ai: ./yoga-ai [model_file]" << endl;
        return -1;
    }
    auto det = OpenPose::create("openpose_pruned_0_3");
    int width = det->getInputWidth();
    int height = det->getInputHeight();

    auto graph = Graph::deserialize(argv[1]);
    auto subgraph = get_dpu_subgraph(graph.get());
    auto runner = Runner::create_runner(subgraph[0], "run");
    remove(PLOT_IMAGE_NAME.c_str());
    VideoCapture cap(-1);
    thread pose_th(run_3D_pose, ref(det), ref(runner));
    pose_th.detach();
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
    Mat first_frame;
    // Capture a frame for size
    cap >> first_frame;
    thread plot_th(run_plotting, first_frame.rows, first_frame.cols);
    plot_th.detach();
    Mat plot;
    while (1)
    {
        bool skip = false;
        Mat frame;
        // Capture frame-by-frame
        cap >> frame;

        // If the frame is empty, break immediately
        if (frame.empty())
            break;

        frame_queue_mutex.lock();
        frame_queue.push(frame);
        frame_queue_mutex.unlock();

        result_queue_mutex.lock();
        if (!result_queue.empty())
        {
            frame = result_queue.front();
            result_queue.pop();
        }
        else
        {
            skip = true;
        }
        result_queue_mutex.unlock();
        if (skip)
        {
            continue;
        }

        frame = display_fps(frame, fps);

        int rows = frame.rows;
        int cols = frame.cols * 2;
        // Create a black image
        Mat3b res(rows, cols, Vec3b(0, 0, 0));

        if (save_image_mutex.try_lock())
        {
            plot = imread(PLOT_IMAGE_NAME);
            save_image_mutex.unlock();
        }
        // Copy images in correct position
        frame.copyTo(res(Rect(0, 0, frame.cols, frame.rows)));
        if (!plot.empty())
        {
            plot.copyTo(res(Rect(frame.cols, 0, plot.cols, plot.rows)));
        };

        // Display the resulting frame
        imshow("Yoga-AI", res);

        // Press  ESC on keyboard to exit
        char c = (char)waitKey(5);
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
    run_program = false;
    result_vector_3D_cv.notify_all();
    // When everything done, release the video capture object
    cap.release();

    // Closes all the frames
    destroyAllWindows();
    remove(PLOT_IMAGE_NAME.c_str());
    return 0;
}