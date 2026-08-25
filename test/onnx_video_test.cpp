#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <csignal>
#include "inference_engine.h"
#include "preprocess.h"
#include "postprocess.h"
#include "thread_safe_queue.h"
#include <fstream>
#include <sys/stat.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
volatile std::sig_atomic_t g_stop_requested = 0;

ThreadSafeQueue<cv::Mat> inference_queue(5);
PreprocessParameter preprocess_parameter;
int tensor_data_size;

void sigint_handler(int /*sig*/)
{
    g_stop_requested = 1;
}

void preprocess_thread(cv::VideoCapture& cap){
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video." << std::endl;
        inference_queue.stop();
        return;
    }
    cv::Mat frame;
    while(g_stop_requested == 0 && cap.read(frame)){
        if (frame.empty()) continue;
        inference_queue.push(frame);
    }
    inference_queue.stop();
}
using json = nlohmann::json;

double calc_std(const std::vector<double>& data, double mean)
{
    if (data.size() <= 1) return 0.0;
    double sum_sq = 0.0;
    for (auto v : data) {
        double d = v - mean;
        sum_sq += d * d;
    }
    return std::sqrt(sum_sq / (data.size() - 1));
}

bool mkdir_if_not_exist(const std::string& dir)
{
    struct stat st;
    if (stat(dir.c_str(), &st) == 0) {
        return S_ISDIR(st.st_mode);
    }
    // linux mkdir -p
    return mkdir(dir.c_str(), 0755) == 0;
}

void inference_thread(InferenceEngine& inference_engine){
    InferenceOutput inference_output;
    cv::Mat frame;
    std::vector<float> tensor_data(tensor_data_size);

    // ========== 保存每帧耗时数组（用于标准差计算） ==========
    std::vector<double> prep_list;
    std::vector<double> infer_list;
    std::vector<double> post_list;
    std::vector<double> total_list;

    double total_prep = 0.0;
    double total_infer = 0.0;
    double total_post = 0.0;
    int frame_cnt = 0;

    while (inference_queue.pop(frame)) {
        frame_cnt++;
        // ========== 1. 前处理 preprocess_image ==========
        auto t0 = std::chrono::high_resolution_clock::now();
        preprocess_image(frame, preprocess_parameter, tensor_data, TENSOR_NCHW);
        auto t1 = std::chrono::high_resolution_clock::now();
        // ========== 2. 推理 inference_engine.run ==========
        auto inference_output_list = inference_engine.run(tensor_data);
        auto t2 = std::chrono::high_resolution_clock::now();
        // ========== 3. 后处理 ==========
        InferenceOutput inference_output;
        if (!inference_output_list.empty())
        {
            // 直接move第一个输出，避免拷贝
            inference_output = std::move(inference_output_list[0]);
            int64_t sum_dim1 = inference_output.shape[1];
            const int64_t dim2 = inference_output.shape[2];

            // 从第二个开始追加
            for (size_t i = 1; i < inference_output_list.size(); ++i)
            {
                auto& out = inference_output_list[i];
                inference_output.data.insert(inference_output.data.end(),
                                            std::make_move_iterator(out.data.begin()),
                                            std::make_move_iterator(out.data.end()));
                sum_dim1 += out.shape[1];
            }
            inference_output.shape = {1, sum_dim1, dim2};
        }
        auto detections_original = postprocess_image(
            inference_output.data.data(),
            inference_output.shape[1],
            inference_output.shape[2],
            0.25,
            0.70F,
            preprocess_parameter
        );
        auto t3 = std::chrono::high_resolution_clock::now();

        // 转毫秒
        double prep_ms   = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double infer_ms  = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double post_ms   = std::chrono::duration<double, std::milli>(t3 - t2).count();
        double frame_total_ms = prep_ms + infer_ms + post_ms;

        total_prep += prep_ms;
        total_infer += infer_ms;
        total_post += post_ms;

        prep_list.push_back(prep_ms);
        infer_list.push_back(infer_ms);
        post_list.push_back(post_ms);
        total_list.push_back(frame_total_ms);
    }

    // 退出循环后统计并写入JSON
    if(frame_cnt > 0){
        double avg_prep  = total_prep / frame_cnt;
        double avg_infer = total_infer / frame_cnt;
        double avg_post  = total_post / frame_cnt;

        double std_prep  = calc_std(prep_list, avg_prep);
        double std_infer = calc_std(infer_list, avg_infer);
        double std_post  = calc_std(post_list, avg_post);

        double avg_total = (total_prep + total_infer + total_post) / frame_cnt;
        double std_total = calc_std(total_list, avg_total);

        std::cout << "\n===== Stat over " << frame_cnt << " frames =====" << std::endl;
        std::cout << std::fixed << std::setprecision(3)
                  << "Avg Prep: " << avg_prep << " ms | Std Prep: " << std_prep << " ms\n"
                  << "Avg Infer: " << avg_infer << " ms | Std Infer: " << std_infer << " ms\n"
                  << "Avg Post: " << avg_post << " ms | Std Post: " << std_post << " ms\n"
                  << "Avg FrameTotal: " << avg_total << " ms | Std FrameTotal: " << std_total << " ms"
                  << std::endl;

        const std::string out_dir = "artifacts/onnx";
        const std::string json_path = out_dir + "/orangepi_infer_stats.json";
        mkdir_if_not_exist(out_dir);

        json stats;
        stats["frame_count"] = frame_cnt;
        stats["unit"] = "ms";

        stats["preprocess"]["mean"] = avg_prep;
        stats["preprocess"]["std"]  = std_prep;

        stats["infer"]["mean"] = avg_infer;
        stats["infer"]["std"]  = std_infer;

        stats["postprocess"]["mean"] = avg_post;
        stats["postprocess"]["std"]  = std_post;

        stats["frame_total"]["mean"] = avg_total;
        stats["frame_total"]["std"]  = std_total;

        std::ofstream f(json_path);
        if(f.is_open()){
            f << stats.dump(4);
            f.close();
            std::cout << "\n✅ Stats json saved to: " << json_path << std::endl;
        }else{
            std::cout << "\n[WARN] No frame processed, skip save json" << std::endl;
        }
    }
}


int main(int argc, char* argv[]){
    signal(SIGINT, sigint_handler);
    const std::string model_path = (argc == 1) 
    ? "models/onnx/yolo11s_640_split.onnx" 
    : argv[1];
    // 初始化, 当前已经封装了, 就是构建session
    std::string video_path = "assets/regression/input.mp4";
    cv::VideoCapture cap(video_path);
    InferenceEngine inference_engine(model_path);
    auto input_shape = inference_engine.input_shape();
    int width  = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    preprocess_parameter = get_preprocess_parameter(width,height,input_shape[3],input_shape[2]);
    tensor_data_size = input_shape[2]*input_shape[3]*3;
    std::thread t1(preprocess_thread,std::ref(cap));
    std::thread t2(inference_thread,std::ref(inference_engine));
    t1.join();
    t2.join();
    cap.release();
    std::cout << "\n[INFO] Program exit gracefully" << std::endl;
    return 0;
}
