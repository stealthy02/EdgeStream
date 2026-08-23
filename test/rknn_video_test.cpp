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
#include "rknn_engine.hpp"
#include "preprocess.h"
#include "postprocess.h"
#include "thread_safe_queue.h"
#include <fstream>
#include <sys/stat.h>
#include <nlohmann/json.hpp>

ThreadSafeQueue<cv::Mat> inference_queue(5);
PreprocessParameter preprocess_parameter;
int tensor_data_size;
void preprocess_thread(cv::VideoCapture& cap){
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video." << std::endl;
        return;
    }
    cv::Mat frame;
    while(cap.read(frame)){
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
    return mkdir(dir.c_str(), 0755) == 0;
}
void inference_thread(RknnEngine& rknn_engine){

    cv::Mat frame;
    std::vector<uint16_t> tensor_data(tensor_data_size);
    std::vector<float> output_data;
    const auto& out_attr = rknn_engine.get_output_attr(0);
    uint32_t field_count = 0;
    uint32_t candidate_count = 0;
    if (rknn_engine.get_output_num() == 2)
    {
        const auto& boxes_attr = rknn_engine.get_output_attr(0);
        const auto& scores_attr = rknn_engine.get_output_attr(1);
        if (boxes_attr.n_dims != 3 || scores_attr.n_dims != 3 ||
            boxes_attr.dims[1] != 4 || boxes_attr.dims[2] != scores_attr.dims[2])
        {
            std::cerr << "RKNN split 输出维度不符合 [1,4,N] + [1,C,N]" << std::endl;
            return;
        }
        field_count = boxes_attr.dims[1] + scores_attr.dims[1];
        candidate_count = boxes_attr.dims[2];
    }
    else if (out_attr.n_dims == 3)
    {
        uint32_t d1 = out_attr.dims[1];
        uint32_t d2 = out_attr.dims[2];
        if (d1 == 84)
        {
            field_count = d1;
            candidate_count = d2;
        }
        else if (d2 == 84)
        {
            field_count = d2;
            candidate_count = d1;
        }
        else
        {
            std::cerr << "RKNN输出维度不符合YOLO单输出格式！" << std::endl;
            return;
        }
    }
    else
    {
        std::cerr << "RKNN模型输出不是3维张量，仅支持YOLO单输出" << std::endl;
        return;
    }

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
        preprocess_image(frame, preprocess_parameter, tensor_data, TENSOR_NHWC);
        auto t1 = std::chrono::high_resolution_clock::now();
        // ========== 2. 推理 inference_engine.run ==========
        rknn_engine.run(tensor_data.data(), output_data);
        auto t2 = std::chrono::high_resolution_clock::now();
        // ========== 3. 后处理 ==========
        auto detections_original = postprocess_image(
            output_data.data(),
            static_cast<int>(field_count),
            static_cast<int>(candidate_count),
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
        const std::string raw_path = "artifacts/rknn/orangepi_300f_raw.jsonl";
        mkdir_if_not_exist("artifacts/rknn");
        std::ofstream raw(raw_path);
        for (std::size_t i = 0; i < total_list.size(); ++i) {
            json sample{{"frame", i + 1}, {"preprocess_ms", prep_list[i]},
                        {"infer_ms", infer_list[i]}, {"postprocess_ms", post_list[i]},
                        {"frame_total_ms", total_list[i]}};
            raw << sample.dump() << '\n';
        }
        std::cout << "Raw timings saved to: " << raw_path << std::endl;
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

        // ========== 输出JSON到 artifacts/onnx/infer_stats.json ==========
        const std::string out_dir = "artifacts/rknn";
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
            f << stats.dump(4); // 4空格格式化，方便阅读
            f.close();
            std::cout << "\n✅ Stats json saved to: " << json_path << std::endl;
        }else{
            std::cerr << "\n❌ Failed to write json to " << json_path << std::endl;
        }
    }
}


int main(int argc, char* argv[]){
    const std::string model_path = (argc == 1) 
    ? "models/rknn/yolo11s_640.rknn" 
    : argv[1];
    // 初始化, 当前已经封装了, 就是构建session
    std::string video_path = "assets/regression/input.mp4";
    cv::VideoCapture cap(video_path);
    RknnEngine rknn_engine;
    rknn_engine.init(model_path);
    rknn_engine.input_setting(RKNN_TENSOR_FLOAT16);
    const auto& in_attr = rknn_engine.get_input_attr(0);
    uint32_t model_in_h = 0;
    uint32_t model_in_w = 0;
    uint32_t channel = 0;

    if (in_attr.fmt == RKNN_TENSOR_NHWC)
    {
        model_in_h = in_attr.dims[1];
        model_in_w = in_attr.dims[2];
        channel = in_attr.dims[3];
    }
    else if (in_attr.fmt == RKNN_TENSOR_NCHW)
    {
        model_in_h = in_attr.dims[2];
        model_in_w = in_attr.dims[3];
        channel = in_attr.dims[1];
    }
    else
    {
        std::cerr << "不支持的输入排布 fmt=" << in_attr.fmt << std::endl;
        return -1;
    }
    int width  = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    preprocess_parameter = get_preprocess_parameter(width,height, model_in_w, model_in_h);
    tensor_data_size = channel * model_in_h * model_in_w;
    std::thread t1(preprocess_thread,std::ref(cap));
    std::thread t2(inference_thread,std::ref(rknn_engine));
    t1.join();
    t2.join();

    return 0;
}
