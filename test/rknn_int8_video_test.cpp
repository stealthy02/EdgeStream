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

#include "rknn_engine.hpp"
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

void preprocess_thread(cv::VideoCapture& cap)
{
    if (!cap.isOpened()) {
        inference_queue.stop();
        return;
    }
    cv::Mat frame;
    while (g_stop_requested == 0 && cap.read(frame))
    {
        if (frame.empty())
            continue;
        inference_queue.push(frame);
    }
    inference_queue.stop();
}

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

void inference_thread(RknnEngine& rknn_engine, bool breakdown_enabled){
    cv::Mat frame;
    std::vector<int8_t> tensor_data(tensor_data_size);
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

    std::vector<double> prep_list;
    std::vector<double> infer_list;
    std::vector<double> post_list;
    std::vector<double> total_list;
    std::vector<double> input_copy_list;
    std::vector<double> inputs_set_list;
    std::vector<double> run_call_list;
    std::vector<double> outputs_get_list;
    std::vector<double> output_pack_list;
    std::vector<double> outputs_release_list;
    std::vector<double> infer_total_list;
    std::vector<double> prep_resize_list;
    std::vector<double> prep_padding_list;
    std::vector<double> prep_pack_list;
    std::vector<double> prep_total_list;

    double total_prep = 0.0;
    double total_infer = 0.0;
    double total_post = 0.0;
    int frame_cnt = 0;
    while (inference_queue.pop(frame))
    {
        frame_cnt++;
        auto t0 = std::chrono::high_resolution_clock::now();
        PreprocessTiming prep_timing;
        preprocess_image(
            frame,
            preprocess_parameter,
            tensor_data,
            TENSOR_NHWC,
            breakdown_enabled ? &prep_timing : nullptr
        );
        auto t1 = std::chrono::high_resolution_clock::now();

        RknnRunTiming run_timing;
        const int run_ret = rknn_engine.run(
            tensor_data.data(),
            output_data,
            breakdown_enabled ? &run_timing : nullptr
        );
        if (run_ret != 0) {
            const int attempted_frame = frame_cnt;
            --frame_cnt;
            std::cerr << "RKNN inference failed at frame " << attempted_frame
                      << ", ret=" << run_ret << std::endl;
            break;
        }
        auto t2 = std::chrono::high_resolution_clock::now();

        auto detections_original = postprocess_image(
            output_data.data(),
            static_cast<int>(field_count),
            static_cast<int>(candidate_count),
            0.25,
            0.70F,
            preprocess_parameter
        );
        auto t3 = std::chrono::high_resolution_clock::now();

        double prep_ms    = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double infer_ms   = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double post_ms    = std::chrono::duration<double, std::milli>(t3 - t2).count();
        double frame_total_ms = prep_ms + infer_ms + post_ms;

        total_prep += prep_ms;
        total_infer += infer_ms;
        total_post += post_ms;

        prep_list.push_back(prep_ms);
        infer_list.push_back(infer_ms);
        post_list.push_back(post_ms);
        total_list.push_back(frame_total_ms);
        if (breakdown_enabled) {
            input_copy_list.push_back(run_timing.input_copy_us);
            inputs_set_list.push_back(run_timing.inputs_set_us);
            run_call_list.push_back(run_timing.run_call_us);
            outputs_get_list.push_back(run_timing.outputs_get_us);
            output_pack_list.push_back(run_timing.output_pack_us);
            outputs_release_list.push_back(run_timing.outputs_release_us);
            infer_total_list.push_back(run_timing.total_us);
            prep_resize_list.push_back(prep_timing.resize_us);
            prep_padding_list.push_back(prep_timing.padding_us);
            prep_pack_list.push_back(prep_timing.pack_us);
            prep_total_list.push_back(prep_timing.total_us);
        }
    }

    // 正常跑完 / Ctrl+C触发优雅停止，排空队列后都会进入统计保存json
    if(frame_cnt > 0){
        const std::string raw_path = breakdown_enabled
            ? "artifacts/rknn/orangepi_int8_breakdown_raw.jsonl"
            : "artifacts/rknn/orangepi_int8_300f_raw.jsonl";
        mkdir_if_not_exist("artifacts/rknn");
        std::ofstream raw(raw_path);
        if (!raw.is_open()) {
            std::cerr << "\n❌ Failed to write raw timings to " << raw_path << std::endl;
        } else {
            for (std::size_t i = 0; i < total_list.size(); ++i) {
                json sample{{"frame", i + 1}, {"preprocess_ms", prep_list[i]},
                            {"infer_ms", infer_list[i]}, {"postprocess_ms", post_list[i]},
                            {"frame_total_ms", total_list[i]}};
                if (breakdown_enabled) {
                    sample["preprocess_breakdown_us"] = {
                        {"resize", prep_resize_list[i]},
                        {"padding", prep_padding_list[i]},
                        {"pack", prep_pack_list[i]},
                        {"total", prep_total_list[i]}
                    };
                    sample["infer_breakdown_us"] = {
                        {"input_copy", input_copy_list[i]},
                        {"inputs_set", inputs_set_list[i]},
                        {"run_call", run_call_list[i]},
                        {"outputs_get", outputs_get_list[i]},
                        {"output_pack", output_pack_list[i]},
                        {"outputs_release", outputs_release_list[i]},
                        {"total", infer_total_list[i]}
                    };
                }
                raw << sample.dump() << '\n';
            }
            std::cout << "Raw timings saved to: " << raw_path << std::endl;
        }

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

        const std::string out_dir = "artifacts/rknn";
        const std::string json_path = breakdown_enabled
            ? out_dir + "/orangepi_int8_breakdown_stats.json"
            : out_dir + "/orangepi_int8_infer_stats.json";
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

        if (breakdown_enabled) {
            auto save_breakdown = [&stats](
                const char* category,
                const char* name,
                const std::vector<double>& values,
                double mean) {
                stats["breakdown"][category][name]["mean_us"] = mean;
                stats["breakdown"][category][name]["std_us"] = calc_std(values, mean);
            };
            auto mean_of = [](const std::vector<double>& values) {
                double total = 0.0;
                for (double value : values) total += value;
                return values.empty() ? 0.0 : total / values.size();
            };
            save_breakdown("preprocess", "resize", prep_resize_list, mean_of(prep_resize_list));
            save_breakdown("preprocess", "padding", prep_padding_list, mean_of(prep_padding_list));
            save_breakdown("preprocess", "pack", prep_pack_list, mean_of(prep_pack_list));
            save_breakdown("preprocess", "total", prep_total_list, mean_of(prep_total_list));
            save_breakdown("infer", "input_copy", input_copy_list, mean_of(input_copy_list));
            save_breakdown("infer", "inputs_set", inputs_set_list, mean_of(inputs_set_list));
            save_breakdown("infer", "run_call", run_call_list, mean_of(run_call_list));
            save_breakdown("infer", "outputs_get", outputs_get_list, mean_of(outputs_get_list));
            save_breakdown("infer", "output_pack", output_pack_list, mean_of(output_pack_list));
            save_breakdown("infer", "outputs_release", outputs_release_list, mean_of(outputs_release_list));
            save_breakdown("infer", "total", infer_total_list, mean_of(infer_total_list));
        }

        std::ofstream f(json_path);
        if(f.is_open()){
            f << stats.dump(4);
            f.close();
            std::cout << "\n✅ Stats json saved to: " << json_path << std::endl;
        }else{
            std::cerr << "\n❌ Failed to write json to " << json_path << std::endl;
        }
    } else {
        std::cout << "\n[WARN] No frame processed, skip save json" << std::endl;
    }
}

int main(int argc, char* argv[])
{
    // 注册 SIGINT 信号处理器
    signal(SIGINT, sigint_handler);

    const std::string default_model_path = "models/rknn/yolo11s_640_split_int8.rknn";
    std::string model_path = default_model_path;
    bool breakdown_enabled = false;
    bool model_path_set = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--breakdown") {
            breakdown_enabled = true;
        } else if (!model_path_set) {
            model_path = arg;
            model_path_set = true;
        } else {
            std::cerr << "用法: " << argv[0]
                      << " [model_path] [--breakdown]" << std::endl;
            return -1;
        }
    }

    std::string video_path = "assets/regression/input.mp4";
    cv::VideoCapture cap(video_path);
    RknnEngine rknn_engine;
    rknn_engine.init(model_path);
    rknn_engine.input_setting(RKNN_TENSOR_INT8);

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
    std::thread t2(inference_thread,std::ref(rknn_engine), breakdown_enabled);

    t1.join();
    t2.join();

    cap.release();
    std::cout << "\n[INFO] Program exit gracefully" << std::endl;
    return 0;
}
