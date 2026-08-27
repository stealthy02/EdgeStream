#include "postprocess.h"
#include "preprocess.h"
#include "inference_engine.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <sys/stat.h>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include "rknn_engine.hpp"
#include <iostream>


using json = nlohmann::json;

// 工具：创建目录（兼容orangepi linux）
bool mkdir_if_not_exist(const std::string& dir)
{
    struct stat st;
    if (stat(dir.c_str(), &st) == 0) {
        return S_ISDIR(st.st_mode);
    }
    return mkdir(dir.c_str(), 0755) == 0;
}

// COCO类别映射
std::string get_class_name(int class_id)
{
    static const std::unordered_map<int, std::string> cls_map = {
        {0, "person"},
        {5, "bus"}
    };
    auto it = cls_map.find(class_id);
    if(it != cls_map.end()){
        return it->second;
    }
    return "";
}

bool save_detections_to_jsonl(const std::vector<Detection>& detections, const std::string& save_path)
{
    if(detections.empty())
    {
        printf("detections is empty\n");
        return false;
    }
    size_t last_slash = save_path.find_last_of('/');
    if(last_slash != std::string::npos)
    {
        std::string dir = save_path.substr(0, last_slash);
        if(!mkdir_if_not_exist(dir))
        {
            std::cerr << "Failed create dir: " << dir << std::endl;
            return false;
        }
    }

    std::ofstream fout(save_path);
    if(!fout.is_open())
    {
        std::cerr << "Failed open file: " << save_path << std::endl;
        return false;
    }

    for (size_t idx = 0; idx < detections.size(); idx++)
    {
        const auto& det = detections[idx];
        json j;
        j["class_id"] = det.class_id;
        j["confidence"] = det.score;
        j["detection_index"] = static_cast<int>(idx);
        j["xyxy"] = json::array({
            det.rectangle.x1,
            det.rectangle.y1,
            det.rectangle.x2,
            det.rectangle.y2
        });

        std::string cls_name = get_class_name(det.class_id);
        if(!cls_name.empty())
        {
            j["class_name"] = cls_name;
        }
        fout << j.dump() << "\n";
    }
    fout.close();
    return true;
}


void onnx_bus_test(
    const cv::Mat& image,
    std::string model_path,
    std::string save_path
){
    InferenceEngine inference_engine(model_path);
    auto input_shape = inference_engine.input_shape();
    PreprocessParameter preprocess_parameter = get_preprocess_parameter(image.cols ,image.rows, input_shape[3],input_shape[2]);
    std::vector<float> tensor_data(3*input_shape[2]*input_shape[3]);
    
    preprocess_image(image,preprocess_parameter,tensor_data, TENSOR_NCHW);

    auto inference_output_list = inference_engine.run(tensor_data);

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
        0.25f,
        0.70f,
        preprocess_parameter
    );

    if(!detections_original.empty())
    {
        size_t last_slash = save_path.find_last_of('/');
        if(last_slash != std::string::npos)
        {
            std::string dir = save_path.substr(0, last_slash);
            mkdir_if_not_exist(dir);
        }

        std::ofstream fout(save_path);
        if(!fout.is_open())
        {
            std::cerr << "Failed open file: " << save_path << std::endl;
            return;
        }

        for (size_t idx = 0; idx < detections_original.size(); idx++)
        {
            const auto& det = detections_original[idx];
            json j;
            j["class_id"] = det.class_id;
            j["confidence"] = det.score;
            j["detection_index"] = static_cast<int>(idx);
            j["xyxy"] = json::array({
                det.rectangle.x1,
                det.rectangle.y1,
                det.rectangle.x2,
                det.rectangle.y2
            });

            std::string cls_name = get_class_name(det.class_id);
            if(!cls_name.empty())
            {
                j["class_name"] = cls_name;
            }
            fout << j.dump() << "\n";
        }
        fout.close();
        std::cout << "✅ ONNX Detection json saved to: " << save_path << std::endl;
    }
}



inline int8_t fp_to_int8(float fp)
{
    const float  in_scale = 0.00392157f;
    const int8_t in_zp   = -128;
    float q = fp / in_scale + in_zp;
    // 四舍五入
    int32_t val = static_cast<int32_t>(nearbyint(q));
    // 钳位 int8 范围 [-128, 127]
    if(val < -128) val = -128;
    if(val > 127)  val = 127;
    // std::cout<<val<<" ";
    return static_cast<int8_t>(val);
}

void rknn_bus_test(
    const cv::Mat& image,
    std::string model_path,
    std::string save_path,
    bool quantization
){
    RknnEngine rknn_engine;
    int init_ret = rknn_engine.init(model_path);
    if (init_ret != 0)
    {
        std::cerr << "RKNN 模型初始化失败！model=" << model_path << std::endl;
        return;
    }

    // 自动读取模型输入张量信息
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
        return;
    }

    PreprocessParameter preprocess_parameter = get_preprocess_parameter(image.cols, image.rows, model_in_w, model_in_h);

    std::vector<float> output_data;
    int run_ret;
    if(quantization){
        std::vector<int8_t> tensor_data(channel * model_in_h * model_in_w);
        preprocess_image(image, preprocess_parameter, tensor_data, TENSOR_NHWC);
        // 推理
        rknn_engine.input_setting(RKNN_TENSOR_INT8);
        run_ret = rknn_engine.run(tensor_data.data(), output_data); 
    }else{
        std::vector<uint16_t> tensor_data(channel * model_in_h * model_in_w);
        preprocess_image(image, preprocess_parameter, tensor_data, TENSOR_NHWC);
        rknn_engine.input_setting(RKNN_TENSOR_FLOAT16);
        run_ret = rknn_engine.run(tensor_data.data(), output_data);
    }

    if (run_ret != 0)
    {
        std::cerr << "RKNN 推理失败！" << std::endl;
        return;
    }
    // 自动解析输出维度
    
    auto detections_original = postprocess_image(
        output_data.data(),
        84,
        8400,
        0.25f,
        0.70f,
        preprocess_parameter
    );

    if(save_detections_to_jsonl(detections_original, save_path))
    {
        std::cout << "✅ RKNN Detection json saved to: " << save_path << std::endl;
    }else{
        std::cout << "❌ RKNN Detection json save failed " << std::endl;
    }

}


int main(int argc, char* argv[]){
    bool run_onnx = true;
    bool run_rknn = true;

    // 解析命令行参数
    for(int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if(arg == "--onnx")
        {
            run_onnx = true;
            run_rknn = false;
        }
        else if(arg == "--rknn")
        {
            run_onnx = false;
            run_rknn = true;
        }
        else
        {
            std::cerr << "参数错误！支持参数： --onnx | --rknn \n不传参数默认全部运行" << std::endl;
            return -1;
        }
    }

    // 固定路径，后续你也可以改成命令行传图片/模型路径
    std::string image_path = "assets/regression/bus.jpg";

    std::string onnx_model_path = "models/onnx/yolo11s_640_split.onnx";
    std::string onnx_output = "artifacts/onnx/orangepi_dets.jsonl";

    std::string rknn_model_path = "models/rknn/yolo11s_640_split.rknn";
    std::string rknn_output = "artifacts/rknn/orangepi_dets.jsonl";

    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "错误：无法读取图片，请确认路径是否正确: " << image_path << std::endl;
        return -1;
    }

    if(run_onnx)
    {
        std::cout << "\n===== 开始运行 ONNX 推理 =====\n";
        onnx_bus_test(image, onnx_model_path, onnx_output);
    }
    if(run_rknn)
    {
        std::cout << "\n===== 开始运行 RKNN 推理 =====\n";
        rknn_bus_test(image, rknn_model_path, rknn_output,false);
        rknn_model_path = "models/rknn/yolo11s_640_split_int8.rknn";
        rknn_output = "artifacts/rknn/orangepi_int8_dets.jsonl";
        rknn_bus_test(image, rknn_model_path, rknn_output,true);
    }

    std::cout << "\n全部任务执行完成\n";
    return 0;
}
