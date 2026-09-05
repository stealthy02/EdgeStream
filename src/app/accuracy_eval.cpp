// ============================================================================
// 【阶段 2 / 3】C++ 多图精度测试 —— 验证你自己写的预处理/推理/后处理代码
//
// 流水线（3 个后端，每张图各跑一次）：
//     ONNX (Ort C++)       → onnx.jsonl
//     RKNN FP16 (librknnrt) → rknn_fp16.jsonl
//     RKNN INT8 (librknnrt) → rknn_int8.jsonl
//
// 输出目录：与【阶段 1 generate_reference.py】+【阶段 3 accuracy_benchmark.py】三方对齐，
//          按后端分文件夹，固定无日期，可单独重跑而不影响其他后端。
//          artifacts/accuracy_eval/
//            ├── pytorch/   {stem}.jsonl    ← 阶段 1 已写
//            ├── onnx/      {stem}.jsonl    ← 本程序写
//            ├── rknn_fp16/ {stem}.jsonl    ← 本程序写
//            └── rknn_int8/ {stem}.jsonl    ← 本程序写
//
// 用法：
//   # 1) 先在 PC 端跑阶段 1（PT 参考）
//   python3 python/generate_reference.py --images-dir assets/regression/test_image
//
//   # 2) 在 OrangePi 板端（同步好 pytorch/ 后）跑阶段 2，默认输出到 artifacts/accuracy_eval：
//   ./accuracy_eval --images-dir assets/regression/test_image
//
//   # 板端没装 RKNN 驱动？只验证 ONNX C++ 链路：
//   ./accuracy_eval --images-dir assets/regression/test_image --onnx-only
// ============================================================================

#include "edgestream/core/file_io.h"
#include "edgestream/core/jsonl.h"
#include "edgestream/inference/onnx_engine.h"
#include "edgestream/inference/rknn_engine.h"
#include "edgestream/yolo/postprocess.h"
#include "edgestream/yolo/preprocess.h"
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

// ============================================================================
// ONNX 推理 —— 引擎外部传入，避免每次加载模型
// ============================================================================
static bool onnx_infer(
    OnnxEngine& engine,
    const cv::Mat& image,
    const std::string& save_path,
    float conf_threshold = 0.25f,
    float nms_iou = 0.70f)
{
    auto input_shape = engine.input_shape();
    PreprocessParameter pp = get_preprocess_parameter(
        image.cols, image.rows, input_shape[3], input_shape[2]
    );
    std::vector<float> tensor_data(3 * input_shape[2] * input_shape[3]);
    preprocess_image(image, pp, tensor_data, TENSOR_NCHW);

    auto output_list = engine.run(tensor_data);
    InferenceOutput merged = OnnxEngine::merge_split_outputs(std::move(output_list));

    auto detections = postprocess_image(
        merged.data.data(),
        static_cast<int>(merged.shape[1]),
        static_cast<int>(merged.shape[2]),
        conf_threshold, nms_iou, pp
    );

    if (save_detections_to_jsonl(detections, save_path)) {
        std::cout << "    onnx    → " << detections.size() << " dets";
        return true;
    } else {
        std::cout << "    onnx    → ❌ 保存失败";
        return false;
    }
}

// ============================================================================
// RKNN 推理 —— FP16 / INT8 由 quantization 参数区分
// ============================================================================
static bool rknn_infer(
    RknnEngine& engine,
    const cv::Mat& image,
    const std::string& save_path,
    bool quantization,
    float conf_threshold = 0.25f,
    float nms_iou = 0.70f)
{
    uint32_t in_h = 0, in_w = 0, ch = 0;
    if (!engine.get_input_hw_c(0, in_h, in_w, ch)) {
        std::cerr << "❌ 不支持的 RKNN 输入排布" << std::endl;
        return false;
    }

    PreprocessParameter pp = get_preprocess_parameter(image.cols, image.rows, in_w, in_h);
    std::vector<float> output_data;
    int run_ret;

    if (quantization) {
        std::vector<int8_t> tensor_data(ch * in_h * in_w);
        preprocess_image(image, pp, tensor_data, TENSOR_NHWC);
        engine.input_setting(RKNN_TENSOR_INT8);
        run_ret = engine.run(tensor_data.data(), output_data);
    } else {
        std::vector<uint16_t> tensor_data(ch * in_h * in_w);
        preprocess_image(image, pp, tensor_data, TENSOR_NHWC);
        engine.input_setting(RKNN_TENSOR_FLOAT16);
        run_ret = engine.run(tensor_data.data(), output_data);
    }
    if (run_ret != 0) {
        std::cerr << " ❌ RKNN 推理失败 ret=" << run_ret << std::endl;
        return false;
    }

    // split 模型 outputs 合并后 shape [1, 84, 8400]，此处直接给 (84, 8400) 让 postprocess 处理
    auto detections = postprocess_image(
        output_data.data(),
        84, 8400,
        conf_threshold, nms_iou, pp
    );
    const char* tag = quantization ? "rknn_i8" : "rknn_fp";
    if (save_detections_to_jsonl(detections, save_path)) {
        std::cout << "    " << tag << " → " << detections.size() << " dets";
        return true;
    } else {
        std::cout << "    " << tag << " → ❌ 保存失败";
        return false;
    }
}

// ============================================================================
// 扫描图片目录，返回有序 stem→full_path
// ============================================================================
static std::vector<std::pair<std::string, std::string>> scan_image_dir(const std::string& dir)
{
    std::vector<std::pair<std::string, std::string>> result;
    static const std::set<std::string> exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"};
    if (!fs::is_directory(dir)) {
        std::cerr << "❌ 图片目录不存在: " << dir << std::endl;
        return result;
    }
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        if (exts.count(ext)) {
            result.emplace_back(entry.path().stem().string(),
                                entry.path().string());
        }
    }
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b){ return a.first < b.first; });
    return result;
}

// ============================================================================
// 参数解析（轻量手动版，不引 boost）
// ============================================================================
struct Args {
    std::string output_root = "artifacts/accuracy_eval";
    std::string images_dir = "assets/regression/test_image";
    std::string onnx_model  = "models/onnx/yolo11s_640_split.onnx";
    std::string rknn_fp16   = "models/rknn/yolo11s_640_split.rknn";
    std::string rknn_int8   = "models/rknn/yolo11s_640_split_int8.rknn";
    bool run_onnx = true;
    bool run_rknn = true;
    float conf = 0.25f;
    float iou  = 0.70f;
};

static void print_usage(const char* prog)
{
    std::cerr
        << "用法: " << prog << " [选项]\n\n"
        << "  --output-root <DIR>    输出根目录 (默认: artifacts/accuracy_eval)\n"
        << "                          结果按后端写入 {root}/onnx|rknn_fp16|rknn_int8/{stem}.jsonl\n"
        << "  --images-dir <DIR>     测试图片目录 (默认: assets/regression/test_image)\n"
        << "  --onnx-model <PATH>    ONNX split 模型路径\n"
        << "  --rknn-fp16 <PATH>     RKNN FP16 模型路径\n"
        << "  --rknn-int8 <PATH>     RKNN INT8 模型路径\n"
        << "  --onnx-only            只验证 ONNX C++ 链路 (跳过 RKNN)\n"
        << "  --rknn-only            只验证 RKNN C++ 链路 (跳过 ONNX)\n"
        << "  --conf <float>         置信度阈值 (默认 0.25)\n"
        << "  --iou  <float>         NMS IoU 阈值 (默认 0.70)\n"
        << "  -h / --help            显示此帮助\n";
}

static Args parse_args(int argc, char* argv[])
{
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto need_next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("参数 ") + name + " 缺少值");
            }
            return std::string(argv[++i]);
        };
        if (s == "-h" || s == "--help")        { print_usage(argv[0]); exit(0); }
        else if (s == "--output-root")         { a.output_root = need_next("--output-root"); }
        else if (s == "--images-dir")          { a.images_dir = need_next("--images-dir"); }
        else if (s == "--onnx-model")          { a.onnx_model = need_next("--onnx-model"); }
        else if (s == "--rknn-fp16")           { a.rknn_fp16  = need_next("--rknn-fp16"); }
        else if (s == "--rknn-int8")           { a.rknn_int8  = need_next("--rknn-int8"); }
        else if (s == "--onnx-only")           { a.run_onnx = true;  a.run_rknn = false; }
        else if (s == "--rknn-only")           { a.run_onnx = false; a.run_rknn = true; }
        // 保持和原版 bus_test 完全兼容的两个老参数
        else if (s == "--onnx")                { a.run_onnx = true;  a.run_rknn = false; }
        else if (s == "--rknn")                { a.run_onnx = false; a.run_rknn = true; }
        else if (s == "--conf")                { a.conf = std::stof(need_next("--conf")); }
        else if (s == "--iou")                 { a.iou  = std::stof(need_next("--iou")); }
        else {
            std::cerr << "⚠️  未知参数: " << s << "\n\n";
            print_usage(argv[0]);
            throw std::runtime_error("参数解析失败");
        }
    }
    return a;
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char* argv[])
{
    Args args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << std::endl;
        return 1;
    }

    if (!mkdir_if_not_exist(args.output_root)) {
        std::cerr << "❌ output_root 不存在且无法创建: " << args.output_root << std::endl;
        return 1;
    }
    // 按后端建子目录（pytorch/ 由阶段 1 写，这里只建 C++ 负责的三个）
    std::string onnx_dir    = args.output_root + "/onnx";
    std::string rknn_fp_dir = args.output_root + "/rknn_fp16";
    std::string rknn_i8_dir = args.output_root + "/rknn_int8";
    if (args.run_onnx) mkdir_if_not_exist(onnx_dir);
    if (args.run_rknn) { mkdir_if_not_exist(rknn_fp_dir); mkdir_if_not_exist(rknn_i8_dir); }

    std::cout << "========================================" << std::endl;
    std::cout << "【阶段 2 / 3】C++ 多图精度验证" << std::endl;
    std::cout << "  output_root = " << args.output_root << std::endl;
    std::cout << "  images    = " << args.images_dir << std::endl;
    std::cout << "  后端      = "
              << (args.run_onnx ? "ONNX " : "")
              << (args.run_rknn ? "RKNN-FP16 RKNN-INT8 " : "") << std::endl;
    std::cout << "  conf/iou  = " << args.conf << " / " << args.iou << std::endl;
    std::cout << "========================================" << std::endl;

    // 扫描图片
    auto images = scan_image_dir(args.images_dir);
    if (images.empty()) {
        std::cerr << "❌ 图片目录里没有找到可用图片: " << args.images_dir << std::endl;
        return 1;
    }
    std::cout << "[信息] 找到 " << images.size() << " 张图片\n" << std::endl;

    // —— ONNX 引擎：只加载一次，循环复用（省时间）——
    std::unique_ptr<OnnxEngine> onnx_engine;
    if (args.run_onnx) {
        if (!fs::exists(args.onnx_model)) {
            std::cerr << "❌ ONNX 模型不存在: " << args.onnx_model << std::endl;
            return 1;
        }
        std::cout << "[加载] ONNX 模型: " << args.onnx_model << std::endl;
        onnx_engine = std::make_unique<OnnxEngine>(args.onnx_model);
    }
    // —— RKNN 引擎：FP16 / INT8 各加载一次，循环复用（和 ONNX 一致）——
    std::unique_ptr<RknnEngine> rknn_fp_engine;
    std::unique_ptr<RknnEngine> rknn_i8_engine;
    if (args.run_rknn) {
        for (const auto& p : {args.rknn_fp16, args.rknn_int8}) {
            if (!fs::exists(p)) {
                std::cerr << "❌ RKNN 模型不存在: " << p << std::endl;
                return 1;
            }
        }
        std::cout << "[加载] RKNN FP16 模型: " << args.rknn_fp16 << std::endl;
        rknn_fp_engine = std::make_unique<RknnEngine>();
        if (rknn_fp_engine->init(args.rknn_fp16) != 0) {
            std::cerr << "❌ RKNN FP16 初始化失败: " << args.rknn_fp16 << std::endl;
            return 1;
        }
        std::cout << "[加载] RKNN INT8 模型: " << args.rknn_int8 << std::endl;
        rknn_i8_engine = std::make_unique<RknnEngine>();
        if (rknn_i8_engine->init(args.rknn_int8) != 0) {
            std::cerr << "❌ RKNN INT8 初始化失败: " << args.rknn_int8 << std::endl;
            return 1;
        }
        std::cout << std::endl;
    }

    // —— 逐图循环 ——
    int ok_cnt = 0;
    for (std::size_t i = 0; i < images.size(); ++i) {
        const auto& stem = images[i].first;
        const auto& img_path = images[i].second;
        std::cout << "[" << (i + 1) << "/" << images.size() << "] "
                  << stem << " (" << fs::path(img_path).filename().string() << ")" << std::endl;

        cv::Mat image = cv::imread(img_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cout << "  ❌ 无法读取图片，跳过" << std::endl;
            continue;
        }

        bool stage_ok = true;
        if (args.run_onnx) {
            std::string out = onnx_dir + "/" + stem + ".jsonl";
            stage_ok &= onnx_infer(*onnx_engine, image, out, args.conf, args.iou);
            std::cout << std::endl;
        }
        if (args.run_rknn) {
            std::string out_fp = rknn_fp_dir + "/" + stem + ".jsonl";
            stage_ok &= rknn_infer(*rknn_fp_engine, image, out_fp, /*quant=*/false, args.conf, args.iou);
            std::cout << std::endl;
            std::string out_i8 = rknn_i8_dir + "/" + stem + ".jsonl";
            stage_ok &= rknn_infer(*rknn_i8_engine, image, out_i8, /*quant=*/true, args.conf, args.iou);
            std::cout << std::endl;
        }
        if (stage_ok) {
            ++ok_cnt;
        } else {
            std::cerr << "  ❌ 后端输出失败" << std::endl;
        }
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << (ok_cnt == static_cast<int>(images.size()) ? "✅" : "❌")
              << " 完成 " << ok_cnt << "/" << images.size() << " 张" << std::endl;
    std::cout << "📁 输出根目录: " << args.output_root << std::endl;
    std::cout << "▶  下一步跑阶段 3:" << std::endl;
    std::cout << "   python3 scripts/accuracy_benchmark.py" << std::endl;
    return ok_cnt == static_cast<int>(images.size()) ? 0 : 1;
}
