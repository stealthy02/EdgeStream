// ============================================================================
// 【阶段 2 / 3】C++ 多图精度测试 —— 验证你自己写的预处理/推理/后处理代码
//
// 流水线（3 个后端，每张图各跑一次）：
//     ONNX (Ort C++)       → onnx.jsonl
//     RKNN FP16 (librknnrt) → rknn_fp16.jsonl
//     RKNN INT8 (librknnrt) → rknn_int8.jsonl
//
// 输出目录：必须指向【阶段 1 generate_reference.py】生成的同一个 run_dir，
//          这样【阶段 3 accuracy_benchmark.py】直接扫一个目录就能对比 4 份结果。
//          artifacts/accuracy_eval/{run_id}/
//            ├── {image_stem_1}/
//            │   ├── pt.jsonl         ← 阶段 1 已写
//            │   ├── onnx.jsonl       ← 本程序写
//            │   ├── rknn_fp16.jsonl  ← 本程序写
//            │   └── rknn_int8.jsonl  ← 本程序写
//
// 用法：
//   # 1) 先在 PC 端跑阶段 1（PT 参考），拿到 RUN_DIR
//   python3 python/generate_reference.py --images-dir assets/regression/coco8
//
//   # 2) 在 OrangePi 板端拷贝好同目录结构后，跑阶段 2：
//   ./accuracy_eval --run-dir artifacts/accuracy_eval/20260828_120000 --images-dir assets/regression/coco8
//
//   # 板端没装 RKNN 驱动？只验证 ONNX C++ 链路：
//   ./accuracy_eval --run-dir ... --images-dir ... --onnx-only
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
static void onnx_infer(
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
    } else {
        std::cout << "    onnx    → ❌ 保存失败";
    }
}

// ============================================================================
// RKNN 推理 —— FP16 / INT8 由 quantization 参数区分
// ============================================================================
static void rknn_infer(
    const std::string& model_path,
    const cv::Mat& image,
    const std::string& save_path,
    bool quantization,
    float conf_threshold = 0.25f,
    float nms_iou = 0.70f)
{
    RknnEngine rknn_engine;
    int init_ret = rknn_engine.init(model_path);
    if (init_ret != 0) {
        std::cerr << "\n❌ RKNN 初始化失败: " << model_path << std::endl;
        return;
    }

    uint32_t in_h = 0, in_w = 0, ch = 0;
    if (!rknn_engine.get_input_hw_c(0, in_h, in_w, ch)) {
        std::cerr << "❌ 不支持的 RKNN 输入排布" << std::endl;
        return;
    }

    PreprocessParameter pp = get_preprocess_parameter(image.cols, image.rows, in_w, in_h);
    std::vector<float> output_data;
    int run_ret;

    if (quantization) {
        std::vector<int8_t> tensor_data(ch * in_h * in_w);
        preprocess_image(image, pp, tensor_data, TENSOR_NHWC);
        rknn_engine.input_setting(RKNN_TENSOR_INT8);
        run_ret = rknn_engine.run(tensor_data.data(), output_data);
    } else {
        std::vector<uint16_t> tensor_data(ch * in_h * in_w);
        preprocess_image(image, pp, tensor_data, TENSOR_NHWC);
        rknn_engine.input_setting(RKNN_TENSOR_FLOAT16);
        run_ret = rknn_engine.run(tensor_data.data(), output_data);
    }
    if (run_ret != 0) {
        std::cerr << " ❌ RKNN 推理失败 ret=" << run_ret << std::endl;
        return;
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
    } else {
        std::cout << "    " << tag << " → ❌ 保存失败";
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
    std::string run_dir;
    std::string images_dir = "assets/regression/coco8";
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
        << "用法: " << prog << " --run-dir <DIR> [选项]\n\n"
        << "  --run-dir <DIR>        【必填】阶段 1 generate_reference.py 输出的 run 目录\n"
        << "                          例: artifacts/accuracy_eval/20260828_120000\n"
        << "  --images-dir <DIR>     测试图片目录 (默认: assets/regression/coco8)\n"
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
        else if (s == "--run-dir")             { a.run_dir = need_next("--run-dir"); }
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
    if (a.run_dir.empty()) {
        print_usage(argv[0]);
        throw std::runtime_error("--run-dir 参数必填");
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

    if (!mkdir_if_not_exist(args.run_dir)) {
        std::cerr << "❌ run_dir 不存在且无法创建: " << args.run_dir << std::endl;
        return 1;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "【阶段 2 / 3】C++ 多图精度验证" << std::endl;
    std::cout << "  run_dir   = " << args.run_dir << std::endl;
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
    // RKNN 模型文件检查
    if (args.run_rknn) {
        for (const auto& p : {args.rknn_fp16, args.rknn_int8}) {
            if (!fs::exists(p)) {
                std::cerr << "❌ RKNN 模型不存在: " << p << std::endl;
                return 1;
            }
        }
        std::cout << "[加载] RKNN 模型将在每次推理时单独 init/release (稳定优先)\n"
                  << "       FP16=" << args.rknn_fp16 << "\n"
                  << "       INT8=" << args.rknn_int8 << "\n" << std::endl;
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

        std::string img_out_dir = args.run_dir + "/" + stem;
        mkdir_if_not_exist(img_out_dir);

        bool stage_ok = true;
        if (args.run_onnx) {
            std::string out = img_out_dir + "/onnx.jsonl";
            onnx_infer(*onnx_engine, image, out, args.conf, args.iou);
            std::cout << std::endl;
        }
        if (args.run_rknn) {
            std::string out_fp = img_out_dir + "/rknn_fp16.jsonl";
            rknn_infer(args.rknn_fp16, image, out_fp, /*quant=*/false, args.conf, args.iou);
            std::cout << std::endl;
            std::string out_i8 = img_out_dir + "/rknn_int8.jsonl";
            rknn_infer(args.rknn_int8, image, out_i8, /*quant=*/true,  args.conf, args.iou);
            std::cout << std::endl;
        }
        if (stage_ok) ++ok_cnt;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "✅ 全部完成 " << ok_cnt << "/" << images.size() << " 张" << std::endl;
    std::cout << "📁 输出目录: " << args.run_dir << std::endl;
    std::cout << "▶  下一步跑阶段 3:" << std::endl;
    std::cout << "   python3 scripts/accuracy_benchmark.py --run-dir " << args.run_dir << std::endl;
    return 0;
}
