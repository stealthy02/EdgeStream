#include "rknn_engine.hpp"
int main(){
    RknnEngine rknn_engine;
    std::string rknn_model_path = "models/rknn/yolo11s_640_int8.rknn";
    rknn_engine.init(rknn_model_path);
    rknn_engine.print_model_tensor_info();
    return 0;
}