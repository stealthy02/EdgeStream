#!/usr/bin/env python3
"""
下载 YOLO11s 模型，并导出两种标准 ONNX 模型：
1. 单输出 ONNX（原始导出，一个输出张量）
2. 拆分输出 ONNX（将原始输出切分为 boxes 框坐标 和 scores 置信度+类别 两个独立输出）
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    """
    主执行函数：参数解析 -> 下载/加载权重 -> 导出标准ONNX -> 生成分割输出ONNX
    返回值：0 代表程序正常结束
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("models/pytorch/yolo11s.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/onnx"))
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    # 创建目录
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 判断本地是否已经存在权重文件
    weights_were_missing = not args.weights.exists()
    # 如果本地有权重则加载本地文件；缺失则传入模型名称，ultralytics会自动下载
    model = YOLO(str(args.weights) if not weights_were_missing else "yolo11s.pt")

    # 如果刚刚下载了权重，把下载到当前目录的 yolo11s.pt 复制到配置的 weights 路径下
    if weights_were_missing:
        downloaded = Path("yolo11s.pt")
        # 判断下载文件存在，并且和目标路径不是同一个文件，避免重复拷贝
        if downloaded.is_file() and downloaded.resolve() != args.weights.resolve():
            shutil.copy2(downloaded, args.weights)

    # 导出原始单输出 ONNX 模型
    exported = Path(model.export(format="onnx", imgsz=args.imgsz, batch=1, dynamic=False,
                                 simplify=False, opset=12))

    # 定义单输出ONNX最终保存路径
    single = args.output_dir / f"yolo11s_{args.imgsz}.onnx"
    # 如果导出文件和目标路径不一致，复制到目标目录
    if exported.resolve() != single.resolve():
        shutil.copy2(exported, single)

    # 定义拆分输出（boxes + scores双输出）ONNX保存路径
    split = args.output_dir / f"yolo11s_{args.imgsz}_split.onnx"
    # 调用函数：基于单输出onnx构造拆分双输出onnx模型
    make_split_model(single, split)

    # 打印两个模型保存位置
    print(f"single output: {single}\nsplit output:  {split}")
    return 0


def make_split_model(source: Path, destination: Path) -> None:
    """
    【核心函数】将 YOLO 原始单输出 ONNX 修改为双输出 ONNX
    YOLO11 原始输出张量形状: [batch, num_attributes, num_dets]
    num_attributes 前4维：x,y,w,h 检测框坐标
    剩余维度：各个类别的置信分数
    本函数通过 Slice 算子把一个输出切成 boxes 和 scores 两个输出

    参数:
        source: 输入，原始单输出ONNX文件路径
        destination: 输出，拆分后的双输出ONNX保存路径
    """
    import onnx
    from onnx import helper, numpy_helper
    import numpy as np

    # 加载原始onnx模型
    model = onnx.load(str(source))
    graph = model.graph  # 获取模型计算图
    original = graph.output[0]  # 获取原始唯一输出张量信息

    # 读取原始输出张量各维度尺寸
    shape = [d.dim_value for d in original.type.tensor_type.shape.dim]
    # 合法性校验：YOLO输出必须是3维，且属性维度至少大于4（xywh + 至少一类分数）
    if len(shape) != 3 or shape[1] < 5:
        raise RuntimeError(f"Unexpected YOLO output shape: {shape}")
    print(shape)
    original_name = original.name  # 原始输出张量名称

    # 构造 Slice 切片算子所需的常量初始化器
    # boxes：取第1维 [0:4]，对应 x,y,w,h
    starts = numpy_helper.from_array(np.array([0], dtype=np.int64), "boxes_start")
    ends = numpy_helper.from_array(np.array([4], dtype=np.int64), "boxes_end")
    # scores：取第1维 [4:end]，对应各类置信度
    score_start = numpy_helper.from_array(np.array([4], dtype=np.int64), "scores_start")
    score_end = numpy_helper.from_array(np.array([shape[1]], dtype=np.int64), "scores_end")
    # 指定切片维度：axis=1，在第二个维度上做切分
    axis = numpy_helper.from_array(np.array([1], dtype=np.int64), "split_axis")

    # 将常量加入计算图初始化器
    graph.initializer.extend([starts, ends, score_start, score_end, axis])

    # 新增两个 Slice 节点，用来分割原始输出张量
    graph.node.extend([
        # 切片得到 boxes: [batch,4,num_dets]
        helper.make_node("Slice", [original_name, "boxes_start", "boxes_end", "split_axis"], ["boxes"]),
        # 切片得到 scores: [batch, num_classes, num_dets]
        helper.make_node("Slice", [original_name, "scores_start", "scores_end", "split_axis"], ["scores"]),
    ])

    # 删除原来唯一的输出节点
    graph.output.remove(original)
    # 添加两个新输出：boxes 和 scores
    graph.output.extend([
        helper.make_tensor_value_info("boxes", onnx.TensorProto.FLOAT, [1, 4, shape[2]]),
        helper.make_tensor_value_info("scores", onnx.TensorProto.FLOAT, [1, shape[1] - 4, shape[2]]),
    ])

    # ONNX 模型合法性校验，检查修改后的图是否合规
    onnx.checker.check_model(model)
    # 保存修改完成的双输出onnx模型
    onnx.save(model, str(destination))


if __name__ == "__main__":
    # 程序入口，调用主函数并使用返回值退出程序
    raise SystemExit(main())
