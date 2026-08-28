#!/bin/bash
# release_check.sh - P0-2 质量门禁脚本
# 串联 6 步：环境快照 → SHA256 → 构建 → CTest → 固定图回归 → benchmark
# 任何一步失败立即退出并打印失败命令

# set -e —— 任何命令返回非零（失败）则立即退出脚本。比如cat一个不存在的文件
# set -u —— 使用未定义变量时触发错误并退出。比如$value没有定义, 脚本中出现了$value就终止
# set -o pipefail  管道中任何命令失败，整个管道返回失败状态（而不仅仅是最后一条）
set -euo pipefail

# ========== 全局变量 ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"
RELEASE_DIR="${PROJECT_ROOT}/artifacts/release/${TIMESTAMP}"
BUILD_DIR="${PROJECT_ROOT}/build"
LOG_DIR="${RELEASE_DIR}/logs"

# ========== 错误处理 ==========
# trap 捕获错误，打印失败命令和行号
trap 'echo "❌ FAILED at line $LINENO: $BASH_COMMAND" >&2; exit 1' ERR

# ========== 工具函数 ==========
log_step() {
    echo ""
    echo "=========================================="
    echo "[STEP $1/$2] $3"
    echo "=========================================="
}

# ========== 初始化 ==========
init() {
    mkdir -p "${RELEASE_DIR}"
    mkdir -p "${LOG_DIR}"
    echo "📁 发布目录: ${RELEASE_DIR}"
    echo "🕐 时间戳: ${TIMESTAMP}"
}

# ========== 步骤① 环境快照 ==========
step1_env_snapshot() {
    log_step 1 6 "collecting environment snapshot..."
    
    local env_json="${RELEASE_DIR}/env.json"
    
    # 收集各项信息
    local git_commit=$(git rev-parse HEAD 2>/dev/null || echo "N/A")
    local uname_info=$(uname -a)
    local cmake_version=$(cmake --version 2>/dev/null | head -1 || echo "N/A")
    local gcc_version=$(gcc --version 2>/dev/null | head -1 || echo "N/A")
    local gpp_version=$(g++ --version 2>/dev/null | head -1 || echo "N/A")
    local cpu_temp=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo "N/A")
    # 先尝试 x86 的 model name
    local cpu_model=$(grep -m1 "model name" /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs 2>/dev/null || true)

    # 如果没有（ARM 架构），用 implementer + part 识别
    if [[ -z "${cpu_model}" ]]; then
        local implementer=$(grep -m1 "CPU implementer" /proc/cpuinfo 2>/dev/null | awk '{print $3}' || echo "")
        local part=$(grep -m1 "CPU part" /proc/cpuinfo 2>/dev/null | awk '{print $3}' || echo "")
        
        if [[ "${implementer}" == "0x41" && "${part}" == "0xd0b" ]]; then
            cpu_model="ARM Cortex-A76"
        elif [[ "${implementer}" == "0x41" && "${part}" == "0xd05" ]]; then
            cpu_model="ARM Cortex-A55"
        elif [[ "${implementer}" == "0x41" && "${part}" == "0xd05" ]]; then
            cpu_model="ARM Cortex-A55"
        else
            cpu_model="ARM implementer=${implementer} part=${part}"
        fi
    fi
    local cpu_cores=$(grep -c "^processor" /proc/cpuinfo 2>/dev/null || echo "N/A")
    
    # 生成 JSON
    cat > "${env_json}" << JSONEOF
{
    "timestamp": "${TIMESTAMP}",
    "git_commit": "${git_commit}",
    "system": {
        "uname": "${uname_info}",
        "cpu_model": "${cpu_model}",
        "cpu_cores": "${cpu_cores}",
        "cpu_temp_millidegree": "${cpu_temp}"
    },
    "toolchain": {
        "cmake": "${cmake_version}",
        "gcc": "${gcc_version}",
        "gpp": "${gpp_version}"
    }
}
JSONEOF
    
    echo "✅ 环境快照已保存: ${env_json}"
}
# ========== 步骤② SHA256 清单 ==========
step2_sha256_manifest() {
    log_step 2 6 "computing SHA256 checksums..."
    
    local manifest_json="${RELEASE_DIR}/manifest.json"
    
    # 检查所有文件是否存在
    local files_to_hash=(
        "${PROJECT_ROOT}/models/rknn/yolo11s_640_split_int8.rknn"
        "${PROJECT_ROOT}/models/rknn/yolo11s_640_split.rknn"
        "${PROJECT_ROOT}/models/onnx/yolo11s_640.onnx"
        "${PROJECT_ROOT}/models/onnx/yolo11s_640_split.onnx"
        "${PROJECT_ROOT}/assets/regression/input.mp4"
        "/usr/lib/librknnrt.so"
    )
    
    for file in "${files_to_hash[@]}"; do
        if [[ ! -f "$file" ]]; then
            echo "❌ 文件不存在: $file" >&2
            return 1
        fi
    done
    
    # 计算 SHA256
    echo "{" > "${manifest_json}"
    echo '    "files": [' >> "${manifest_json}"
    
    local file_count=${#files_to_hash[@]}
    local index=0
    
    for file in "${files_to_hash[@]}"; do
        index=$((index + 1))
        local rel_path="${file#${PROJECT_ROOT}/}"
        local sha256=$(sha256sum "$file" | awk '{print $1}')
        local size=$(stat -c %s "$file" 2>/dev/null || stat -f %z "$file" 2>/dev/null)
        
        # 判断是否最后一个元素
        local comma=""
        if [[ $index -lt $file_count ]]; then
            comma=","
        fi
        
        cat >> "${manifest_json}" << JSONEOF
        {
            "path": "${rel_path}",
            "sha256": "${sha256}",
            "size_bytes": ${size}
        }${comma}
JSONEOF
    done
    
    echo '    ]' >> "${manifest_json}"
    echo "}" >> "${manifest_json}"
    
    echo "✅ SHA256 清单已保存: ${manifest_json}"
}

# ========== 步骤③ Release 构建 ==========
step3_release_build() {
    log_step 3 6 "building release..."
    
    local build_log="${LOG_DIR}/build.log"
    
    # 配置 CMake（如果是首次构建）
    if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
        echo "🔧 首次构建，运行 cmake configure..."
        cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release 2>&1 | tee "${build_log}"
    else
        echo "🔧 已存在构建目录，跳过 configure..."
        cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release 2>&1 | tee "${build_log}"
    fi
    
    # 构建
    echo "🔨 开始编译..."
    cmake --build "${BUILD_DIR}" -j$(nproc) 2>&1 | tee -a "${build_log}"
    
    echo "✅ Release 构建完成，日志: ${build_log}"
}
# ========== 步骤④ CTest ==========
step4_ctest() {
    log_step 4 6 "running CTest..."
    
    local ctest_log="${LOG_DIR}/ctest.log"
    
    # 运行 CTest
    ctest --test-dir "${BUILD_DIR}" --output-on-failure 2>&1 | tee "${ctest_log}"
    
    # 检查是否有失败的测试
    if grep -q "Errors" "${ctest_log}"; then
        echo "❌ CTest 有失败测试！" >&2
        return 1
    fi
    
    echo "✅ CTest 全部通过，日志: ${ctest_log}"
}

# ========== 主流程 ==========
main() {
    init
    step1_env_snapshot
    step2_sha256_manifest
    step3_release_build
    step4_ctest
    
    echo ""
    echo "=========================================="
    echo "✅ 步骤①②③④ 完成"
    echo "=========================================="
}

# 执行主流程
main "$@"