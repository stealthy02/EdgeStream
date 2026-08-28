#pragma once

#include <csignal>

// 全局停止标志，由 SIGINT 处理器置位。
extern volatile std::sig_atomic_t g_stop_requested;

// 安装 SIGINT 处理器，使 g_stop_requested 在 Ctrl+C 时被置 1。
void install_sigint_handler();

// 当前是否收到停止请求。
bool is_stop_requested();
