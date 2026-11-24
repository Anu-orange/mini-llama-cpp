// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_BACKEND_H_
#define INCLUDE_MINI_LLAMA_BACKEND_H_

#include <string>

namespace mini_llama {

/**
 * @brief 后端的枚举类型，分为CPU和GPU
 * 
 */
enum class BackendKind {
	kCpu,
	kCuda,
};

/**
 * @brief 后端配置中，默认使用CPU
 * 
 */
struct BackendConfig {
	BackendKind kind = BackendKind::kCpu;
	int device_id = 0;
	bool device_id_set = false; // 用户是否明确提供了--device参数
};

/**
 * @brief 根据传递的参数进行匹配，看是CPU还是GPU
 * 
 * @param text 
 * @param kind 
 * @return true 
 * @return false 
 */
bool ParseBackendKind(const std::string& text, BackendKind& kind);
std::string BackendKindName(BackendKind kind);

bool CudaBackendBuilt();
bool CudaBackendAvailable(int device_id = 0);
std::string CudaDeviceSummary(int device_id = 0);

// Throws when the requested backend cannot run in this build or on this host.
/**
 * @brief 运行前的配置校验
 * 
 * @param config 
 */
void ValidateBackend(const BackendConfig& config);

// Explains which execution path is active.
std::string BackendExecutionNote(const BackendConfig& config);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_BACKEND_H_
