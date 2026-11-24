// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/backend.h"

#include <stdexcept>
#include <string>

#include "mini_llama/cuda_runtime.h"

namespace mini_llama {

bool ParseBackendKind(const std::string& text, BackendKind& kind) {
	if (text == "cpu") {
		kind = BackendKind::kCpu;
		return true;
	}
	if (text == "cuda") {
		kind = BackendKind::kCuda;
		return true;
	}
	return false;
}

std::string BackendKindName(BackendKind kind) {
	switch (kind) {
		case BackendKind::kCpu:
			return "cpu";
		case BackendKind::kCuda:
			return "cuda";
	}
	return "unknown";
}

bool CudaBackendBuilt() { return CudaRuntimeBuilt(); }

bool CudaBackendAvailable(int device_id) {
	try {
		int count = CudaDeviceCount();
		return device_id >= 0 && device_id < count;
	} catch (const std::exception&) {
		return false;
	}
}

std::string CudaDeviceSummary(int device_id) {
	try {
		return CudaFormatDeviceInfo(CudaGetDeviceInfo(device_id));
	} catch (const std::exception& e) {
		return e.what();
	}
}

void ValidateBackend(const BackendConfig& config) {
	// 1、CPU后端下禁止使用--device
	if (config.kind == BackendKind::kCpu) {
		// 如果当前是CPU模式，但是用户明确传递了--device参数的话
		// 那么此时就需要抛出错误
		if (config.device_id_set) {
			throw std::runtime_error(
			    "--device can only be used with --backend cuda.");
		}
		return;
	}

	// 2、CUDA后端下，必须确保开启了MINI_LLAMA_CUDA宏
	if (config.kind == BackendKind::kCuda) {
		if (!CudaBackendBuilt()) {
			throw std::runtime_error(
			    "CUDA backend was not built. Reconfigure with "
			    "-DMINI_LLAMA_CUDA=ON "
			    "on a NVIDIA CUDA machine.");
		}

		// 3、CUDA后端指定的device_id在当前系统中真实可用
		if (!CudaBackendAvailable(config.device_id)) {
			throw std::runtime_error(CudaDeviceSummary(config.device_id));
		}
		return;
	}

	throw std::runtime_error("unknown backend");
}

std::string BackendExecutionNote(const BackendConfig& config) {
	if (config.kind == BackendKind::kCpu) {
		return "compute: cpu";
	}
	return "compute: cuda Linear (F32/Q8_0/Q4_0/Q4_1 where uploaded) + CUDA "
	       "attention over GPU KV cache + device-resident Decode path; CPU "
	       "sampler";
}

}  // namespace mini_llama
