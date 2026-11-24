// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_CONTEXT_H_
#define INCLUDE_MINI_LLAMA_CONTEXT_H_

#include <vector>

#include "mini_llama/cuda_kv_cache.h"
#include "mini_llama/kv_cache.h"
#include "mini_llama/model.h"

namespace mini_llama {

// Inference context: holds KV cache, current position, token history, and
// stats.
struct MiniLlamaContext {
	const MiniLlamaModel* model = nullptr; // 关联正在使用的模型
	KvCache kv_cache; // 维护CPU的推理历史K,V
	CudaKvCache cuda_kv_cache; // 维护GPU的推理历史K,V
	int pos = 0; // 当前处理的位置

	// All tokens that have been fed through this context (including prefill).
	std::vector<int> token_history; // 记录已送入上下文计算的token

	// Stats
	int n_prefill_tokens = 0;
	int n_decode_tokens = 0;

	MiniLlamaContext() = default;
	MiniLlamaContext(const MiniLlamaContext&) = delete;
	MiniLlamaContext& operator=(const MiniLlamaContext&) = delete;
	MiniLlamaContext(MiniLlamaContext&&) noexcept = default;
	MiniLlamaContext& operator=(MiniLlamaContext&&) noexcept = default;
	explicit MiniLlamaContext(const MiniLlamaModel* model);
};

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_CONTEXT_H_
