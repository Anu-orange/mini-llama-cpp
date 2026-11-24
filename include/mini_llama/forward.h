// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_FORWARD_H_
#define INCLUDE_MINI_LLAMA_FORWARD_H_

#include "mini_llama/batch.h"
#include "mini_llama/context.h"
#include "mini_llama/model.h"

namespace mini_llama {

// Forward pass for a single token
// Returns logits: [vocab_size]
/**
 * @brief 该函数是处理一个token的起始位置，这里的token已经是由tokenizer得到的
 * 
 * @param ctx 
 * @param model 
 * @param token 
 * @return Tensor 
 */
Tensor ForwardToken(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    int token);

// Forward pass for a batch of tokens.
// Internally processes tokens sequentially and returns logits for the last
// token. This unifies prefill (multi-token) and Decode (single-token) paths.
Tensor ForwardBatch(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    const MiniBatch& batch);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_FORWARD_H_
