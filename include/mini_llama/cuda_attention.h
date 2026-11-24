// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_CUDA_ATTENTION_H_
#define INCLUDE_MINI_LLAMA_CUDA_ATTENTION_H_

#include "mini_llama/cuda_kv_cache.h"
#include "mini_llama/cuda_tensor.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

bool CudaAttentionBuilt();

/**
 * @brief CUDA上的Attention注意力计算，输入数据需要上传
 * 
 * @param q 
 * @param kv_cache 
 * @param layer 
 * @param pos 
 * @param n_heads 
 * @param n_kv_heads 
 * @param head_dim 
 * @param device_id 
 * @return Tensor 
 */
Tensor CudaAttentionDecode(const Tensor& q, const CudaKvCache& kv_cache,
                           int layer, int pos, int n_heads, int n_kv_heads,
                           int head_dim, int device_id = 0);

/**
 * @brief 面向CUDA内部的attention注意力计算
 * 
 * @param q 
 * @param kv_cache 
 * @param layer 
 * @param pos 
 * @param n_heads 
 * @param n_kv_heads 
 * @param head_dim 
 * @param device_id 
 * @return CudaTensor 
 */
CudaTensor CudaAttentionDecodeDeviceInput(const CudaTensor& q,
                                          const CudaKvCache& kv_cache,
                                          int layer, int pos, int n_heads,
                                          int n_kv_heads, int head_dim,
                                          int device_id = 0);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_CUDA_ATTENTION_H_
