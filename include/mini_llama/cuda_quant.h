// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_CUDA_QUANT_H_
#define INCLUDE_MINI_LLAMA_CUDA_QUANT_H_

#include <vector>

#include "mini_llama/cuda_tensor.h"
#include "mini_llama/quantized_tensor.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

bool CudaQuantBuilt();

Tensor CudaQ80Linear(const Tensor& x, const std::vector<BlockQ80>& weight,
                     const std::vector<int>& weight_shape,
                     const Tensor* bias = nullptr, int device_id = 0);

Tensor CudaQ80LinearDeviceWeight(const Tensor& x, const void* q8_0_device,
                                 size_t q8_0_block_count,
                                 const std::vector<int>& weight_shape,
                                 const Tensor* bias = nullptr,
                                 int device_id = 0);

CudaTensor CudaQ80LinearDeviceInput(const CudaTensor& x,
                                    const void* q8_0_device,
                                    size_t q8_0_block_count,
                                    const std::vector<int>& weight_shape,
                                    int device_id = 0);

/**
 * @brief 用于接收主机端的量化数据，并将输入x和权重上传，GPU kernel调用，最终的结果下载
 * 
 * @param x 
 * @param weight 
 * @param weight_shape 
 * @param bias 
 * @param device_id 
 * @return Tensor 
 */
Tensor CudaQ40Linear(const Tensor& x, const std::vector<BlockQ40>& weight,
                     const std::vector<int>& weight_shape,
                     const Tensor* bias = nullptr, int device_id = 0);

/**
 * @brief 权重已经存在于GPU显存中，只需要将输入张量x进行上传
 * 
 * @param x 
 * @param q4_0_device 
 * @param q4_0_block_count 
 * @param weight_shape 
 * @param bias 
 * @param device_id 
 * @return Tensor 
 */
Tensor CudaQ40LinearDeviceWeight(const Tensor& x, const void* q4_0_device,
                                 size_t q4_0_block_count,
                                 const std::vector<int>& weight_shape,
                                 const Tensor* bias = nullptr,
                                 int device_id = 0);

/**
 * @brief 权重和输入张量都在GPU显存上
 * 
 * @param x 
 * @param q4_0_device 
 * @param q4_0_block_count 
 * @param weight_shape 
 * @param device_id 
 * @return CudaTensor 
 */
CudaTensor CudaQ40LinearDeviceInput(const CudaTensor& x,
                                    const void* q4_0_device,
                                    size_t q4_0_block_count,
                                    const std::vector<int>& weight_shape,
                                    int device_id = 0);

Tensor CudaQ41Linear(const Tensor& x, const std::vector<BlockQ41>& weight,
                     const std::vector<int>& weight_shape,
                     const Tensor* bias = nullptr, int device_id = 0);

Tensor CudaQ41LinearDeviceWeight(const Tensor& x, const void* q4_1_device,
                                 size_t q4_1_block_count,
                                 const std::vector<int>& weight_shape,
                                 const Tensor* bias = nullptr,
                                 int device_id = 0);

CudaTensor CudaQ41LinearDeviceInput(const CudaTensor& x,
                                    const void* q4_1_device,
                                    size_t q4_1_block_count,
                                    const std::vector<int>& weight_shape,
                                    int device_id = 0);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_CUDA_QUANT_H_
