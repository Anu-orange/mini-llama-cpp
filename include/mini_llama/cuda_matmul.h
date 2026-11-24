// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_CUDA_MATMUL_H_
#define INCLUDE_MINI_LLAMA_CUDA_MATMUL_H_

#include <vector>

#include "mini_llama/cuda_tensor.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

bool CudaMatmulBuilt();

// F32 matrix multiplication on CUDA using cuBLAS.
// a: [m, k], b: [k, n], result: [m, n]
/**
 * @brief GPU上的矩阵乘法，向量点积计算，使用了cublas库
 * 
 * @param a 张量1
 * @param b 张量2
 * @param device_id 设备id
 * @return Tensor 
 */
Tensor CudaMatmul(const Tensor& a, const Tensor& b, int device_id = 0);

// F32 Linear projection on CUDA using cuBLAS.
// x: [in_features] or [batch, in_features]
// weight: [out_features, in_features]
// bias: optional [out_features], applied after the cuBLAS Matmul
// result: [out_features] for 1D input, [batch, out_features] for 2D input

/**
 * @brief 单精度浮点型线性投影的接口
 * 
 * @param x 输入的张量数据
 * @param weight 权重
 * @param bias 偏置
 * @param device_id 设备id 
 * @return Tensor 
 */
Tensor CudaLinear(const Tensor& x, const Tensor& weight,
                  const Tensor* bias = nullptr, int device_id = 0);

// F32 Linear projection using a weight matrix that already lives in CUDA device
// memory. The input activation is copied host->device, and the result is copied
// device->host.
/**
 * @brief Host输入混合device的权重
 * 
 * @param x host上输入的张量
 * @param w_device device上的权重矩阵指针
 * @param w_shape 权重维度
 * @param bias 偏置
 * @param device_id 设备id
 * @return Tensor 
 */
Tensor CudaLinearDeviceWeight(const Tensor& x, const void* w_device,
                              const std::vector<int>& w_shape,
                              const Tensor* bias = nullptr, int device_id = 0);

/**
 * @brief CUDA显存内部的的数据流动
 * 
 * @param x 显存上的张量
 * @param w_device 权重指针
 * @param w_shape 维度信息
 * @param device_id 
 * @return CudaTensor 
 */
CudaTensor CudaLinearDeviceInput(const CudaTensor& x, const void* w_device,
                                 const std::vector<int>& w_shape,
                                 int device_id = 0);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_CUDA_MATMUL_H_
