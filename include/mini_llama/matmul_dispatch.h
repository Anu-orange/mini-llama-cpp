// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_MATMUL_DISPATCH_H_
#define INCLUDE_MINI_LLAMA_MATMUL_DISPATCH_H_

#include "mini_llama/tensor.h"

namespace mini_llama {

enum class MatmulMode {
	kNaive,  // 朴素计算模式，通过普通循环完成矩阵乘法计算
	kThreaded,  // 多线程计算模式，将输出任务划分给多个线程，每个线程内部使用普通循环
	kSimd,  // SIMD向量计算，单线程、一条向量指令同时处理多个元素
	kThreadedSimd,  // 多线程+SIMD，多个线程分担，每个线程内部使用SIMD
};

// Default mode used by the main inference path.
MatmulMode DefaultMatmulMode();

// F32 matrix multiplication [m,k] x [k,n] -> [m,n]
Tensor MatmulDispatch(const Tensor& a, const Tensor& b, MatmulMode mode);

// F32 Linear: x @ weight^T
// Supports x: [in_features] or [1, in_features]
// weight: [out_features, in_features]
// result rank matches input rank.
Tensor LinearDispatch(const Tensor& x, const Tensor& weight, MatmulMode mode);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_MATMUL_DISPATCH_H_
