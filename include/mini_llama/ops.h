// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_OPS_H_
#define INCLUDE_MINI_LLAMA_OPS_H_

#include <cmath>

#include "mini_llama/model.h"
#include "mini_llama/quantized_tensor.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

// Matrix multiplication: c = a * b
// a: [m, k], b: [k, n], c: [m, n]
Tensor Matmul(const Tensor& a, const Tensor& b);

// Linear projection: y = x * weight^T
// x: [in_features] or [1, in_features]
// weight: [out_features, in_features]
// result keeps x rank: [out_features] or [1, out_features]
Tensor Linear(const Tensor& x, const Tensor& weight);

// Linear projection with quantized weight.
// Dispatches to F32, Q8_0, or Q4_0 path based on weight.type.
Tensor Linear(const Tensor& x, const QuantizedTensor& weight);

// RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
// x: [dim], weight: [dim], result: [dim]
/**
 * @brief 均方根归一化算子
 * 计算步骤： 1、先计算每个向量元素的平方
 *			2、然后计算上面平方均值(使用平方和除以元素个数)
 * 			3、然后将第二步的值加一个极小常数(比如1e-5)防止除零
 *			4、然后得到归一化向量：使用每个张量元素除以sqrt(第三步的值)
 *			5、最后再使用第四步的向量乘以一个权重向量
 * 
 * @param x 输入张量
 * @param weight 
 * @param eps 
 * @return Tensor 
 */
Tensor RmsNorm(const Tensor& x, const Tensor& weight, float eps);

// Softmax over a 1D tensor
// x: [n], result: [n]
/**
 * @brief 将注意力分数转换为总和为1的概率分布
 * 
 * @param x 输入的注意力分数张量
 * @return Tensor 
 */
Tensor Softmax(const Tensor& x);

// SiLU activation: Silu(x) = x * sigmoid(x)
/**
 * @brief Silu = x * sigmod(x)
 * 		其中sigmod(x) = 1 / (1 + e^-x)
 * @param x 输入张量
 * @return Tensor 
 */
Tensor Silu(const Tensor& x);

// Element-wise multiplication
/**
 * @brief 逐元素乘法算子
 * 
 * @param a 
 * @param b 
 * @return Tensor 
 */
Tensor ElementwiseMul(const Tensor& a, const Tensor& b);

// SwiGLU: SwiGlu(gate, up) = Silu(gate) * up
/**
 * @brief SwiGlu = sigmod(x) 逐元素乘以 up
 * 
 * @param gate 由x * Wgate得来
 * @param up 由x * Wup得来
 * @return Tensor 
 */
Tensor SwiGlu(const Tensor& gate, const Tensor& up);

// RoPE (Rotary Position Embedding) applied to Q and k
// q: [n_heads, head_dim], k: [n_kv_heads, head_dim]
// pos: current token position
/**
 * @brief Rope旋转位置编码实现
 * 
 * @param q q向量
 * @param k 
 * @param pos 
 * @param theta 
 * @param rope_type 
 */
void Rope(Tensor& q, Tensor& k, int pos, float theta,
          RopeType rope_type = RopeType::kNormal);

// Argmax: return FlatIndex of maximum value
/**
 * @brief 计算Tensor张量中最大的元素值
 * 
 * @param x 
 * @return int 
 */
int ArgMax(const Tensor& x);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_OPS_H_
