// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

// Primary header for the optimized dispatch implementation.
// clang-format off
#include "mini_llama/matmul_dispatch.h"
// clang-format on

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "mini_llama/thread_pool.h"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define MINI_LLAMA_USE_NEON 1
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define MINI_LLAMA_USE_AVX2 1
#endif

namespace mini_llama {

// ---------------------------------------------------------------------------
// Naive implementations (kept for readability and fallback)
// ---------------------------------------------------------------------------

static Tensor MatmulNaive(const Tensor& a, const Tensor& b) {
	int m = a.shape[0];
	int inner_dim = a.shape[1];
	int n = b.shape[1];
	Tensor c({m, n}, 0.0f);
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			float sum = 0.0f;
			for (int k = 0; k < inner_dim; ++k) {
				sum += a.data[i * inner_dim + k] * b.data[k * n + j];
			}
			c.data[i * n + j] = sum;
		}
	}
	return c;
}

static Tensor LinearNaive(const Tensor& x, const Tensor& weight) {
	int in_features = weight.shape[1];
	int out_features = weight.shape[0];
	bool is_1d = (x.num_dims() == 1);

	Tensor result(is_1d ? std::vector<int>{out_features}
	                    : std::vector<int>{1, out_features},
	              0.0f);
	for (int j = 0; j < out_features; ++j) {
		float sum = 0.0f;
		for (int k = 0; k < in_features; ++k) {
			sum += x.data[k] * weight.data[j * in_features + k];
		}
		result.data[j] = sum;
	}
	return result;
}

// ---------------------------------------------------------------------------
// Threaded implementations
// ---------------------------------------------------------------------------

/**
 * @brief 多个线程分组实现张量乘法(普通的矩阵乘法)
 * 
 * @param a 
 * @param b 
 * @return Tensor 
 */
static Tensor MatmulThreaded(const Tensor& a, const Tensor& b) {
	// 分别取出a和b的维度，其中b.shape[0]=a.shape[1]
	int m = a.shape[0];
	int inner_dim = a.shape[1];
	int n = b.shape[1];
	Tensor c({m, n}, 0.0f);
	// 按照m进行分组，在parallerfor中每个线程将至少承担：m / n_threads个任务
	ParallelFor(m, [&](int begin, int end) {
		for (int i = begin; i < end; ++i) {
			for (int j = 0; j < n; ++j) {
				float sum = 0.0f;
				for (int k = 0; k < inner_dim; ++k) {
					// 这里的求和要从两个1维数组的视角来进行看待
					sum += a.data[i * inner_dim + k] * b.data[k * n + j];
				}
				// 上面两个数组相乘后得到[m,n]
				// 从1维视角来看就是连续m个n个张量元素的张量
				// 画个图就知道这个i*n+j是怎么来的了
				c.data[i * n + j] = sum;
			}
		}
	});
	return c;
}

/**
 * @brief 输入向量和权重矩阵的乘法(右侧权重矩阵一次参与运算的是一行)
 * 
 * @param x 
 * @param weight 
 * @return Tensor 
 */
static Tensor LinearThreaded(const Tensor& x, const Tensor& weight) {
	int in_features = weight.shape[1];
	int out_features = weight.shape[0];
	bool is_1d = (x.num_dims() == 1);

	Tensor result(is_1d ? std::vector<int>{out_features}
	                    : std::vector<int>{1, out_features},
	              0.0f);
	// 按照out_features进行分块
	// 当前这段代码未加锁是因为每个线程写入的位置不一样，每个线程负责一段[begin, end)，每段[begin, end)
	// 这个区间是不会重叠的
	ParallelFor(out_features, [&](int begin, int end) {
		for (int j = begin; j < end; ++j) {
			float sum = 0.0f;
			for (int k = 0; k < in_features; ++k) {
				sum += x.data[k] * weight.data[j * in_features + k];
			}
			result.data[j] = sum;
		}
	});
	return result;
}

// ---------------------------------------------------------------------------
// SIMD dot-product helpers (platform-specific)
// ---------------------------------------------------------------------------

/**
 * @brief NEON版本的SIMD优化加速，NEON版本一次处理4对浮点数
 * 
 */
#ifdef MINI_LLAMA_USE_NEON
static float DotSimdNeon(const float* a, const float* b, int n) {

	// sum表示一个包含4个32位浮点分量的向量
	float32x4_t sum = vdupq_n_f32(0.0f);
	int i = 0;
	for (; i + 4 <= n; i += 4) {
		// 每一次循环从两个数组中各取4个浮点数出来，存到va和vb中
		float32x4_t va = vld1q_f32(a + i);
		float32x4_t vb = vld1q_f32(b + i);
		// 计算四对浮点数的点积，存到sum中并累加，且sum不是一个累加和
		// 而是包含4个分量的向量，每个分量才是两两的点积累加和
		sum = vfmaq_f32(sum, va, vb);
	}

	// 取出这个向量中的低半部分和高半部分，也就是将四个分量分成两组
	// 然后再计算两个向量对应分量的和，得到一个两个分量的向量
	// vadd_f32的两个参数是含有两个分量的向量
	float32x2_t r = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));

	// 再计算每个输入向量相邻两个分量相加
	// vpadd_f32的两个参数都是含有两个分量的向量
	r = vpadd_f32(r, r);
	// 然后取第0个分量，其实也就是r这个向量相邻分量的和
	float result = vget_lane_f32(r, 0);

	// 当i+4大于n，但是不+4又不满足n时
	// 再使用普通循环补齐剩余的累加和
	// 这个尾部收尾循环遍历很关键，当向量维度不是4的倍数时就用上了
	for (; i < n; ++i) {
		result += a[i] * b[i];
	}
	return result;
}
#endif

/**
 * @brief 该函数的做法和NEON很相似，只不过AVX2一条指令可以操作8对float浮点数
 * 
 */
#ifdef MINI_LLAMA_USE_AVX2
static float DotSimdAvx2(const float* a, const float* b, int n) {
	__m256 sum = _mm256_setzero_ps();

	// 1、批量加载，然后进行批量乘机加求和
	int i = 0;
	for (; i + 8 <= n; i += 8) {
		__m256 va = _mm256_loadu_ps(a + i);
		__m256 vb = _mm256_loadu_ps(b + i);
		sum = _mm256_fmadd_ps(va, vb, sum);
	}
	// Horizontal reduce
	// 2、分别取高低各一半
	__m128 sum_lo = _mm256_castps256_ps128(sum);
	__m128 sum_hi = _mm256_extractf128_ps(sum, 1);

	// 3、然后两个向量逐分量相加得到一个4个分量的向量
	sum_lo = _mm_add_ps(sum_lo, sum_hi);
	// 4、对两个相同的向量水平相加，得到一个四个分量的向量
	sum_lo = _mm_hadd_ps(sum_lo, sum_lo);
	// 5、再把这个四个分量的向量按照上面的算法再计算一次
	sum_lo = _mm_hadd_ps(sum_lo, sum_lo);
	// 最后取第0维的标量
	float result = _mm_cvtss_f32(sum_lo);
	for (; i < n; ++i) {
		result += a[i] * b[i];
	}
	return result;
}
#endif

/**
 * @brief n对浮点数做乘积累加和计算
 * 
 * @param a 
 * @param b 
 * @param n 
 * @return float 
 */
static float DotSimd(const float* a, const float* b, int n) {
#if defined(MINI_LLAMA_USE_AVX2)
	return DotSimdAvx2(a, b, n);
#elif defined(MINI_LLAMA_USE_NEON)
	return DotSimdNeon(a, b, n);
#else
	float sum = 0.0f;
	// 普通的标量循环
	for (int i = 0; i < n; ++i) {
		sum += a[i] * b[i];
	}
	return sum;
#endif
}

// ---------------------------------------------------------------------------
// SIMD implementations
// ---------------------------------------------------------------------------

/**
 * @brief 使用单指令多数据技术来计算线形层的结果(一个线程来做所有out_features行的计算)
 * 
 * @param x 
 * @param weight 
 * @return Tensor 
 */
static Tensor LinearSimd(const Tensor& x, const Tensor& weight) {
	int in_features = weight.shape[1];
	int out_features = weight.shape[0];
	bool is_1d = (x.num_dims() == 1);

	Tensor result(is_1d ? std::vector<int>{out_features}
	                    : std::vector<int>{1, out_features},
	              0.0f);
	for (int j = 0; j < out_features; ++j) {
		result.data[j] = DotSimd(
		    x.data.data(), weight.data.data() + j * in_features, in_features);
	}
	return result;
}

/**
 * @brief 在linearsimd的基础上进一步使用多线程的方式，将out_features划分到多个线程上
 * 然后进行分块计算
 * 
 * @param x 
 * @param weight 
 * @return Tensor 
 */
static Tensor LinearThreadedSimd(const Tensor& x, const Tensor& weight) {
	int in_features = weight.shape[1];
	int out_features = weight.shape[0];
	bool is_1d = (x.num_dims() == 1);

	Tensor result(is_1d ? std::vector<int>{out_features}
	                    : std::vector<int>{1, out_features},
	              0.0f);
	// 多个线程分块同时计算
	ParallelFor(out_features, [&](int begin, int end) {
		for (int j = begin; j < end; ++j) {
			result.data[j] =
			    DotSimd(x.data.data(), weight.data.data() + j * in_features,
			            in_features);
		}
	});
	return result;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

MatmulMode DefaultMatmulMode() { return MatmulMode::kThreadedSimd; }

/**
 * @brief 张量乘法运算的分发
 *
 * @param a
 * @param b
 * @param mode
 * @return Tensor
 */
Tensor MatmulDispatch(const Tensor& a, const Tensor& b, MatmulMode mode) {
	if (a.num_dims() != 2 || b.num_dims() != 2) {
		throw std::runtime_error("MatmulDispatch: expected 2D tensors");
	}
	if (a.shape[1] != b.shape[0]) {
		throw std::runtime_error("MatmulDispatch: dimension mismatch");
	}
	switch (mode) {
		case MatmulMode::kNaive:
			return MatmulNaive(a, b);
		case MatmulMode::kThreaded:
			return MatmulThreaded(a, b);
		case MatmulMode::kSimd:
		case MatmulMode::kThreadedSimd:
			// For general Matmul, b column access is not contiguous,
			// so SIMD is complex. Fall back to threaded for now.
			return MatmulThreaded(a, b);
	}
	return MatmulThreaded(a, b);
}

Tensor LinearDispatch(const Tensor& x, const Tensor& weight, MatmulMode mode) {
	if (weight.num_dims() != 2) {
		throw std::runtime_error(
		    "LinearDispatch: expected weight shape [out_features, "
		    "in_features]");
	}
	int in_features;
	if (x.num_dims() == 1) {
		in_features = x.shape[0];
	} else if (x.num_dims() == 2 && x.shape[0] == 1) {
		in_features = x.shape[1];
	} else {
		throw std::runtime_error(
		    "LinearDispatch: expected x shape [in_features] or [1, "
		    "in_features]");
	}
	if (weight.shape[1] != in_features) {
		throw std::runtime_error("LinearDispatch: dimension mismatch");
	}
	switch (mode) {
		case MatmulMode::kNaive:
			return LinearNaive(x, weight);
		case MatmulMode::kThreaded:
			return LinearThreaded(x, weight);
		case MatmulMode::kSimd:
			return LinearSimd(x, weight);
		case MatmulMode::kThreadedSimd:
			return LinearThreadedSimd(x, weight);
	}
	return LinearThreadedSimd(x, weight);
}

}  // namespace mini_llama
