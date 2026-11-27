// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <limits>

#include "mini_llama/ops.h"
#include "mini_llama/quant.h"
#include "mini_llama/tensor.h"
#include "tests/test_main.h"
#include "tests/test_names.h"

static bool TestQ80BlockLayout() {
	MINI_LLAMA_ASSERT_EQ(kQ80BlockSize, 32);
	MINI_LLAMA_ASSERT_EQ(sizeof(BlockQ80), 34);
	return true;
}

// ---------------------------------------------------------------------------
// Q8_0 quantization / dequantization
// ---------------------------------------------------------------------------
static bool TestQ80RoundtripIdentity() {
	Tensor src({32}, 0.0f);
	for (int i = 0; i < 32; ++i) {
		src.data[i] = static_cast<float>(i) * 0.1f;
	}

	auto blocks = QuantizeToQ80(src);
	MINI_LLAMA_ASSERT_EQ(blocks.size(), 1);

	Tensor dst = DequantizeFromQ80(blocks, src.shape);
	MINI_LLAMA_ASSERT_EQ(dst.shape.size(), 1);
	MINI_LLAMA_ASSERT_EQ(dst.shape[0], 32);

	// Should be close but not exact due to quantization.
	// Q8_0 roundtrip error can be up to ~0.5 * scale per element.
	float max_err = 0.0f;
	for (size_t i = 0; i < src.size(); ++i) {
		float err = std::abs(src.data[i] - dst.data[i]);
		if (err > max_err) {
			max_err = err;
		}
	}
	MINI_LLAMA_ASSERT_TRUE(max_err < 6e-2f);
	return true;
}

static bool TestQ80AllZeros() {
	Tensor src({64}, 0.0f);
	auto blocks = QuantizeToQ80(src);
	MINI_LLAMA_ASSERT_EQ(blocks.size(), 2);

	Tensor dst = DequantizeFromQ80(blocks, src.shape);
	for (size_t i = 0; i < dst.size(); ++i) {
		MINI_LLAMA_ASSERT_NEAR(dst.data[i], 0.0f, 1e-6f);
	}
	return true;
}

static bool TestQ80MultiBlock() {
	Tensor src({100}, 0.0f);
	for (int i = 0; i < 100; ++i) {
		src.data[i] = static_cast<float>(i) * 0.05f;
	}

	auto blocks = QuantizeToQ80(src);
	MINI_LLAMA_ASSERT_EQ(blocks.size(), 4);  // ceil(100 / 32) = 4

	Tensor dst = DequantizeFromQ80(blocks, src.shape);
	MINI_LLAMA_ASSERT_EQ(dst.shape[0], 100);

	float max_err = 0.0f;
	for (size_t i = 0; i < src.size(); ++i) {
		float err = std::abs(src.data[i] - dst.data[i]);
		if (err > max_err) {
			max_err = err;
		}
	}
	MINI_LLAMA_ASSERT_TRUE(max_err < 6e-2f);
	return true;
}

static bool TestQ80EmptyTensor() {
	try {
		Tensor src({0}, 0.0f);
		(void)src;
		MINI_LLAMA_ASSERT_FAIL("expected exception for zero-dimension tensor");
	} catch (const std::runtime_error&) {
		// expected: Tensor rejects zero dimension
	}
	return true;
}

// ---------------------------------------------------------------------------
// Q8_0 Matmul
// ---------------------------------------------------------------------------
static bool TestMatmulQ80MatchesF32() {
	// Small exact test
	Tensor a({2, 3}, 0.0f);
	a.data[0] = 1.0f;
	a.data[1] = 2.0f;
	a.data[2] = 3.0f;
	a.data[3] = 4.0f;
	a.data[4] = 5.0f;
	a.data[5] = 6.0f;

	Tensor b({3, 2}, 0.0f);
	b.data[0] = 1.0f;
	b.data[1] = 2.0f;
	b.data[2] = 3.0f;
	b.data[3] = 4.0f;
	b.data[4] = 5.0f;
	b.data[5] = 6.0f;

	auto qA = QuantizeToQ80(a);
	Tensor c_f32 = Matmul(a, b);
	Tensor c_q8 = MatmulQ80(qA, b, a.shape);

	MINI_LLAMA_ASSERT_EQ(c_f32.shape.size(), 2);
	MINI_LLAMA_ASSERT_EQ(c_q8.shape.size(), 2);
	MINI_LLAMA_ASSERT_EQ(c_f32.shape[0], c_q8.shape[0]);
	MINI_LLAMA_ASSERT_EQ(c_f32.shape[1], c_q8.shape[1]);

	for (size_t i = 0; i < c_f32.size(); ++i) {
		// Q8_0 Matmul accumulates per-element quantization error;
		// tolerance must be looser than raw roundtrip.
		MINI_LLAMA_ASSERT_NEAR(c_f32.data[i], c_q8.data[i], 2e-1f);
	}
	return true;
}

static bool TestCompareMatmulError() {
	Tensor a({16, 16}, 0.0f);
	Tensor b({16, 16}, 0.0f);
	for (size_t i = 0; i < a.size(); ++i) {
		a.data[i] = static_cast<float>(i) * 0.01f - 1.0f;
	}
	for (size_t i = 0; i < b.size(); ++i) {
		b.data[i] = static_cast<float>(i) * 0.01f - 0.5f;
	}

	float err = CompareMatmulError(a, b);
	MINI_LLAMA_ASSERT_TRUE(err >= 0.0f);
	MINI_LLAMA_ASSERT_TRUE(err < 1e-1f);  // Q8_0 error should be small
	return true;
}

static bool TestDequantizeRejectsBadShape() {
	Tensor src({32}, 1.0f);
	auto blocks = QuantizeToQ80(src);

	try {
		(void)DequantizeFromQ80(blocks, {-1});
		MINI_LLAMA_ASSERT_FAIL("expected exception for negative shape");
	} catch (const std::runtime_error&) {
		// expected
	}

	try {
		(void)DequantizeFromQ80(blocks, {0});
		MINI_LLAMA_ASSERT_FAIL("expected exception for zero shape");
	} catch (const std::runtime_error&) {
		// expected
	}

	return true;
}

static bool TestDequantizeRejectsBlockCountMismatch() {
	Tensor src({64}, 1.0f);
	auto blocks = QuantizeToQ80(src);
	blocks.pop_back();

	try {
		(void)DequantizeFromQ80(blocks, src.shape);
		MINI_LLAMA_ASSERT_FAIL("expected exception for block count mismatch");
	} catch (const std::runtime_error&) {
		// expected
	}
	return true;
}

// ---------------------------------------------------------------------------
// Q4_0 quantization / dequantization
// ---------------------------------------------------------------------------
static bool TestQ40BlockLayout() {
	MINI_LLAMA_ASSERT_EQ(kQ40BlockSize, 32);
	MINI_LLAMA_ASSERT_EQ(sizeof(BlockQ40), 18);
	return true;
}

static bool TestQ41BlockLayout() {
	MINI_LLAMA_ASSERT_EQ(kQ41BlockSize, 32);
	MINI_LLAMA_ASSERT_EQ(sizeof(BlockQ41), 20);
	return true;
}

static bool TestQ40RoundtripIdentity() {
	Tensor src({32}, 0.0f);
	for (int i = 0; i < 32; ++i) {
		src.data[i] = static_cast<float>(i) * 0.1f;
	}

	auto blocks = QuantizeToQ40(src);
	MINI_LLAMA_ASSERT_EQ(blocks.size(), 1);

	Tensor dst = DequantizeFromQ40(blocks, src.shape);
	MINI_LLAMA_ASSERT_EQ(dst.shape[0], 32);

	float max_err = 0.0f;
	for (size_t i = 0; i < src.size(); ++i) {
		float err = std::abs(src.data[i] - dst.data[i]);
		if (err > max_err) {
			max_err = err;
		}
	}
	MINI_LLAMA_ASSERT_TRUE(max_err < 3e-1f);
	return true;
}

static bool TestQ40AllZeros() {
	Tensor src({64}, 0.0f);
	auto blocks = QuantizeToQ40(src);
	MINI_LLAMA_ASSERT_EQ(blocks.size(), 2);

	Tensor dst = DequantizeFromQ40(blocks, src.shape);
	for (size_t i = 0; i < dst.size(); ++i) {
		MINI_LLAMA_ASSERT_NEAR(dst.data[i], 0.0f, 1e-6f);
	}
	return true;
}

static bool TestQ41DequantizeKnownValues() {
	BlockQ41 block{};
	block.d = 0x3800;  // fp16(0.5)
	block.m = 0xbc00;  // fp16(-1.0)
	for (int i = 0; i < 16; ++i) {
		block.qs[i] = static_cast<uint8_t>(i | ((15 - i) << 4));
	}

	Tensor dst = DequantizeFromQ41({block}, {32});
	MINI_LLAMA_ASSERT_EQ(dst.shape[0], 32);
	for (int i = 0; i < 16; ++i) {
		MINI_LLAMA_ASSERT_NEAR(dst.data[i],
		                       -1.0f + 0.5f * static_cast<float>(i), 1e-6f);
		MINI_LLAMA_ASSERT_NEAR(
		    dst.data[i + 16], -1.0f + 0.5f * static_cast<float>(15 - i), 1e-6f);
	}
	return true;
}

static bool TestQ41LinearKnownValues() {
	BlockQ41 block{};
	block.d = 0x3c00;  // fp16(1.0)
	block.m = 0x0000;  // fp16(0.0)
	for (int i = 0; i < 16; ++i) {
		block.qs[i] = 0x11;  // q0=1, q1=1
	}

	Tensor x({32}, 2.0f);
	Tensor y = LinearQ41(x, {block}, {1, 32});
	MINI_LLAMA_ASSERT_EQ(y.shape[0], 1);
	MINI_LLAMA_ASSERT_NEAR(y.data[0], 64.0f, 1e-6f);
	return true;
}

static bool TestQ40LinearMatchesF32() {
	// Use in_features divisible by kQ40BlockSize (32)
	Tensor weight({4, 32}, 0.0f);
	for (size_t i = 0; i < weight.size(); ++i) {
		weight.data[i] = static_cast<float>(i) * 0.05f - 1.0f;
	}

	Tensor x({32}, 0.0f);
	for (size_t i = 0; i < x.size(); ++i) {
		x.data[i] = static_cast<float>(i) * 0.1f - 0.4f;
	}

	auto qW = QuantizeToQ40(weight);
	Tensor y_f32 = Linear(x, weight);
	Tensor y_q4 = LinearQ40(x, qW, weight.shape);

	MINI_LLAMA_ASSERT_EQ(y_f32.shape, y_q4.shape);
	for (size_t i = 0; i < y_f32.size(); ++i) {
		MINI_LLAMA_ASSERT_NEAR(y_f32.data[i], y_q4.data[i], 2.0f);
	}
	return true;
}

static bool TestCompareQ40Error() {
	// Use dimensions divisible by kQ40BlockSize (32)
	Tensor weight({8, 32}, 0.0f);
	Tensor x({32}, 0.0f);
	for (size_t i = 0; i < weight.size(); ++i) {
		weight.data[i] = static_cast<float>(i) * 0.01f - 1.0f;
	}
	for (size_t i = 0; i < x.size(); ++i) {
		x.data[i] = static_cast<float>(i) * 0.01f - 0.5f;
	}

	float err = CompareQ40Error(weight, x);
	MINI_LLAMA_ASSERT_TRUE(err >= 0.0f);
	MINI_LLAMA_ASSERT_TRUE(err < 2.0f);
	return true;
}

// ---------------------------------------------------------------------------
// Auto-register
// ---------------------------------------------------------------------------
static bool TestQ41PackedRoundtrip() {
	Tensor src({32}, 0.0f);
	for (int i = 0; i < 32; ++i) {
		src.data[i] = static_cast<float>(i < 16 ? i : 31 - i) - 4.0f;
	}
	auto blocks = QuantizeToQ41(src);
	MINI_LLAMA_ASSERT_EQ(blocks.size(), 1);
	MINI_LLAMA_ASSERT_EQ(blocks[0].d, 0x3c00);  // FP16 1.0
	MINI_LLAMA_ASSERT_EQ(blocks[0].m, 0xc400);  // FP16 -4.0
	for (int j = 0; j < 16; ++j) {
		MINI_LLAMA_ASSERT_EQ(blocks[0].qs[j], j | ((15 - j) << 4));
	}
	Tensor dst = DequantizeFromQ41(blocks, src.shape);
	MINI_LLAMA_ASSERT_EQ(dst.shape, src.shape);
	for (size_t i = 0; i < src.size(); ++i) {
		MINI_LLAMA_ASSERT_NEAR(dst.data[i], src.data[i], 0.0f);
	}

	return true;
}

static bool TestQ41ConstantsAndZeros() {
	for (float value : {0.0f, 3.5f, -2.25f, 1.2345f}) {
		Tensor src({64}, value);
		auto blocks = QuantizeToQ41(src);
		MINI_LLAMA_ASSERT_EQ(blocks.size(), 2);
		for (const auto& block : blocks) {
			MINI_LLAMA_ASSERT_EQ(block.d, 0);
			for (uint8_t q : block.qs) {
				MINI_LLAMA_ASSERT_EQ(q, 0);
			}
		}
		Tensor dst = DequantizeFromQ41(blocks, src.shape);
		for (float x : dst.data) {
			MINI_LLAMA_ASSERT_TRUE(std::isfinite(x));
			MINI_LLAMA_ASSERT_NEAR(x, value, 1e-3f);
		}
	}

	return true;
}

static bool TestQ41PartialRows() {
	// Two blocks per row; each row has its own range and a 3-element tail.
	Tensor src({3, 35}, 0.0f);
	for (int row = 0; row < 3; ++row) {
		for (int k = 0; k < 35; ++k) {
			src.data[row * 35 + k] = (row - 1) * 20.0f + k * 0.125f;
		}
	}
	auto blocks = QuantizeToQ41(src);
	MINI_LLAMA_ASSERT_EQ(blocks.size(), 6);
	Tensor dst = DequantizeFromQ41(blocks, src.shape);
	MINI_LLAMA_ASSERT_EQ(dst.shape, src.shape);
	// Full-block step is 3.875/15; allow half a step plus FP16 rounding.
	for (size_t i = 0; i < src.size(); ++i) {
		MINI_LLAMA_ASSERT_TRUE(std::isfinite(dst.data[i]));
		MINI_LLAMA_ASSERT_NEAR(dst.data[i], src.data[i], 0.14f);
	}
	for (int row = 0; row < 3; ++row) {
		const auto& tail = blocks[row * 2 + 1];
		// Padding must not consume elements from the following row.
		for (int j = 0; j < 16; ++j) {
			MINI_LLAMA_ASSERT_EQ(tail.qs[j] >> 4, 0);
			if (j >= 3) {
				MINI_LLAMA_ASSERT_EQ(tail.qs[j] & 15, 0);
			}
		}
	}

	return true;
}

static bool TestQ41EmptyAndTinyRange() {
	MINI_LLAMA_ASSERT_TRUE(QuantizeToQ41(Tensor{}).empty());
	Tensor src({32}, 0.0f);
	for (int i = 0; i < 32; ++i) {
		src.data[i] = (i % 2) * 1e-8f;
	}
	auto blocks = QuantizeToQ41(src);
	MINI_LLAMA_ASSERT_EQ(blocks[0].d, 0);  // Scale underflows to FP16 zero.
	Tensor dst = DequantizeFromQ41(blocks, src.shape);
	for (float x : dst.data) {
		MINI_LLAMA_ASSERT_TRUE(std::isfinite(x));
		MINI_LLAMA_ASSERT_NEAR(x, 0.0f, 1e-8f);
	}

	return true;
}

static bool TestQ41RejectsUnrepresentableValues() {
	for (float value : {std::numeric_limits<float>::quiet_NaN(),
	                    std::numeric_limits<float>::infinity(),
	                    -std::numeric_limits<float>::infinity(), 70000.0f}) {
		Tensor src({32}, value);
		bool threw = false;
		try {
			(void)QuantizeToQ41(src);
		} catch (const std::runtime_error&) {
			threw = true;
		}
		MINI_LLAMA_ASSERT_TRUE(threw);
	}

	return true;
}

static bool TestQ41LinearMatchesF32() {
	// Exact integer weights make arithmetic errors distinguishable from loss.
	Tensor weight({3, 35}, 0.0f);
	for (int row = 0; row < 3; ++row) {
		for (int k = 0; k < 35; ++k) {
			weight.data[row * 35 + k] =
			    static_cast<float>(row * 2 + (k < 32 ? k % 16 : (k % 2) * 15));
		}
	}
	auto blocks = QuantizeToQ41(weight);
	for (const auto& shape : {std::vector<int>{35}, std::vector<int>{2, 35}}) {
		Tensor x(shape, 0.0f);
		for (size_t i = 0; i < x.size(); ++i) {
			x.data[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.25f;
		}
		// F32 LinearDispatch only accepts one row; use an independent
		// scalar reference so the quantized batch path is tested as well.
		int rows = x.num_dims() == 1 ? 1 : x.shape[0];
		Tensor expected(
		    x.num_dims() == 1 ? std::vector<int>{3} : std::vector<int>{rows, 3},
		    0.0f);
		for (int row = 0; row < rows; ++row) {
			for (int out = 0; out < 3; ++out) {
				for (int k = 0; k < 35; ++k) {
					expected.data[row * 3 + out] +=
					    x.data[row * 35 + k] * weight.data[out * 35 + k];
				}
			}
		}
		Tensor actual = LinearQ41(x, blocks, weight.shape);
		MINI_LLAMA_ASSERT_EQ(actual.shape, expected.shape);
		for (size_t i = 0; i < actual.size(); ++i) {
			MINI_LLAMA_ASSERT_TRUE(std::isfinite(actual.data[i]));
			MINI_LLAMA_ASSERT_NEAR(actual.data[i], expected.data[i], 0.02f);
		}
	}

	return true;
}

static bool TestCompareQ41Error() {
	Tensor weight({4, 32}, 0.0f);
	Tensor x({32}, 0.0f);
	for (size_t i = 0; i < weight.size(); ++i) {
		weight.data[i] = static_cast<float>(i) * 0.01f - 0.5f;
	}
	for (size_t i = 0; i < x.size(); ++i) {
		x.data[i] = static_cast<float>(i) * 0.01f - 0.2f;
	}
	float err = CompareQ41Error(weight, x);
	MINI_LLAMA_ASSERT_TRUE(std::isfinite(err));
	MINI_LLAMA_ASSERT_TRUE(err >= 0.0f);
	MINI_LLAMA_ASSERT_TRUE(err < 0.05f);

	return true;
}

static struct QuantTestRegistrar {
	QuantTestRegistrar() {
		RegisterTest("TestQ41PackedRoundtrip", TestQ41PackedRoundtrip);
		RegisterTest("TestQ41ConstantsAndZeros", TestQ41ConstantsAndZeros);
		RegisterTest("TestQ41PartialRows", TestQ41PartialRows);
		RegisterTest("TestQ41EmptyAndTinyRange", TestQ41EmptyAndTinyRange);
		RegisterTest("TestQ41RejectsUnrepresentableValues",
		             TestQ41RejectsUnrepresentableValues);
		RegisterTest("TestQ41LinearMatchesF32", TestQ41LinearMatchesF32);
		RegisterTest("TestCompareQ41Error", TestCompareQ41Error);
		RegisterTest("q8_0_block_layout", TestQ80BlockLayout);
		RegisterTest("q8_0_roundtrip_identity", TestQ80RoundtripIdentity);
		RegisterTest("q8_0_all_zeros", TestQ80AllZeros);
		RegisterTest("q8_0_multi_block", TestQ80MultiBlock);
		RegisterTest("q8_0_empty_tensor", TestQ80EmptyTensor);
		RegisterTest("matmul_q8_0_matches_f32", TestMatmulQ80MatchesF32);
		RegisterTest("CompareMatmulError", TestCompareMatmulError);
		RegisterTest("dequantize_rejects_bad_shape",
		             TestDequantizeRejectsBadShape);
		RegisterTest("dequantize_rejects_block_count_mismatch",
		             TestDequantizeRejectsBlockCountMismatch);
		RegisterTest("q4_0_block_layout", TestQ40BlockLayout);
		RegisterTest("q4_1_block_layout", TestQ41BlockLayout);
		RegisterTest("q4_0_roundtrip_identity", TestQ40RoundtripIdentity);
		RegisterTest("q4_0_all_zeros", TestQ40AllZeros);
		RegisterTest("q4_1_dequantize_known_values",
		             TestQ41DequantizeKnownValues);
		RegisterTest("q4_1_linear_known_values", TestQ41LinearKnownValues);
		RegisterTest("q4_0_linear_matches_f32", TestQ40LinearMatchesF32);
		RegisterTest("CompareQ40Error", TestCompareQ40Error);
	}
} quant_test_registrar;
