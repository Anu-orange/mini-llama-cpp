// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/quant.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/ops.h"
#include "mini_llama/thread_pool.h"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define MINI_LLAMA_USE_NEON 1
#endif

namespace mini_llama {

namespace {

uint16_t FloatToFp16(float value) {
	uint32_t bits = 0;
	std::memcpy(&bits, &value, sizeof(bits));

	uint32_t sign = (bits >> 16) & 0x8000u;
	int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
	uint32_t mantissa = bits & 0x7fffffu;

	if (exponent <= 0) {
		if (exponent < -10) {
			return static_cast<uint16_t>(sign);
		}
		mantissa |= 0x800000u;
		uint32_t shifted = mantissa >> (1 - exponent);
		if ((shifted & 0x00001000u) != 0) {
			shifted += 0x00002000u;
		}
		return static_cast<uint16_t>(sign | (shifted >> 13));
	}

	if (exponent >= 31) {
		return static_cast<uint16_t>(sign | 0x7c00u);
	}

	if ((mantissa & 0x00001000u) != 0) {
		mantissa += 0x00002000u;
		if ((mantissa & 0x00800000u) != 0) {
			mantissa = 0;
			++exponent;
			if (exponent >= 31) {
				return static_cast<uint16_t>(sign | 0x7c00u);
			}
		}
	}

	return static_cast<uint16_t>(
	    sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
}

float Fp16ToFloat(uint16_t value) {
	uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
	uint32_t exponent = (value >> 10) & 0x1fu;
	uint32_t mantissa = value & 0x03ffu;
	uint32_t bits = 0;

	if (exponent == 0) {
		if (mantissa == 0) {
			bits = sign;
		} else {
			exponent = 1;
			while ((mantissa & 0x0400u) == 0) {
				mantissa <<= 1;
				--exponent;
			}
			mantissa &= 0x03ffu;
			uint32_t exp32 = exponent + (127 - 15);
			bits = sign | (exp32 << 23) | (mantissa << 13);
		}
	} else if (exponent == 31) {
		bits = sign | 0x7f800000u | (mantissa << 13);
	} else {
		uint32_t exp32 = exponent + (127 - 15);
		bits = sign | (exp32 << 23) | (mantissa << 13);
	}

	float out = 0.0f;
	std::memcpy(&out, &bits, sizeof(out));
	return out;
}

size_t CheckedNumel(const std::vector<int>& shape, const char* caller) {
	size_t total = 1;
	for (size_t axis = 0; axis < shape.size(); ++axis) {
		int dim = shape[axis];
		if (dim <= 0) {
			throw std::runtime_error(
			    std::string(caller) + ": dimension at axis " +
			    std::to_string(axis) + " must be positive, got " +
			    std::to_string(dim));
		}
		size_t dim_size = static_cast<size_t>(dim);
		if (total > std::numeric_limits<size_t>::max() / dim_size) {
			throw std::runtime_error(std::string(caller) +
			                         ": shape element count overflow");
		}
		total *= dim_size;
	}
	return total;
}

}  // namespace

// ---------------------------------------------------------------------------
// Q8_0 quantize / dequantize
// ---------------------------------------------------------------------------
/**
 * @brief q8_0量化过程
 * 
 * @param src 
 * @return std::vector<BlockQ80> 
 */
std::vector<BlockQ80> QuantizeToQ80(const Tensor& src) {
	if (src.size() == 0) {
		return {};
	}

	// 首先计算一行的元素
	int row_size =
	    src.num_dims() >= 2 ? src.shape.back() : static_cast<int>(src.size());
	// 然后计算总共有多少行
	int n_rows =
	    src.num_dims() >= 2 ? static_cast<int>(src.size()) / row_size : 1;
	
	// 计算一行数据按照32个浮点数为一块的话，看能分成多少块(这个公式的推导看gguf_loader.c:175附近的过程)
	int row_blocks = (row_size + kQ80BlockSize - 1) / kQ80BlockSize;
	
	// 然后计算总的块数
	size_t total_blocks = static_cast<size_t>(n_rows) * row_blocks;

	std::vector<BlockQ80> blocks;
	blocks.reserve(total_blocks);

	// 这个量化过程可以这样来看：最外层遍历有多少行，次外层遍历一行有多少块，最内层遍历一块要处理多少个浮点数
	// 使用for循环一行多块的进行量化
	for (int row = 0; row < n_rows; ++row) {
		// 先计算当前行的起始偏移
		int row_offset = row * row_size;
		for (int rb = 0; rb < row_blocks; ++rb) {
			// 一行内的元素偏移
			int base = row_offset + rb * kQ80BlockSize;
			int k_end = std::min(base + kQ80BlockSize, row_offset + row_size);
			// 计算一个块包含的浮点数
			int block_len = k_end - base;
			
			// 计算这一个块中所有元素的最大绝对值
			float max_abs = 0.0f;
			for (int k = base; k < k_end; ++k) {
				float abs_val = std::abs(src.data[k]);
				if (abs_val > max_abs) {
					max_abs = abs_val;
				}
			}

			// 定义一个q8_0的结构体并全部初始化为0
			BlockQ80 block;
			std::memset(&block, 0, sizeof(block));
			if (max_abs > 0.0f) {
				// 计算缩放因子
				float d = max_abs / 127.0f;
				// 将缩放因子转换为fp16存储
				block.d = FloatToFp16(d);
				float stored_d = Fp16ToFloat(block.d);
				// 计算缩放因子的倒数
				float id = stored_d == 0.0f ? 0.0f : 1.0f / stored_d;
				for (int i = 0; i < block_len; ++i) {
					// 计算原浮点数乘以id后的值
					float q = src.data[base + i] * id;
					// 然后计算量化后的值(进行四舍五入)
					int qi = static_cast<int>(std::round(q));
					if (qi > 127) {
						qi = 127;
					} else if (qi < -127) {
						qi = -127;
					}
					// 将计算后的值转换成int8_t后进行存储
					block.qs[i] = static_cast<int8_t>(qi);
				}
			} else {
				block.d = 0;
			}
			// Pad remainder with 0
			for (int i = block_len; i < kQ80BlockSize; ++i) {
				block.qs[i] = 0;
			}
			// 存到
			blocks.push_back(block);
		}
	}

	return blocks;
}

/**
 * @brief 反量化，将量化的数据恢复成F32
 * 
 * @param blocks 量化后的块数组
 * @param shape 张量维度
 * @return Tensor 
 */
Tensor DequantizeFromQ80(const std::vector<BlockQ80>& blocks,
                         const std::vector<int>& shape) {
	// 计算总元素个数
	size_t total = CheckedNumel(shape, "DequantizeFromQ80");
	
	// 每行的元素个数
	int row_size = shape.size() >= 2 ? shape.back() : static_cast<int>(total);
	// 行数
	int n_rows = shape.size() >= 2 ? static_cast<int>(total) / row_size : 1;
	// 每行的块数
	int row_blocks = (row_size + kQ80BlockSize - 1) / kQ80BlockSize;
	// 期待的总块数
	size_t expected_blocks = static_cast<size_t>(n_rows) * row_blocks;
	// 如果说期待的块数和实际传递的块数不一致就报错
	if (blocks.size() != expected_blocks) {
		throw std::runtime_error(
		    "DequantizeFromQ80: block count mismatch: expected " +
		    std::to_string(expected_blocks) + ", got " +
		    std::to_string(blocks.size()));
	}

	// 创建一个空白张量
	Tensor dst(shape, 0.0f);
	size_t block_idx = 0;
	// 遍历每一行
	for (int row = 0; row < n_rows; ++row) {
		// 计算每行的行偏移，行偏移其实就是索引乘以每行的元素个数，无论逻辑是几维的
		// 在内存中都是一维数组
		int row_offset = row * row_size;
		// 遍历每行的块数
		for (int rb = 0; rb < row_blocks; ++rb) {
			// 计算出当前这一行中某块的偏移
			// 行偏移起始加上块大小乘以块索引 
			int base = row_offset + rb * kQ80BlockSize;
			// 记录一行中块的结束下标
			int k_end = std::min(base + kQ80BlockSize, row_offset + row_size);
			// 取出量化后的一块数据
			const BlockQ80& block = blocks[block_idx++];
			// 把缩放因子转成float
			float d = Fp16ToFloat(block.d);
			// 然后使用k进行遍历
			for (int k = base; k < k_end; ++k) {
				// 使用缩放因子乘以float形式的量化值就等于反量化后的值
				dst.data[k] = d * static_cast<float>(block.qs[k - base]);
			}
		}
	}

	return dst;
}

// ---------------------------------------------------------------------------
// Q4_0 quantize / dequantize
// ---------------------------------------------------------------------------
/**
 * @brief q4_0向量化的实现过程
 * 
 * @param src 
 * @return std::vector<BlockQ40> 
 */
std::vector<BlockQ40> QuantizeToQ40(const Tensor& src) {
	if (src.size() == 0) {
		return {};
	}

	// 首先计算行元素个数以及行数、一行的块数
	int row_size =
	    src.num_dims() >= 2 ? src.shape.back() : static_cast<int>(src.size());
	int n_rows =
	    src.num_dims() >= 2 ? static_cast<int>(src.size()) / row_size : 1;
	int row_blocks = (row_size + kQ40BlockSize - 1) / kQ40BlockSize;
	size_t total_blocks = static_cast<size_t>(n_rows) * row_blocks;

	std::vector<BlockQ40> blocks;
	blocks.reserve(total_blocks);

	for (int row = 0; row < n_rows; ++row) {
		// 先遍历每一行，找到每行的起始偏移
		int row_offset = row * row_size;
		for (int rb = 0; rb < row_blocks; ++rb) {
			// 再遍历每行的块数，计算每块的起始位置
			int base = row_offset + rb * kQ40BlockSize;
			// 计算块的终止偏移
			int k_end = std::min(base + kQ40BlockSize, row_offset + row_size);
			// 计算每块的元素个数
			int block_len = k_end - base;

			// 计算多个元素中绝对值最大的那个
			float max_abs = 0.0f;
			for (int k = base; k < k_end; ++k) {
				float abs_val = std::abs(src.data[k]);
				if (abs_val > max_abs) {
					max_abs = abs_val;
				}
			}

			BlockQ40 block;
			std::memset(&block, 0, sizeof(block));
			if (max_abs > 0.0f) {
				// 计算出缩放因子
				float d =
				    max_abs / 7.0f;  // max representable is 7 (q=15 -> 15-8=7)
				// 把缩放因子转成fp16后存起来
				block.d = FloatToFp16(d);
				float stored_d = Fp16ToFloat(block.d);
				float id = stored_d == 0.0f ? 0.0f : 1.0f / stored_d;

				for (int j = 0; j < kQ40BlockSize / 2; ++j) {
					// 遍历块长度的一半
					float x0 = 0.0f, x1 = 0.0f;
					// 计算两个下标位置，中间相隔16个位置
					int idx0 = base + j;
					int idx1 = base + j + kQ40BlockSize / 2;

					// 使用原始值计算两个x的值
					if (idx0 < k_end) {
						x0 = src.data[idx0] * id;
					}
					if (idx1 < k_end) {
						x1 = src.data[idx1] * id;
					}

					// 计算出量化值，加8是因为uint8_t被拆分成两个4bit，一个4bit保存一个量化值
					// 模型权重有正有负，所以我们需要表示正负数，但因为是无符号，所以要将负数映射到
					// 正数区域，从原来的[-8,7]映射到[0,15]，刚好是4bit的范围
					int qi0 = static_cast<int>(std::round(x0 + 8.0f));
					int qi1 = static_cast<int>(std::round(x1 + 8.0f));
					if (qi0 < 0) {
						qi0 = 0;
					}
					if (qi0 > 15) {
						qi0 = 15;
					}
					if (qi1 < 0) {
						qi1 = 0;
					}
					if (qi1 > 15) {
						qi1 = 15;
					}
					// qi0保存在低4bit，qi1保存在高4bit
					// 所以先把qi1往左移动4位，跑到高四位上去，然后再与qi0进行或运算
					block.qs[j] = static_cast<uint8_t>(qi0 | (qi1 << 4));
				}
			} else {
				block.d = 0;
			}

			blocks.push_back(block);
		}
	}

	return blocks;
}

/**
 * @brief 反量化q40数据
 * 
 * @param blocks 
 * @param shape 
 * @return Tensor 
 */
Tensor DequantizeFromQ40(const std::vector<BlockQ40>& blocks,
                         const std::vector<int>& shape) {
	size_t total = CheckedNumel(shape, "DequantizeFromQ40");
	// 第一步还是先计算一行的元素个数、行数和一行的块数
	int row_size = shape.size() >= 2 ? shape.back() : static_cast<int>(total);
	int n_rows = shape.size() >= 2 ? static_cast<int>(total) / row_size : 1;
	int row_blocks = (row_size + kQ40BlockSize - 1) / kQ40BlockSize;
	// 总块数等于一行的块数乘以行数
	size_t expected_blocks = static_cast<size_t>(n_rows) * row_blocks;
	if (blocks.size() != expected_blocks) {
		throw std::runtime_error(
		    "DequantizeFromQ40: block count mismatch: expected " +
		    std::to_string(expected_blocks) + ", got " +
		    std::to_string(blocks.size()));
	}

	// 初始化一个张量
	Tensor dst(shape, 0.0f);
	size_t block_idx = 0;
	for (int row = 0; row < n_rows; ++row) {
		// 遍历每一行，计算行的起始偏移
		int row_offset = row * row_size;
		for (int rb = 0; rb < row_blocks; ++rb) {
			// 遍历每一块，计算每一块的起始偏移
			int base = row_offset + rb * kQ40BlockSize;
			// 计算每一块的结束偏移
			int k_end = std::min(base + kQ40BlockSize, row_offset + row_size);
			// 取出一块量化的数据
			const BlockQ40& block = blocks[block_idx++];
			// 先恢复d
			float d = Fp16ToFloat(block.d);

			// 然后把这一块对半进行遍历
			for (int j = 0; j < kQ40BlockSize / 2; ++j) {
				// 计算出两边的下标值
				int idx0 = base + j;
				int idx1 = base + j + kQ40BlockSize / 2;

				// 先恢复一个字节中的低4位，使用量化值和00001111进行与操作，那么高四位被抵消为0
				// 低四位就出来了，然后减掉之前量化时做映射范围的8
				int q0 = static_cast<int>(block.qs[j] & 0x0F) - 8;
				int q1 = static_cast<int>(block.qs[j] >> 4) - 8;

				// 然后使用计算出来的值乘以缩放因子得到原始的F32权重数据
				if (idx0 < k_end) {
					dst.data[idx0] = d * static_cast<float>(q0);
				}
				if (idx1 < k_end) {
					dst.data[idx1] = d * static_cast<float>(q1);
				}
			}
		}
	}

	return dst;
}

// ---------------------------------------------------------------------------
// Q4_1 quantize / dequantize
// ---------------------------------------------------------------------------
/**
 * @brief 真正实现量化的函数逻辑
 * 
 * @param src 
 * @return std::vector<BlockQ41> 
 */
std::vector<BlockQ41> QuantizeToQ41(const Tensor& src) {
	// 首先进行维度和形状的检查
	if (src.size() == 0) {
		return {};
	}
	if (CheckedNumel(src.shape, "QuantizeToQ41") != src.size()) {
		throw std::runtime_error("QuantizeToQ41: shape/data size mismatch");
	}

	// 计算出一行的权重元素个数
	const size_t row_size = src.num_dims() >= 2
	                            ? static_cast<size_t>(src.shape.back())
	                            : src.size();
	// 计算总共有多少行
	const size_t n_rows = src.size() / row_size;

	// 计算一行可以分成多少块
	// 这里的计算很巧妙，先取整数块，再看余数，余数非0说明还有剩余则要加1块
	// 如果余数为0，说明能够以整数块包含一行的权重个数
	const size_t row_blocks =
	    row_size / kQ41BlockSize + (row_size % kQ41BlockSize != 0);
	std::vector<BlockQ41> blocks;
	blocks.reserve(n_rows * row_blocks);

	for (size_t row = 0; row < n_rows; ++row) {
		// 先计算一行的行偏移
		const size_t row_offset = row * row_size;
		for (size_t rb = 0; rb < row_blocks; ++rb) {
			// 一行内的块偏移
			const size_t offset = rb * kQ41BlockSize;
			// 一行内块的起始位置
			const size_t base = row_offset + offset;
			// 一块的长度，为什么要加row_size - offset，这是因为
			// 防止一个块不足32个元素的情况
			const size_t block_len =
			    std::min<size_t>(kQ41BlockSize, row_size - offset);
			float min_value = src.data[base];
			float max_value = min_value;
			// 采用for循环从一个块中找出元素最大以及元素最小的值
			for (size_t j = 0; j < block_len; ++j) {
				const float value = src.data[base + j];
				if (!std::isfinite(value)) {
					throw std::runtime_error("QuantizeToQ41: non-finite input");
				}
				min_value = std::min(min_value, value);
				max_value = std::max(max_value, value);
			}

			// 构造一个Q4_1的块
			BlockQ41 block{};
			// 先转成fp16类型的数据，主要是改变它的精度
			block.d = FloatToFp16((max_value - min_value) / 15.0f);
			block.m = FloatToFp16(min_value);
			// 然后再转回成float类型数据便于后面的计算
			const float stored_d = Fp16ToFloat(block.d);
			const float stored_m = Fp16ToFloat(block.m);
			if (!std::isfinite(stored_d) || !std::isfinite(stored_m)) {
				throw std::runtime_error(
				    "QuantizeToQ41: scale/offset exceeds FP16 range");
			}

			// Use the parameters actually stored in the block. Constant blocks
			// and scales that underflow to zero are represented by m alone.
			if (stored_d != 0.0f) {
				const float inv_d = 1.0f / stored_d;
				for (size_t j = 0; j < kQ41BlockSize / 2; ++j) {
					// 量化操作
					auto quantize = [&](size_t index) -> uint8_t {
						if (index >= block_len) {
							return 0;
						}
						// 对应公式为 q = (x - min) / d
						const float q = std::round(
						    (src.data[base + index] - stored_m) * inv_d);
						// 将计算出来的量化值限定在0-15之间，比0小的返回0，比15大的返回15
						// 其它的正常返回
						return static_cast<uint8_t>(std::clamp(q, 0.0f, 15.0f));
					};
					// 采用对称的方式，0和16，1和17，分别计算出高4位和低4位的值
					const uint8_t low = quantize(j);
					const uint8_t high = quantize(j + kQ41BlockSize / 2);
					// 最后的量化值为，高4位要左移后与低4位进行或运算
					block.qs[j] = static_cast<uint8_t>(low | (high << 4));
				}
			}
			blocks.push_back(block);
		}
	}
	return blocks;
}

Tensor DequantizeFromQ41(const std::vector<BlockQ41>& blocks,
                         const std::vector<int>& shape) {
	size_t total = CheckedNumel(shape, "DequantizeFromQ41");

	int row_size = shape.size() >= 2 ? shape.back() : static_cast<int>(total);
	int n_rows = shape.size() >= 2 ? static_cast<int>(total) / row_size : 1;
	int row_blocks = (row_size + kQ41BlockSize - 1) / kQ41BlockSize;
	size_t expected_blocks = static_cast<size_t>(n_rows) * row_blocks;
	if (blocks.size() != expected_blocks) {
		throw std::runtime_error(
		    "DequantizeFromQ41: block count mismatch: expected " +
		    std::to_string(expected_blocks) + ", got " +
		    std::to_string(blocks.size()));
	}

	Tensor dst(shape, 0.0f);
	size_t block_idx = 0;
	for (int row = 0; row < n_rows; ++row) {
		int row_offset = row * row_size;
		for (int rb = 0; rb < row_blocks; ++rb) {
			int base = row_offset + rb * kQ41BlockSize;
			int k_end = std::min(base + kQ41BlockSize, row_offset + row_size);
			const BlockQ41& block = blocks[block_idx++];
			float d = Fp16ToFloat(block.d);
			float m = Fp16ToFloat(block.m);

			for (int j = 0; j < kQ41BlockSize / 2; ++j) {
				int idx0 = base + j;
				int idx1 = base + j + kQ41BlockSize / 2;
				int q0 = static_cast<int>(block.qs[j] & 0x0F);
				int q1 = static_cast<int>(block.qs[j] >> 4);
				if (idx0 < k_end) {
					dst.data[idx0] = d * static_cast<float>(q0) + m;
				}
				if (idx1 < k_end) {
					dst.data[idx1] = d * static_cast<float>(q1) + m;
				}
			}
		}
	}

	return dst;
}

// ---------------------------------------------------------------------------
// True quantized Linear (block-level on-the-fly)
// ---------------------------------------------------------------------------

// Shared helper: Linear with quantized 2D weight [out_features, in_features]
// and 1D input [in_features] or 2D input [batch, in_features].
// result rank matches input rank.
template <typename BlockType, int kBlockSize>
static Tensor LinearQuantizedImpl(const Tensor& x,
                                  const std::vector<BlockType>& blocks,
                                  const std::vector<int>& weight_shape,
                                  float (*dequant_fn)(const BlockType&, int)) {
	if (weight_shape.size() != 2) {
		throw std::runtime_error("linear_quantized: expected 2D weight shape");
	}

	int in_features;
	int rows = 1;
	bool is_1d = false;
	if (x.num_dims() == 1) {
		in_features = x.shape[0];
		is_1d = true;
	} else if (x.num_dims() == 2) {
		rows = x.shape[0];
		in_features = x.shape[1];
	} else {
		throw std::runtime_error(
		    "linear_quantized: expected x shape [in_features] or [batch, "
		    "in_features], got " +
		    x.ShapeStringShort());
	}

	int out_features = weight_shape[0];
	if (weight_shape[1] != in_features) {
		throw std::runtime_error(
		    "linear_quantized: dimension mismatch x=" + x.ShapeStringShort() +
		    " weight=" +
		    QuantizedTensor{QuantType::kF32, weight_shape}.ShapeStringShort());
	}

	int n_blocks_per_row = (in_features + kBlockSize - 1) / kBlockSize;
	size_t expected_blocks =
	    static_cast<size_t>(out_features) * n_blocks_per_row;
	if (blocks.size() != expected_blocks) {
		throw std::runtime_error(
		    "linear_quantized: block count mismatch: expected " +
		    std::to_string(expected_blocks) + ", got " +
		    std::to_string(blocks.size()));
	}

	// 兼容两种维度，要么是1维要么是2维
	Tensor result(is_1d ? std::vector<int>{out_features}
	                    : std::vector<int>{rows, out_features},
	              0.0f);

	ParallelFor(rows * out_features, [&](int begin, int end) {
		for (int flat_index = begin; flat_index < end; ++flat_index) {
			int row = flat_index / out_features;
			int j = flat_index % out_features;
			// 这表示一行输入的向量起始偏移
			const float* x_row =
			    x.data.data() + static_cast<size_t>(row) * in_features;
			float sum = 0.0f;
			int block_base = j * n_blocks_per_row;
			for (int b = 0; b < n_blocks_per_row; ++b) {
				// 遍历每一行的块
				const BlockType& block = blocks[block_base + b];
				// 块的起始位置
				int base_k = b * kBlockSize;
				// 计算当前这一块的结束位置
				int k_end = std::min(base_k + kBlockSize, in_features);
				for (int k = base_k; k < k_end; ++k) {
					// 遍历每一个量化值，将其反量化后与当前的这个输入行进行点积求和
					// 计算时，是边反量化边计算
					float w = dequant_fn(block, k - base_k);
					sum += w * x_row[k];
				}
			}
			// 将计算的结果直接存到result中
			result.data[static_cast<size_t>(row) * out_features + j] = sum;
		}
	});

	return result;
}

static float DequantQ80(const BlockQ80& block, int idx) {
	float d = Fp16ToFloat(block.d);
	return d * static_cast<float>(block.qs[idx]);
}

/**
 * @brief 根据idx取出block中一个字节的数据进行反量化
 * 
 * @param block 里面量化数组长度整个其实只有16个元素，但是外部传递的是遍历32次，所以在内部处理的时候就需要以
 16作为分界线，看到底是取量化值的低4位还是高4位
 * @param idx 
 * @return float 
 */
static float DequantQ40(const BlockQ40& block, int idx) {
	float d = Fp16ToFloat(block.d);
	int q;
	// 当idx小于16时，此时取的就是每个量化值的低4位
	if (idx < 16) {
		q = static_cast<int>(block.qs[idx] & 0x0F) - 8;
	} else {
	// 当idx大于等于16时，说明此时进入了后半段的计算，使用idx减掉16取出对应位置的量化值之后再右移4位得到
	// 高4位的值进行反量化
		q = static_cast<int>(block.qs[idx - 16] >> 4) - 8;
	}
	return d * static_cast<float>(q);
}

static float DequantQ41(const BlockQ41& block, int idx) {
	float d = Fp16ToFloat(block.d);
	float m = Fp16ToFloat(block.m);
	int q;
	if (idx < 16) {
		q = static_cast<int>(block.qs[idx] & 0x0F);
	} else {
		q = static_cast<int>(block.qs[idx - 16] >> 4);
	}
	return d * static_cast<float>(q) + m;
}

#ifdef MINI_LLAMA_USE_NEON
// NEON-optimized Q8_0 Linear: process 8 quantized values per SIMD iteration.
Tensor LinearQ80(const Tensor& x, const std::vector<BlockQ80>& weight,
                 const std::vector<int>& weight_shape) {
	if (weight_shape.size() != 2) {
		throw std::runtime_error("LinearQ80: expected 2D weight shape");
	}

	int in_features;
	int rows = 1;
	bool is_1d = false;
	if (x.num_dims() == 1) {
		in_features = x.shape[0];
		is_1d = true;
	} else if (x.num_dims() == 2) {
		rows = x.shape[0];
		in_features = x.shape[1];
	} else {
		throw std::runtime_error(
		    "LinearQ80: expected x shape [in_features] or [batch, "
		    "in_features], "
		    "got " +
		    x.ShapeStringShort());
	}

	int out_features = weight_shape[0];
	if (weight_shape[1] != in_features) {
		throw std::runtime_error(
		    "LinearQ80: dimension mismatch x=" + x.ShapeStringShort() +
		    " weight=" +
		    QuantizedTensor{QuantType::kF32, weight_shape}.ShapeStringShort());
	}

	int n_blocks_per_row = (in_features + kQ80BlockSize - 1) / kQ80BlockSize;
	size_t expected_blocks =
	    static_cast<size_t>(out_features) * n_blocks_per_row;
	if (weight.size() != expected_blocks) {
		throw std::runtime_error("LinearQ80: block count mismatch: expected " +
		                         std::to_string(expected_blocks) + ", got " +
		                         std::to_string(weight.size()));
	}

	Tensor result(is_1d ? std::vector<int>{out_features}
	                    : std::vector<int>{rows, out_features},
	              0.0f);

	ParallelFor(rows * out_features, [&](int begin, int end) {
		for (int flat_index = begin; flat_index < end; ++flat_index) {
			int row = flat_index / out_features;
			int j = flat_index % out_features;
			const float* x_row =
			    x.data.data() + static_cast<size_t>(row) * in_features;
			int block_base = j * n_blocks_per_row;
			float32x4_t sum_vec = vdupq_n_f32(0.0f);
			float sum_scalar = 0.0f;

			for (int b = 0; b < n_blocks_per_row; ++b) {
				const BlockQ80& block = weight[block_base + b];
				float d = Fp16ToFloat(block.d);
				int base_k = b * kQ80BlockSize;
				int k_end = std::min(base_k + kQ80BlockSize, in_features);
				// Process 8 elements at a time (2x float32x4_t)
				int k = base_k;
				for (; k + 8 <= k_end; k += 8) {
					// 每次加载8个
					int8x8_t q8 = vld1_s8(block.qs + (k - base_k));
					int16x8_t q16 = vmovl_s8(q8);
					
					// 扩展成int32
					int32x4_t q32_lo = vmovl_s16(vget_low_s16(q16));
					int32x4_t q32_hi = vmovl_s16(vget_high_s16(q16));

					// 转成float
					float32x4_t qf_lo = vcvtq_f32_s32(q32_lo);
					float32x4_t qf_hi = vcvtq_f32_s32(q32_hi);

					// 乘以缩放系数
					qf_lo = vmulq_n_f32(qf_lo, d);
					qf_hi = vmulq_n_f32(qf_hi, d);

					float32x4_t xf_lo = vld1q_f32(x_row + k);
					float32x4_t xf_hi = vld1q_f32(x_row + k + 4);

					// 再和输入向量相乘计算反量化值
					sum_vec = vfmaq_f32(sum_vec, qf_lo, xf_lo);
					sum_vec = vfmaq_f32(sum_vec, qf_hi, xf_hi);
				}

				// Horizontal reduce the 4-lane accumulator
				float32x2_t sum_lo = vget_low_f32(sum_vec);
				float32x2_t sum_hi = vget_high_f32(sum_vec);
				sum_lo = vadd_f32(sum_lo, sum_hi);
				float32x2_t sum_fp = vpadd_f32(sum_lo, sum_lo);
				sum_scalar += vget_lane_f32(sum_fp, 0);
				sum_vec = vdupq_n_f32(0.0f);

				// Tail: remaining elements (< 8)
				for (; k < k_end; ++k) {
					float w = d * static_cast<float>(block.qs[k - base_k]);
					sum_scalar += w * x_row[k];
				}
			}

			result.data[static_cast<size_t>(row) * out_features + j] =
			    sum_scalar;
		}
	});

	return result;
}
#else
Tensor LinearQ80(const Tensor& x, const std::vector<BlockQ80>& weight,
                 const std::vector<int>& weight_shape) {
	return LinearQuantizedImpl<BlockQ80, kQ80BlockSize>(x, weight, weight_shape,
	                                                    DequantQ80);
}
#endif

#ifdef MINI_LLAMA_USE_NEON
// NEON-optimized Q4_0 Linear: unpack 4-bit nibbles and process 16 at a time.
Tensor LinearQ40(const Tensor& x, const std::vector<BlockQ40>& weight,
                 const std::vector<int>& weight_shape) {
	if (weight_shape.size() != 2) {
		throw std::runtime_error("LinearQ40: expected 2D weight shape");
	}

	int in_features;
	bool is_1d = false;
	if (x.num_dims() == 1) {
		in_features = x.shape[0];
		is_1d = true;
	} else if (x.num_dims() == 2 && x.shape[0] == 1) {
		in_features = x.shape[1];
	} else {
		throw std::runtime_error(
		    "LinearQ40: expected x shape [in_features] or [1, in_features], "
		    "got " +
		    x.ShapeStringShort());
	}

	int out_features = weight_shape[0];
	if (weight_shape[1] != in_features) {
		throw std::runtime_error(
		    "LinearQ40: dimension mismatch x=" + x.ShapeStringShort() +
		    " weight=" +
		    QuantizedTensor{QuantType::kF32, weight_shape}.ShapeStringShort());
	}

	int n_blocks_per_row = (in_features + kQ40BlockSize - 1) / kQ40BlockSize;
	size_t expected_blocks =
	    static_cast<size_t>(out_features) * n_blocks_per_row;
	if (weight.size() != expected_blocks) {
		throw std::runtime_error("LinearQ40: block count mismatch: expected " +
		                         std::to_string(expected_blocks) + ", got " +
		                         std::to_string(weight.size()));
	}

	Tensor result(is_1d ? std::vector<int>{out_features}
	                    : std::vector<int>{1, out_features},
	              0.0f);

	const uint8x16_t mask_low = vdupq_n_u8(0x0F);
	const int8x16_t zp = vdupq_n_s8(8);

	ParallelFor(out_features, [&](int begin, int end) {
		for (int j = begin; j < end; ++j) {
			int block_base = j * n_blocks_per_row;
			float sum_scalar = 0.0f;

			for (int b = 0; b < n_blocks_per_row; ++b) {
				const BlockQ40& block = weight[block_base + b];
				float d = Fp16ToFloat(block.d);
				int base_k = b * kQ40BlockSize;
				int k_end = std::min(base_k + kQ40BlockSize, in_features);
				int len = k_end - base_k;

				// Process full 16-value half-blocks with NEON. Partial tails
				// stay scalar so loads never cross the valid input range.
				int half = 0;
				for (; half < 2 && base_k + (half + 1) * 16 <= k_end; ++half) {
					uint8x16_t packed = vld1q_u8(block.qs);
					uint8x16_t nibble8;
					if (half == 0) {
						nibble8 = vandq_u8(packed, mask_low);
					} else {
						nibble8 = vshrq_n_u8(packed, 4);
					}
					int8x16_t q8 = vsubq_s8(vreinterpretq_s8_u8(nibble8), zp);

					int16x8_t q16_lo = vmovl_s8(vget_low_s8(q8));
					int16x8_t q16_hi = vmovl_s8(vget_high_s8(q8));

					int32x4_t q32_0 = vmovl_s16(vget_low_s16(q16_lo));
					int32x4_t q32_1 = vmovl_s16(vget_high_s16(q16_lo));
					int32x4_t q32_2 = vmovl_s16(vget_low_s16(q16_hi));
					int32x4_t q32_3 = vmovl_s16(vget_high_s16(q16_hi));

					float32x4_t qf_0 = vmulq_n_f32(vcvtq_f32_s32(q32_0), d);
					float32x4_t qf_1 = vmulq_n_f32(vcvtq_f32_s32(q32_1), d);
					float32x4_t qf_2 = vmulq_n_f32(vcvtq_f32_s32(q32_2), d);
					float32x4_t qf_3 = vmulq_n_f32(vcvtq_f32_s32(q32_3), d);

					int k_off = base_k + half * 16;
					float32x4_t xf_0 = vld1q_f32(x.data.data() + k_off);
					float32x4_t xf_1 = vld1q_f32(x.data.data() + k_off + 4);
					float32x4_t xf_2 = vld1q_f32(x.data.data() + k_off + 8);
					float32x4_t xf_3 = vld1q_f32(x.data.data() + k_off + 12);

					float32x4_t sum0 = vmulq_f32(qf_0, xf_0);
					float32x4_t sum1 = vmlaq_f32(sum0, qf_1, xf_1);
					float32x4_t sum2 = vmlaq_f32(sum1, qf_2, xf_2);
					float32x4_t sum3 = vmlaq_f32(sum2, qf_3, xf_3);

					// Horizontal reduce
					float32x2_t r_lo = vget_low_f32(sum3);
					float32x2_t r_hi = vget_high_f32(sum3);
					r_lo = vadd_f32(r_lo, r_hi);
					float32x2_t r = vpadd_f32(r_lo, r_lo);
					sum_scalar += vget_lane_f32(r, 0);
				}

				// Tail: should not happen for kQ40BlockSize=32 unless
				// in_features is odd
				int k = base_k + half * 16;
				for (; k < k_end; ++k) {
					int idx = k - base_k;
					int q;
					if (idx < 16) {
						q = static_cast<int>(block.qs[idx] & 0x0F) - 8;
					} else {
						q = static_cast<int>(block.qs[idx - 16] >> 4) - 8;
					}
					float w = d * static_cast<float>(q);
					sum_scalar += w * x.data[k];
				}
			}

			result.data[j] = sum_scalar;
		}
	});

	return result;
}
#else
Tensor LinearQ40(const Tensor& x, const std::vector<BlockQ40>& weight,
                 const std::vector<int>& weight_shape) {
	return LinearQuantizedImpl<BlockQ40, kQ40BlockSize>(x, weight, weight_shape,
	                                                    DequantQ40);
}
#endif

Tensor LinearQ41(const Tensor& x, const std::vector<BlockQ41>& weight,
                 const std::vector<int>& weight_shape) {
	return LinearQuantizedImpl<BlockQ41, kQ41BlockSize>(x, weight, weight_shape,
	                                                    DequantQ41);
}

// ---------------------------------------------------------------------------
// Legacy pseudo-quantized Matmul (dequantizes to F32 then calls Matmul)
// ---------------------------------------------------------------------------
/**
 * @brief 先将权重矩阵进行反量化，然后再进行计算
 * 
 * @param weight 权重矩阵
 * @param input 输入Tensor
 * @param weight_shape 权重维度信息
 * @return Tensor 
 */
Tensor MatmulQ80(const std::vector<BlockQ80>& weight, const Tensor& input,
                 const std::vector<int>& weight_shape) {
	// 先对量化权重矩阵进行反量化			
	Tensor weight_f32 = DequantizeFromQ80(weight, weight_shape);
	// 然后再做矩阵乘法
	return Matmul(weight_f32, input);
}

Tensor MatmulQ40(const std::vector<BlockQ40>& weight, const Tensor& input,
                 const std::vector<int>& weight_shape) {
	Tensor weight_f32 = DequantizeFromQ40(weight, weight_shape);
	return Matmul(weight_f32, input);
}

// ---------------------------------------------------------------------------
// Benchmark helpers
// ---------------------------------------------------------------------------
/**
 * @brief 计算q8_0的结果和原始F32结果之间的误差值
 * 
 * @param weight 
 * @param input 
 * @return float 
 */
float CompareMatmulError(const Tensor& weight, const Tensor& input) {
	Tensor f32_result = Matmul(weight, input);
	std::vector<BlockQ80> qweight = QuantizeToQ80(weight);
	Tensor q8_result = MatmulQ80(qweight, input, weight.shape);

	if (f32_result.shape != q8_result.shape) {
		throw std::runtime_error("CompareMatmulError: shape mismatch");
	}

	float max_err = 0.0f;
	for (size_t i = 0; i < f32_result.size(); ++i) {
		float err = std::abs(f32_result.data[i] - q8_result.data[i]);
		if (err > max_err) {
			max_err = err;
		}
	}
	return max_err;
}

float CompareQ40Error(const Tensor& weight, const Tensor& input) {
	Tensor f32_result = Linear(input, weight);
	std::vector<BlockQ40> qweight = QuantizeToQ40(weight);
	Tensor q4_result = LinearQ40(input, qweight, weight.shape);

	if (f32_result.shape != q4_result.shape) {
		throw std::runtime_error("CompareQ40Error: shape mismatch");
	}

	float max_err = 0.0f;
	for (size_t i = 0; i < f32_result.size(); ++i) {
		float err = std::abs(f32_result.data[i] - q4_result.data[i]);
		if (err > max_err) {
			max_err = err;
		}
	}
	return max_err;
}

float CompareQ41Error(const Tensor& weight, const Tensor& input) {
	Tensor f32_result = Linear(input, weight);
	std::vector<BlockQ41> qweight = QuantizeToQ41(weight);
	Tensor q4_result = LinearQ41(input, qweight, weight.shape);

	if (f32_result.shape != q4_result.shape) {
		throw std::runtime_error("CompareQ41Error: shape mismatch");
	}

	float max_err = 0.0f;
	for (size_t i = 0; i < f32_result.size(); ++i) {
		float err = std::abs(f32_result.data[i] - q4_result.data[i]);
		if (err > max_err) {
			max_err = err;
		}
	}
	return max_err;
}

}  // namespace mini_llama
