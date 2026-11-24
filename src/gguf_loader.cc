// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/gguf_loader.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "mini_llama/gguf.h"
#include "mini_llama/model.h"
#include "mini_llama/quant.h"

namespace mini_llama {

// ---------------------------------------------------------------------------
// Tensor name mapping: GGUF (llama.cpp style) -> our naming
// ---------------------------------------------------------------------------

/**
 * @brief 做GGUF中张量名和MiniLllamaModel结构体属性名的映射
 * 
 * @param gguf_name 
 * @return std::string 
 */
static std::string MapTensorName(const std::string& gguf_name) {
	// Embedding
	if (gguf_name == "token_embd.weight") {
		return "token_embedding";
	}
	// Output norm
	if (gguf_name == "output_norm.weight") {
		return "final_norm";
	}
	// Output projection (lm_head) — may be tied with token_embd
	if (gguf_name == "output.weight") {
		return "lm_head";
	}

	// Layer tensors: blk.{L}.{name}.weight
	// Parse "blk.n.xxx.weight"
	if (gguf_name.rfind("blk.", 0) == 0) {
		size_t dot1 = gguf_name.find('.', 4);
		if (dot1 == std::string::npos) {
			return "";
		}
		std::string layer_str = gguf_name.substr(4, dot1 - 4);
		size_t dot2 = gguf_name.find('.', dot1 + 1);
		if (dot2 == std::string::npos) {
			return "";
		}
		std::string op = gguf_name.substr(dot1 + 1, dot2 - dot1 - 1);
		std::string suffix = gguf_name.substr(dot2 + 1);
		std::string mapped_op;
		if (suffix == "bias") {
			if (op == "attn_q") {
				mapped_op = "bq";
			} else if (op == "attn_k") {
				mapped_op = "bk";
			} else if (op == "attn_v") {
				mapped_op = "bv";
			} else {
				return "";
			}
		} else if (suffix != "weight") {
			return "";
		} else if (op == "attn_norm") {
			mapped_op = "attention_norm";
		} else if (op == "attn_q") {
			mapped_op = "wq";
		} else if (op == "attn_k") {
			mapped_op = "wk";
		} else if (op == "attn_v") {
			mapped_op = "wv";
		} else if (op == "attn_output") {
			mapped_op = "wo";
		} else if (op == "ffn_norm") {
			mapped_op = "ffn_norm";
		} else if (op == "ffn_gate") {
			mapped_op = "w_gate";
		} else if (op == "ffn_up") {
			mapped_op = "w_up";
		} else if (op == "ffn_down") {
			mapped_op = "w_down";
		} else {
			return "";
		}

		return "layers." + layer_str + "." + mapped_op;
	}

	// 针对识别不出来的名字直接返回空串
	return "";
}

// ---------------------------------------------------------------------------
// Convert GGUF shape to our row-major Tensor view.
// GGUF uses ne0 as the fastest-changing dimension. For a 2D weight advertised
// as [in_features, out_features], the raw bytes are already laid out as
// out_features rows of in_features values. Our Tensor wants [out, in].
// ---------------------------------------------------------------------------

/**
 * @brief GGUF的维度信息是列数(最内层，连续变化的维度)在前，行数在后。映射到内存中就是行数在前，列数在后
 * 
 * @param gguf_shape 
 * @return std::vector<int> 
 */
static std::vector<int> MapShape(const std::vector<uint64_t>& gguf_shape) {
	std::vector<int> shape;
	shape.reserve(gguf_shape.size());
	if (gguf_shape.size() == 2) {
		// 如果是二维则交换两个维度
		shape.push_back(static_cast<int>(gguf_shape[1]));
		shape.push_back(static_cast<int>(gguf_shape[0]));
		return shape;
	}
	for (uint64_t dim : gguf_shape) {
		// 如果是其它维度则不变
		shape.push_back(static_cast<int>(dim));
	}
	return shape;
}

// ---------------------------------------------------------------------------
// Load quantized tensor from raw bytes (F32, Q8_0, Q4_0, Q4_1)
// ---------------------------------------------------------------------------

/**
 * @brief 根据tensor的数据类型，
 * 
 * @param raw 从文件中读取的权重原始字节
 * @param shape 维度形状
 * @param ggml_type 张量元素存储类型编号
 * @return QuantizedTensor 
 */
static QuantizedTensor LoadQuantizedTensor(const std::vector<uint8_t>& raw,
                                           const std::vector<int>& shape,
                                           int ggml_type) {
	
	// 1、先计算张量的元素个数
	size_t num_elements = 1;
	for (int d : shape) {
		num_elements *= static_cast<size_t>(d);
	}

	QuantizedTensor q;
	q.shape = shape;

	// 2、根据张量的元素类型进行重新存储
	switch (ggml_type) {
		// 直接保存浮点数
		case kGgmlTypeF32: {
			q.type = QuantType::kF32;
			// 期待字节数就是：元素个数乘以float的字节数
			size_t expected_bytes = num_elements * sizeof(float);
			if (raw.size() != expected_bytes) {
				throw std::runtime_error("F32 tensor size mismatch: expected " +
				                         std::to_string(expected_bytes) +
				                         " bytes, got " +
				                         std::to_string(raw.size()));
			}
			// 分配元素个数后直接做内存字节拷贝
			q.f32_data.resize(num_elements);
			std::memcpy(q.f32_data.data(), raw.data(), expected_bytes);
			break;
		}
		// Q80量化
		case kGgmlTypeQ80: {
			q.type = QuantType::kQ80;
			/**
			 * 关于下面这个块数的计算公式的推导：
			 * 首先对于一个数组来说，如果按块来进行分：
			 * 块大小假设为m，那么就有下面的规律
			 * 下标在[0,m-1]，下标为0块
			 * 下标在[m,2m-1]，下标为1块
			 * 下标在[2m,3m-1]，下标为2块
			 * 。。。。。
			 * 如果我们总共有n个元素进行分块的话，那最后一个元素的下标为
			 * n-1，此时它所属块的下标是多少？
			 * 设某个元素的下标为i，它所在块下标为b，那么b块之前已经有b个完整块了
			 * 所以前面总共有m*b，那么块下标为b的这块的起始下标为m*b，下一个快的起始元素下标为
			 * m(b+1)，所以就有：mb <= i < m(b+1)，此时两边同除以m就得到b <= i/m < b+1
			 * 因为有向下取整，所以b = i/m，那当i为最后一个元素时，就有b = (n-1)/m
			 * 所以最后一个元素的块下标为b，实际块数比b多1，所以实际块数为：(n-1)/m + 1
			 * 那么通分后就有：(n+m-1)/m，这就是下面这个计算方法的严格推导
			 */
			size_t n_blocks =
			    (num_elements + kQ80BlockSize - 1) / kQ80BlockSize;
			size_t expected_bytes = n_blocks * sizeof(BlockQ80);
			if (raw.size() != expected_bytes) {
				throw std::runtime_error(
				    "Q8_0 tensor size mismatch: expected " +
				    std::to_string(expected_bytes) + " bytes, got " +
				    std::to_string(raw.size()));
			}
			// 分配块数个单位
			q.q8_0_data.resize(n_blocks);
			// 然后逐字节拷贝进去
			std::memcpy(q.q8_0_data.data(), raw.data(), expected_bytes);
			break;
		}
		// Q40量化
		case kGgmlTypeQ40: {
			q.type = QuantType::kQ40;
			// 先使用整数向上取整计算块数
			size_t n_blocks =
			    (num_elements + kQ40BlockSize - 1) / kQ40BlockSize;
			
			size_t expected_bytes = n_blocks * sizeof(BlockQ40);
			if (raw.size() != expected_bytes) {
				throw std::runtime_error(
				    "Q4_0 tensor size mismatch: expected " +
				    std::to_string(expected_bytes) + " bytes, got " +
				    std::to_string(raw.size()));
			}
			q.q4_0_data.resize(n_blocks);
			std::memcpy(q.q4_0_data.data(), raw.data(), expected_bytes);
			break;
		}
		// Q41量化
		case kGgmlTypeQ41: {
			q.type = QuantType::kQ41;
			size_t n_blocks =
			    (num_elements + kQ41BlockSize - 1) / kQ41BlockSize;
			size_t expected_bytes = n_blocks * sizeof(BlockQ41);
			if (raw.size() != expected_bytes) {
				throw std::runtime_error(
				    "Q4_1 tensor size mismatch: expected " +
				    std::to_string(expected_bytes) + " bytes, got " +
				    std::to_string(raw.size()));
			}
			q.q4_1_data.resize(n_blocks);
			std::memcpy(q.q4_1_data.data(), raw.data(), expected_bytes);
			break;
		}
		default:
			throw std::runtime_error("Unsupported GGUF tensor type: " +
			                         std::to_string(ggml_type));
	}

	return q;
}

// ---------------------------------------------------------------------------
// Shape validation helpers
// ---------------------------------------------------------------------------

static void RequireShape(const Tensor& tensor, const std::vector<int>& expected,
                         const std::string& name) {
	tensor.AssertShape(expected, name.c_str());
}

static void RequireShape(const QuantizedTensor& tensor,
                         const std::vector<int>& expected,
                         const std::string& name) {
	tensor.AssertShape(expected, name.c_str());
}

static void RequireOptionalShape(const Tensor& tensor,
                                 const std::vector<int>& expected,
                                 const std::string& name) {
	if (tensor.data.empty()) {
		return;
	}
	RequireShape(tensor, expected, name);
}

/**
 * @brief 校验每个张量的维度信息
 * 
 * @param model 
 */
static void ValidateLoadedShapes(const MiniLlamaModel& model) {
	const ModelConfig& c = model.config;
	const int q_dim = c.n_heads * c.head_dim;
	const int kv_dim = c.n_kv_heads * c.head_dim;

	RequireShape(model.token_embedding, {c.vocab_size, c.dim},
	             "GGUF token_embedding");
	RequireShape(model.final_norm, {c.dim}, "GGUF final_norm");
	RequireShape(model.lm_head, {c.vocab_size, c.dim}, "GGUF lm_head");

	if (model.layers.size() != static_cast<size_t>(c.n_layers)) {
		throw std::runtime_error("GGUF layer count mismatch: expected " +
		                         std::to_string(c.n_layers) + ", got " +
		                         std::to_string(model.layers.size()));
	}

	for (int layer = 0; layer < c.n_layers; ++layer) {
		const LayerWeights& lw = model.layers[layer];
		const std::string prefix = "GGUF layers." + std::to_string(layer) + ".";
		RequireShape(lw.attention_norm, {c.dim}, prefix + "attention_norm");
		RequireShape(lw.wq, {q_dim, c.dim}, prefix + "wq");
		RequireShape(lw.wk, {kv_dim, c.dim}, prefix + "wk");
		RequireShape(lw.wv, {kv_dim, c.dim}, prefix + "wv");
		RequireOptionalShape(lw.bq, {q_dim}, prefix + "bq");
		RequireOptionalShape(lw.bk, {kv_dim}, prefix + "bk");
		RequireOptionalShape(lw.bv, {kv_dim}, prefix + "bv");
		RequireShape(lw.wo, {c.dim, q_dim}, prefix + "wo");
		RequireShape(lw.ffn_norm, {c.dim}, prefix + "ffn_norm");
		RequireShape(lw.w_gate, {c.hidden_dim, c.dim}, prefix + "w_gate");
		RequireShape(lw.w_up, {c.hidden_dim, c.dim}, prefix + "w_up");
		RequireShape(lw.w_down, {c.dim, c.hidden_dim}, prefix + "w_down");
	}
}

// ---------------------------------------------------------------------------
// Build ModelConfig from GGUF metadata
// ---------------------------------------------------------------------------

static ModelConfig BuildConfig(const std::string& path) {
	ModelConfig config;

	// 从metadata中读取int和float类型的数据
	auto get_u32 = [&](const std::string& key, int fallback) -> int {
		int64_t v = 0;
		if (GgufGetMetadataInt(path, key, v)) {
			return static_cast<int>(v);
		}
		return fallback;
	};

	auto get_f32 = [&](const std::string& key, float fallback) -> float {
		double v = 0.0;
		if (GgufGetMetadataFloat(path, key, v)) {
			return static_cast<float>(v);
		}
		return fallback;
	};

	// Try qwen2.* keys first, then llama.* keys as fallback

	// 检测到架构是qwen2的时候就将rope置为kNeoX
	std::string arch;
	if (GgufGetMetadataString(path, "general.architecture", arch) &&
	    arch == "qwen2") {
		config.rope_type = RopeType::kNeoX;
	}

	// 这里先把llama对应的结果查出来，但是不作为最终结果
	// 然后作为fallback降级结果跟随qwen2传进去
	// 如果说qwen2的找到了，那么就直接使用qwen2的结果，否则就返回fallback的结果，也就是
	// llama对应的结果
	config.n_layers =
	    get_u32("qwen2.block_count", get_u32("llama.block_count", 0));
	config.dim =
	    get_u32("qwen2.embedding_length", get_u32("llama.embedding_length", 0));
	config.hidden_dim = get_u32("qwen2.feed_forward_length",
	                            get_u32("llama.feed_forward_length", 0));
	config.n_heads = get_u32("qwen2.attention.head_count",
	                         get_u32("llama.attention.head_count", 0));
	config.n_kv_heads = get_u32("qwen2.attention.head_count_kv",
	                            get_u32("llama.attention.head_count_kv", 0));
	config.max_seq_len =
	    get_u32("qwen2.context_length", get_u32("llama.context_length", 0));
	config.rope_theta = get_f32("qwen2.rope.freq_base",
	                            get_f32("llama.rope.freq_base", 10000.0f));
	// 读取均方归一化中的eps参数
	config.rms_norm_eps =
	    get_f32("qwen2.attention.layer_norm_rms_epsilon",
	            get_f32("llama.attention.layer_norm_rms_epsilon", 1e-5f));

	// vocab_size from tokenizer.ggml.tokens array length

	// 读取toknizer词表
	std::vector<std::string> tokens;
	if (GgufGetMetadataStringArray(path, "tokenizer.ggml.tokens", tokens)) {
		config.vocab_size = static_cast<int>(tokens.size());
	}

	// Validate
	if (config.n_layers <= 0 || config.dim <= 0 || config.hidden_dim <= 0 ||
	    config.n_heads <= 0 || config.n_kv_heads <= 0 ||
	    config.max_seq_len <= 0 || config.vocab_size <= 0) {
		throw std::runtime_error("Invalid model config from GGUF: dim=" +
		                         std::to_string(config.dim) +
		                         " layers=" + std::to_string(config.n_layers) +
		                         " heads=" + std::to_string(config.n_heads) +
		                         " vocab=" + std::to_string(config.vocab_size));
	}
	if (config.dim % config.n_heads != 0) {
		throw std::runtime_error("dim not divisible by n_heads");
	}
	if (config.n_heads % config.n_kv_heads != 0) {
		throw std::runtime_error("n_heads not divisible by n_kv_heads");
	}

	config.head_dim = config.dim / config.n_heads;

	std::cout << "GGUF config: vocab=" << config.vocab_size
	          << " dim=" << config.dim << " layers=" << config.n_layers
	          << " heads=" << config.n_heads
	          << " kv_heads=" << config.n_kv_heads
	          << " head_dim=" << config.head_dim
	          << " hidden_dim=" << config.hidden_dim
	          << " max_seq_len=" << config.max_seq_len
	          << " rope_theta=" << config.rope_theta
	          << " rms_norm_eps=" << config.rms_norm_eps << std::endl;

	return config;
}

// ---------------------------------------------------------------------------
// LoadGgufModel
// ---------------------------------------------------------------------------


/**
 * @brief 主要作用是加载配置信息、Tensor名映射、加载权重数据
 * 
 * @param gguf_path 
 * @return MiniLlamaModel 
 */
MiniLlamaModel LoadGgufModel(const std::string& gguf_path) {
	MiniLlamaModel model;

	// Parse header
	GgufReader reader;
	if (!reader.Load(gguf_path)) {
		model.load_error = "GGUF parse error: " + reader.load_error;
		return model;
	}

	// Build config
	try {
		model.config = BuildConfig(gguf_path);
	} catch (const std::exception& e) {
		model.load_error = std::string("Failed to build config: ") + e.what();
		return model;
	}

	const ModelConfig& c = model.config;
	std::cout << "GGUF config: vocab=" << c.vocab_size << " dim=" << c.dim
	          << " layers=" << c.n_layers << " heads=" << c.n_heads
	          << " kv_heads=" << c.n_kv_heads << " head_dim=" << c.head_dim
	          << " hidden_dim=" << c.hidden_dim
	          << " max_seq_len=" << c.max_seq_len << std::endl;

	// Index tensors by mapped name
	std::map<std::string, const GgufTensorInfo*> tensor_map;
	for (const auto& t : reader.tensors) {
		std::string mapped = MapTensorName(t.name);
		if (mapped.empty()) {
			continue;
		}
		// 记录到tensor_map中
		tensor_map[mapped] = &t;
	}

	// Helper to Load a weight tensor (returns QuantizedTensor)
	auto load_weight = [&](const std::string& mapped_name) -> QuantizedTensor {
		
		// 获取tensor的info信息
		auto it = tensor_map.find(mapped_name);
		if (it == tensor_map.end()) {
			throw std::runtime_error("Missing tensor: " + mapped_name);
		}
		const GgufTensorInfo* info = it->second;
		std::vector<int> shape = MapShape(info->shape);

		// raw中保存的就是一个tensor的所有字节数据
		std::vector<uint8_t> raw =
		    ReadGgufTensorRaw(gguf_path, *info, reader.data_offset);
		if (raw.empty() && info->type != kGgmlTypeF32) {
			throw std::runtime_error("Failed to read tensor: " + mapped_name);
		}

		return LoadQuantizedTensor(raw, shape, info->type);
	};

	auto load_optional_weight =
	    [&](const std::string& mapped_name) -> QuantizedTensor {
		if (tensor_map.find(mapped_name) == tensor_map.end()) {
			return QuantizedTensor();
		}
		return load_weight(mapped_name);
	};

	// Helper to Load an F32 tensor (norm, embedding, bias).
	// If the source is quantized, dequantize to F32 on-the-fly.
	auto load_f32 = [&](const std::string& mapped_name) -> Tensor {
		QuantizedTensor q = load_weight(mapped_name);
		switch (q.type) {
			case QuantType::kF32:
				return ToTensor(q);
			case QuantType::kQ80:
				return DequantizeFromQ80(q.q8_0_data, q.shape);
			case QuantType::kQ40:
				return DequantizeFromQ40(q.q4_0_data, q.shape);
			case QuantType::kQ41:
				return DequantizeFromQ41(q.q4_1_data, q.shape);
		}
		throw std::runtime_error("Cannot convert tensor to F32: " +
		                         mapped_name);
	};

	auto load_optional_f32 = [&](const std::string& mapped_name) -> Tensor {
		if (tensor_map.find(mapped_name) == tensor_map.end()) {
			return Tensor();
		}
		return load_f32(mapped_name);
	};

	try {
		// Token embedding (F32)
		// 使用float32反量化是为了简化查表的实现，直接读取浮点数
		model.token_embedding = load_f32("token_embedding");

		// Layers
		model.layers.resize(c.n_layers);
		for (int layer = 0; layer < c.n_layers; ++layer) {
			LayerWeights& lw = model.layers[layer];
			const std::string prefix = "layers." + std::to_string(layer) + ".";
			lw.attention_norm = load_f32(prefix + "attention_norm");
			lw.wq = load_weight(prefix + "wq");
			lw.wk = load_weight(prefix + "wk");
			lw.wv = load_weight(prefix + "wv");
			// bias使用float32反量化也是因为数据量不大
			lw.bq = load_optional_f32(prefix + "bq");
			lw.bk = load_optional_f32(prefix + "bk");
			lw.bv = load_optional_f32(prefix + "bv");
			lw.wo = load_weight(prefix + "wo");
			lw.ffn_norm = load_f32(prefix + "ffn_norm");
			lw.w_gate = load_weight(prefix + "w_gate");
			lw.w_up = load_weight(prefix + "w_up");
			lw.w_down = load_weight(prefix + "w_down");
		}

		// Final norm (F32)
		// 使用float32反量化是因为数据量不大
		model.final_norm = load_f32("final_norm");

		// LM head: use output.weight if present, otherwise tie with
		// token_embedding
		auto it = tensor_map.find("lm_head");
		if (it != tensor_map.end()) {
			model.lm_head = load_weight("lm_head");
		} else {
			// Copy token_embedding for tied weights
			// 如果没有的话，就内存做一个拷贝，把token_embedding拷贝一次出来给lm_head
			model.lm_head = ToQuantizedTensor(model.token_embedding);
		}

		ValidateLoadedShapes(model);
	} catch (const std::exception& e) {
		model.load_error =
		    std::string("Failed to load GGUF weights: ") + e.what();
		return model;
	}

	model.loaded = true;
	std::cout << "GGUF model loaded successfully from " << gguf_path
	          << std::endl;
	return model;
}

}  // namespace mini_llama
