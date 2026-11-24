// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/kv_cache.h"

#include <cstring>
#include <stdexcept>
#include <string>

namespace mini_llama {

static void Require4DCacheShape(const Tensor& t, const char* name) {
	if (t.num_dims() != 4) {
		throw std::runtime_error(std::string(name) + " must be a 4D tensor");
	}
}

KvCache::KvCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim) {
	keys = MakeTensor4D(n_layers, max_seq_len, n_kv_heads, head_dim, 0.0f);
	values = MakeTensor4D(n_layers, max_seq_len, n_kv_heads, head_dim, 0.0f);
}

void KvCache::Write(int layer, int pos, const Tensor& k, const Tensor& v) {
	// k: [n_kv_heads, head_dim]
	// v: [n_kv_heads, head_dim]

	// 首先校验存key和value的两个张量的维度
	Require4DCacheShape(keys, "keys");
	Require4DCacheShape(values, "values");
	if (k.num_dims() != 2 || v.num_dims() != 2 || k.shape != v.shape) {
		throw std::out_of_range(
		    "KvCache::Write expected matching [n_kv_heads, head_dim] tensors");
	}
	// 这里是GQA，使用n_kv_heads表示kv的头数(比Q的要少)
	int n_kv_heads = k.shape[0];
	// 每个头的维度
	int head_dim = k.shape[1];
	if (layer < 0 || layer >= keys.shape[0]) {
		throw std::out_of_range("KvCache::Write layer out of range");
	}
	if (pos < 0 || pos >= keys.shape[1]) {
		throw std::out_of_range("KvCache::Write position out of range");
	}
	if (n_kv_heads != keys.shape[2] || head_dim != keys.shape[3]) {
		throw std::out_of_range(
		    "KvCache::Write tensor shape does not match cache shape");
	}

	// keys shape: [n_layers, max_seq_len, n_kv_heads, head_dim]
	// We need to write at [layer, pos, :, :]

	// 跳过一层需要跨过多少张量元素
	size_t layer_stride =
	    static_cast<size_t>(keys.shape[1] * keys.shape[2] * keys.shape[3]);
	// 跳过一个token需要跨过多少张量元素
	size_t pos_stride = static_cast<size_t>(keys.shape[2] * keys.shape[3]);
	// 跳过一个头需要跨过多少张量元素
	size_t head_stride = static_cast<size_t>(head_dim);

	// 先将当前token的起始位置作为基数
	size_t base = layer * layer_stride + pos * pos_stride;

	for (int h = 0; h < n_kv_heads; ++h) {
		// 按照kv向量的头数一个头一个头的数据进行拷贝
		size_t offset = base + h * head_stride;
		std::memcpy(&keys.data[offset], &k.data[h * head_dim],
		            head_dim * sizeof(float));
		std::memcpy(&values.data[offset], &v.data[h * head_dim],
		            head_dim * sizeof(float));
	}
}

const float* KvCache::KeyPtr(int layer, int pos, int kv_head) const {
	Require4DCacheShape(keys, "keys");
	// 首先检查传递参数的合法性
	if (layer < 0 || layer >= keys.shape[0] || pos < 0 ||
	    pos >= keys.shape[1] || kv_head < 0 || kv_head >= keys.shape[2]) {
		throw std::out_of_range("KvCache::KeyPtr FlatIndex out of range");
	}
	// 然后计算层、token以及头的跨度数据
	size_t layer_stride =
	    static_cast<size_t>(keys.shape[1] * keys.shape[2] * keys.shape[3]);
	size_t pos_stride = static_cast<size_t>(keys.shape[2] * keys.shape[3]);
	size_t head_stride = static_cast<size_t>(keys.shape[3]);
	// 最后从kv cache的keys中取出对应位置张量的起始地址，使用地址是为了避免在进行Attention计算时多余的张量数据拷贝
	return &keys.data[layer * layer_stride + pos * pos_stride +
	                  kv_head * head_stride];
}

const float* KvCache::ValuePtr(int layer, int pos, int kv_head) const {
	Require4DCacheShape(values, "values");
	if (layer < 0 || layer >= values.shape[0] || pos < 0 ||
	    pos >= values.shape[1] || kv_head < 0 || kv_head >= values.shape[2]) {
		throw std::out_of_range("KvCache::ValuePtr FlatIndex out of range");
	}
	size_t layer_stride = static_cast<size_t>(
	    values.shape[1] * values.shape[2] * values.shape[3]);
	size_t pos_stride = static_cast<size_t>(values.shape[2] * values.shape[3]);
	size_t head_stride = static_cast<size_t>(values.shape[3]);
	return &values.data[layer * layer_stride + pos * pos_stride +
	                    kv_head * head_stride];
}

}  // namespace mini_llama
