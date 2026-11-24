// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_CUDA_KV_CACHE_H_
#define INCLUDE_MINI_LLAMA_CUDA_KV_CACHE_H_

#include <cstddef>

#include "mini_llama/cuda_runtime.h"
#include "mini_llama/cuda_tensor.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

bool CudaKvCacheBuilt();

class CudaKvCache {
   public:
	CudaKvCache() = default;
	CudaKvCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim,
	            int device_id = 0);

	CudaKvCache(const CudaKvCache&) = delete;
	CudaKvCache& operator=(const CudaKvCache&) = delete;
	CudaKvCache(CudaKvCache&&) noexcept = default;
	CudaKvCache& operator=(CudaKvCache&&) noexcept = default;

	void Reset();
	void Reset(int n_layers, int max_seq_len, int n_kv_heads, int head_dim,
	           int device_id = 0);
	void Clear();
	/**
	 * @brief HOST侧的显存拷贝，也就是要将CPU上的数据拷贝到GPU显存中
	 * 
	 * @param layer 
	 * @param pos 
	 * @param k 
	 * @param v 
	 */
	void Write(int layer, int pos, const Tensor& k, const Tensor& v);
	/**
	 * @brief DEVICE侧的拷贝，将GPU上的数据拷贝到GPU显存中
	 * 
	 * @param layer 
	 * @param pos 
	 * @param k 
	 * @param v 
	 */
	void WriteDevice(int layer, int pos, const CudaTensor& k,
	                 const CudaTensor& v);
	/**
	 * @brief 从GPU显存中拉取key数组
	 * 
	 * @param layer 
	 * @param pos 
	 * @return Tensor 
	 */
	Tensor ReadKey(int layer, int pos) const;
	/**
	 * @brief 从GPU显存中拉取value数组
	 * 
	 * @param layer 
	 * @param pos 
	 * @return Tensor 
	 */
	Tensor ReadValue(int layer, int pos) const;
	Tensor ReadKeyHead(int layer, int pos, int kv_head) const;
	Tensor ReadValueHead(int layer, int pos, int kv_head) const;

	const void* keys_data() const;
	const void* values_data() const;

	bool empty() const;
	size_t bytes() const;

	int n_layers() const { return n_layers_; }

	int max_seq_len() const { return max_seq_len_; }

	int n_kv_heads() const { return n_kv_heads_; }

	int head_dim() const { return head_dim_; }

	int device_id() const { return device_id_; }

   private:
	void ValidateIndices(int layer, int pos) const;
	void ValidateHeadIndex(int kv_head) const;
	void ValidateWriteTensors(const Tensor& k, const Tensor& v) const;
	/**
	 * @brief 根据层数和位置信息计算
	 * 
	 * @param layer 
	 * @param pos 
	 * @return size_t 
	 */
	size_t SlotOffsetBytes(int layer, int pos) const;
	size_t HeadOffsetBytes(int layer, int pos, int kv_head) const;
	
	// 和CPU中kv cache一样，属于四维张量
	int n_layers_ = 0;
	int max_seq_len_ = 0;
	int n_kv_heads_ = 0;
	int head_dim_ = 0;
	int device_id_ = 0;
	// 使用CudaDeviceBuffer来管理
	// 底层在显存中通过cuda上的内存分配进行分配显存
	CudaDeviceBuffer keys_;
	CudaDeviceBuffer values_;
};

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_CUDA_KV_CACHE_H_
