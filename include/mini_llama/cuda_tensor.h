// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_CUDA_TENSOR_H_
#define INCLUDE_MINI_LLAMA_CUDA_TENSOR_H_

#include <cstddef>
#include <string>
#include <vector>

#include "mini_llama/cuda_runtime.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

/**
 * @brief 构建带有形状信息的cudaTensor，内部存在字节级的封装CudaDeviceBuffer
 * 
 */
class CudaTensor {
   public:
	CudaTensor() = default;
	explicit CudaTensor(const std::vector<int>& shape, int device_id = 0);

	CudaTensor(const CudaTensor&) = delete;
	CudaTensor& operator=(const CudaTensor&) = delete;

	CudaTensor(CudaTensor&&) noexcept = default;
	CudaTensor& operator=(CudaTensor&&) noexcept = default;

	void Reset();
	void Reset(const std::vector<int>& shape, int device_id = 0);
	/**
	 * @brief 将主机内存中的张量数据上传到GPU并填充CudaTensor对象中的成员
	 * 
	 * @param src 主机中张量数据的起始地址
	 */
	void UploadFrom(const Tensor& src);
	/**
	 * @brief 将GPU上的张量数据从显存中拷贝到当前主机内存中
	 * 
	 * @return Tensor 传递下来的张量对象
	 */
	Tensor Download() const;
	void DownloadTo(Tensor& dst) const;

	void* data();
	const void* data() const;

	const std::vector<int>& shape() const { return shape_; }

	int num_dims() const { return static_cast<int>(shape_.size()); }

	size_t size() const { return numel_; }

	size_t bytes() const { return buffer_.bytes(); }

	int device_id() const { return device_id_; }

	bool empty() const { return buffer_.empty(); }

	std::string ShapeStringShort() const;

   private:
    // 张量数据的形状 
	std::vector<int> shape_;
	// 元素个数
	size_t numel_ = 0;
	// 设备所在的device id
	int device_id_ = 0;
	// 显存内部的生命周期管理类对象
	CudaDeviceBuffer buffer_;
};

/**
 * @brief 
 * 
 * @param src 
 * @param device_id 
 * @return CudaTensor 
 */
CudaTensor CudaTensorFromHost(const Tensor& src, int device_id = 0);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_CUDA_TENSOR_H_
