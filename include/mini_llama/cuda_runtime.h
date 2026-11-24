// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_CUDA_RUNTIME_H_
#define INCLUDE_MINI_LLAMA_CUDA_RUNTIME_H_

#include <cstddef>
#include <string>

namespace mini_llama {

/**
 * @brief 记录设备GPU的硬件信息
 * 
 */
struct CudaDeviceInfo {
	int id = 0; // 设备编号，对应命令中的--device
	std::string name; // GPU名称
	// 两个字段一起表示计算能力版本，比如一个8一个9，合起来就是8.9
	int compute_major = 0; 
	int compute_minor = 0;
	// 设备总显存
	size_t total_memory_bytes = 0;
	// 驱动支持的CUDA版本
	int driver_version = 0;
	// 当前使用的CUDA Runtime版本
	int runtime_version = 0;
};

enum class CudaMemcpyKind {
	kHostToDevice,
	kDeviceToHost,
	kDeviceToDevice,
};

/**
 * @brief 判断当前构建的二进制是启用CUDA
 * 
 * @return true 
 * @return false 
 */
bool CudaRuntimeBuilt();
int CudaDeviceCount();
void CudaSetDevice(int device_id);
CudaDeviceInfo CudaGetDeviceInfo(int device_id);
std::string CudaFormatDeviceInfo(const CudaDeviceInfo& info);

void* CudaMallocBytes(size_t bytes);
void CudaFreeBytes(void* ptr);
void CudaMemcpyBytes(void* dst, const void* src, size_t bytes,
                     CudaMemcpyKind kind);

/**
 * @brief 这个类很重要，使用RAII的思想来管理底层申请的显存
 * 
 */
class CudaDeviceBuffer {
   public:
	CudaDeviceBuffer() = default;
	explicit CudaDeviceBuffer(size_t bytes, int device_id = 0);
	~CudaDeviceBuffer();

	CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
	CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;

	CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept;
	CudaDeviceBuffer& operator=(CudaDeviceBuffer&& other) noexcept;

	void Reset();
	void Reset(size_t bytes, int device_id = 0);
	/**
	 * @brief 将数据从CPU拷贝到GPU显存中
	 * 
	 * @param src 
	 * @param bytes 
	 */
	void Upload(const void* src, size_t bytes);

	/**
	 * @brief 将数据从GPU显存中拷贝到CPU中
	 * 
	 * @param dst 
	 * @param bytes 
	 */
	void Download(void* dst, size_t bytes) const;

	void* data() { return data_; }

	const void* data() const { return data_; }

	size_t bytes() const { return bytes_; }

	int device_id() const { return device_id_; }

	bool empty() const { return data_ == nullptr; }

   private:
	void ReleaseNoexcept() noexcept;

	// 显存内容的起始地址
	void* data_ = nullptr;
	// 总共有多少字节的数据
	size_t bytes_ = 0;
	// 显存设备id
	int device_id_ = 0;
};

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_CUDA_RUNTIME_H_
