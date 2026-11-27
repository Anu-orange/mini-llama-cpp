// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_MODEL_H_
#define INCLUDE_MINI_LLAMA_MODEL_H_

#include <memory>
#include <string>
#include <vector>

#include "mini_llama/cuda_runtime.h"
#include "mini_llama/quantized_tensor.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

enum class RopeType {
	kNormal,
	kNeoX,
};

// 模型配置信息
struct ModelConfig {
	int vocab_size = 128;
	int dim = 32;
	int hidden_dim = 86;
	int n_layers = 2;
	int n_heads = 4;
	int n_kv_heads = 4;
	int head_dim = 8;
	int max_seq_len = 128;
	float rope_theta = 10000.0f;
	float rms_norm_eps = 1e-5f;
	RopeType rope_type = RopeType::kNormal;
};

struct LayerWeights {
	Tensor attention_norm;   // [dim] F32
	QuantizedTensor wq;      // [n_heads * head_dim, dim]  F32/Q8_0/Q4_0
	QuantizedTensor wk;      // [n_kv_heads * head_dim, dim]
	QuantizedTensor wv;      // [n_kv_heads * head_dim, dim]
	Tensor bq;               // optional [n_heads * head_dim] F32
	Tensor bk;               // optional [n_kv_heads * head_dim] F32
	Tensor bv;               // optional [n_kv_heads * head_dim] F32
	QuantizedTensor wo;      // [dim, n_heads * head_dim]
	Tensor ffn_norm;         // [dim] F32
	QuantizedTensor w_gate;  // [hidden_dim, dim]
	QuantizedTensor w_up;    // [hidden_dim, dim]
	QuantizedTensor w_down;  // [dim, hidden_dim]
};

/**
 * @brief 上传到显存中的权重信息记录
 * 
 */
struct CudaUploadedWeight {
	std::string name; // 权重张量唯一标识名
	QuantType type = QuantType::kF32; // 权重量化或精度类型
	bool linear = true; // 该权重是否服务于线性投影层
	std::vector<int> shape; // 原始形状维度
	size_t block_count = 0; // 量化区块的总数(当前项目是总元素除以32)
	CudaDeviceBuffer buffer; // 实际持有物理显存的buffer实例
};

/**
 * @brief 模型级权重信息
 * 
 */
struct CudaModelWeights {
	int device_id = 0; // 设备id
	size_t uploaded_weight_count = 0; // 上传的权重张量数量
	size_t uploaded_bytes = 0; // 上传的权重实际占用的字节数
	size_t linear_calls = 0; // 线性层Linear的调用次数
	size_t activation_calls = 0; // 非线性层及基础操作的调用次数
	size_t attention_calls = 0; // 注意力层的调用次数
	size_t attention_cpu_fallbacks = 0;
	size_t kv_cache_write_bytes = 0; // kv cache中写入的字节数
	size_t kv_cache_read_bytes = 0; // kv cache读取的字节数
	size_t host_to_device_copies = 0; // host向device拷贝的次数
	size_t device_to_host_copies = 0; // device向host拷贝的次数
	size_t host_to_device_bytes = 0; // host向device拷贝的字节数
	size_t device_to_host_bytes = 0; // device向host拷贝的字节数
	std::vector<CudaUploadedWeight> weights; // 存储已经上传的权重对象
};

struct MiniLlamaModel {
	ModelConfig config;  // 模型配置
	bool loaded = false; // 是否被加载
	std::string load_error; // 错误信息
	Tensor token_embedding;            // [vocab_size, dim] F32，词表
	std::vector<LayerWeights> layers;  // [n_layers]，多个层的权重
	Tensor final_norm;                 // [dim] F32 经过lm_head后的最后一个归一化权重
	QuantizedTensor lm_head;           // [vocab_size, dim] F32/Q8_0/Q4_0
	std::shared_ptr<CudaModelWeights> cuda_weights; 
};

// Return the actual bytes consumed by model weights in their current format.
size_t ModelWeightBytes(const MiniLlamaModel& model);

// Return the bytes the same weights would consume if stored as F32.
size_t ModelWeightBytesF32(const MiniLlamaModel& model);

// Upload CUDA-supported resident weights and keep CPU weights intact.
/**
 * @brief 上传模型权重到GPU显存
 * 
 * @param model 模型信息
 * @param device_id 显存设备id
 */
void UploadModelWeightsToCuda(MiniLlamaModel& model, int device_id = 0);

/**
 * @brief 清除显存中的模型权重
 * 
 * @param model 
 */
void ClearModelCudaWeights(MiniLlamaModel& model);

/**
 * @brief 模型是否存在CUDA权重
 * 
 * @param model 
 * @return true 
 * @return false 
 */
bool ModelHasCudaWeights(const MiniLlamaModel& model);

/**
 * @brief 上传的权重张量数量，记住是数量，例如：Wq、Wk、Wv，这是三个张量
 * 
 * @param model 
 * @return size_t 
 */
size_t ModelCudaUploadedWeightCount(const MiniLlamaModel& model);

size_t ModelCudaUploadedF32WeightCount(const MiniLlamaModel& model);

size_t ModelCudaUploadedQ80WeightCount(const MiniLlamaModel& model);

size_t ModelCudaUploadedQ40WeightCount(const MiniLlamaModel& model);

size_t ModelCudaUploadedQ41WeightCount(const MiniLlamaModel& model);

/**
 * @brief 上传到显存中实际占用的字节数
 * 
 * @param model 
 * @return size_t 
 */
size_t ModelCudaMemoryBytes(const MiniLlamaModel& model);

void ResetModelCudaRuntimeStats(MiniLlamaModel& model);

size_t ModelCudaLinearCalls(const MiniLlamaModel& model);

size_t ModelCudaActivationCalls(const MiniLlamaModel& model);

size_t ModelCudaAttentionCalls(const MiniLlamaModel& model);

size_t ModelCudaAttentionCpuFallbacks(const MiniLlamaModel& model);

size_t ModelCudaKvCacheWriteBytes(const MiniLlamaModel& model);

size_t ModelCudaKvCacheReadBytes(const MiniLlamaModel& model);

size_t ModelCudaHostToDeviceCopies(const MiniLlamaModel& model);

size_t ModelCudaDeviceToHostCopies(const MiniLlamaModel& model);

size_t ModelCudaHostToDeviceBytes(const MiniLlamaModel& model);

size_t ModelCudaDeviceToHostBytes(const MiniLlamaModel& model);

// Convert all Linear weight QuantizedTensors from F32 to Q8_0 in-place.
// Embedding, norm, and bias tensors remain unchanged.
void QuantizeModelToQ80(MiniLlamaModel& model);

// Convert all Linear weight QuantizedTensors from F32 to Q4_0 in-place.
// Embedding, norm, and bias tensors remain unchanged.
void QuantizeModelToQ40(MiniLlamaModel& model);

// Convert Linear weights to Q4_1, dequantizing other formats first if needed.
// Embedding, norm, and bias tensors remain unchanged. Invalidates CUDA uploads.
void QuantizeModelToQ41(MiniLlamaModel& model);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_MODEL_H_
