// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/gguf.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mini_llama {

// ---------------------------------------------------------------------------
// Type name helpers
// ---------------------------------------------------------------------------
const char* GgmlTypeName(uint32_t type) {
	switch (type) {
		case kGgmlTypeF32:
			return "F32";
		case kGgmlTypeF16:
			return "F16";
		case kGgmlTypeQ40:
			return "Q4_0";
		case kGgmlTypeQ41:
			return "Q4_1";
		case kGgmlTypeQ80:
			return "Q8_0";
		default:
			return "UNKNOWN";
	}
}

const char* GgufValueTypeName(uint32_t type) {
	switch (type) {
		case kGgufTypeUint8:
			return "uint8";
		case kGgufTypeInt8:
			return "int8";
		case kGgufTypeUint16:
			return "uint16";
		case kGgufTypeInt16:
			return "int16";
		case kGgufTypeUint32:
			return "uint32";
		case kGgufTypeInt32:
			return "int32";
		case kGgufTypeFloat32:
			return "float32";
		case kGgufTypeBool:
			return "bool";
		case kGgufTypeString:
			return "string";
		case kGgufTypeArray:
			return "array";
		case kGgufTypeUint64:
			return "uint64";
		case kGgufTypeInt64:
			return "int64";
		case kGgufTypeFloat64:
			return "float64";
		default:
			return "unknown";
	}
}

// ---------------------------------------------------------------------------
// Internal little-endian readers
// ---------------------------------------------------------------------------
namespace {

constexpr uint64_t kDefaultAlignment = 32;
constexpr uint64_t kMaxAlignment = 1024 * 1024;
constexpr uint64_t kMaxStringBytes = 16 * 1024 * 1024;
constexpr uint64_t kMaxMetadataEntries = 1000000;
constexpr uint64_t kMaxTensorEntries = 1000000;
constexpr uint64_t kMaxArrayElements = 10000000;
constexpr uint32_t kMaxTensorDims = 8;
constexpr uint64_t kQuantBlockSize = 32;

template <typename T>
bool ReadLe(std::ifstream& f, T& out) {
	char buf[sizeof(T)];
	if (!f.read(buf, sizeof(T))) {
		return false;
	}
	std::memcpy(&out, buf, sizeof(T));
	return true;
}

bool ReadU32(std::ifstream& f, uint32_t& out) { return ReadLe(f, out); }

bool ReadU64(std::ifstream& f, uint64_t& out) { return ReadLe(f, out); }

bool ReadString(std::ifstream& f, std::string& out) {
	uint64_t len = 0;
	// 1、先读取字符串占用的字节数，底层的ReadLe会修改f的位置以及len的值
	// 读取结束后，len就变成了后面紧跟的字符串的长度了
	if (!ReadU64(f, len)) {
		return false;
	}
	// 长度不超过16MB且也不能超过二进制流式类型的最大值
	if (len > kMaxStringBytes ||
	    len > static_cast<uint64_t>(
	              std::numeric_limits<std::streamsize>::max())) {
		return false;
	}
	// 重新将out进行resize
	out.resize(len);
	// 2、再按照长度进行读取，把真正的字符串存到out中
	if (len > 0 && !f.read(out.data(), static_cast<std::streamsize>(len))) {
		return false;
	}
	return true;
}

bool SkipValue(std::ifstream& f, uint32_t type) {
	switch (type) {
		case kGgufTypeUint8:
		case kGgufTypeInt8: {
			char dummy;
			return f.read(&dummy, 1).good();
		}
		case kGgufTypeUint16:
		case kGgufTypeInt16: {
			char dummy[2];
			return f.read(dummy, 2).good();
		}
		case kGgufTypeUint32:
		case kGgufTypeInt32:
		case kGgufTypeFloat32: {
			char dummy[4];
			return f.read(dummy, 4).good();
		}
		case kGgufTypeBool: {
			char dummy;
			return f.read(&dummy, 1).good();
		}
		case kGgufTypeString: {
			std::string dummy;
			return ReadString(f, dummy);
		}
		case kGgufTypeUint64:
		case kGgufTypeInt64:
		case kGgufTypeFloat64: {
			char dummy[8];
			return f.read(dummy, 8).good();
		}
		case kGgufTypeArray: {
			uint32_t elem_type = 0;
			uint64_t len = 0;
			if (!ReadU32(f, elem_type) || !ReadU64(f, len)) {
				return false;
			}
			if (len > kMaxArrayElements) {
				return false;
			}
			for (uint64_t i = 0; i < len; ++i) {
				if (!SkipValue(f, elem_type)) {
					return false;
				}
			}
			return true;
		}
		default:
			return false;
	}
}
/**
 * @brief 计算两个uint64整数相称是否会溢出
 * 
 * @param a 
 * @param b 
 * @param out 
 * @return true 
 * @return false 
 */
bool MultiplyChecked(uint64_t a, uint64_t b, uint64_t& out) {
	if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
		return false;
	}
	// 保存他们的乘积
	out = a * b;
	return true;
}

bool TensorNumel(const GgufTensorInfo& tensor, uint64_t& out) {
	out = 1;
	for (uint64_t dim : tensor.shape) {
		if (dim == 0) {
			return false;
		}
		if (!MultiplyChecked(out, dim, out)) {
			return false;
		}
	}
	return true;
}

/**
 * @brief 计算指定类型Tensor底层占用字节数
 * 
 * @param tensor 
 * @param out 
 * @return true 
 * @return false 
 */
bool TensorDataSize(const GgufTensorInfo& tensor, uint64_t& out) {
	// 元素个数初始为0
	uint64_t num_elements = 0;
	if (!TensorNumel(tensor, num_elements)) {
		return false;
	}

	switch (tensor.type) {
		case kGgmlTypeF32:
			return MultiplyChecked(num_elements, 4, out);
		case kGgmlTypeF16:
			return MultiplyChecked(num_elements, 2, out);
		
		// 对于下面三种量化数据类型，元素个数一定要是32的整数倍
		case kGgmlTypeQ40: {
			if (num_elements % kQuantBlockSize != 0) {
				return false;
			}
			// 计算出有多少这个块
			uint64_t n_blocks = num_elements / kQuantBlockSize;
			return MultiplyChecked(n_blocks, 18, out);
		}
		case kGgmlTypeQ41: {
			if (num_elements % kQuantBlockSize != 0) {
				return false;
			}
			uint64_t n_blocks = num_elements / kQuantBlockSize;
			return MultiplyChecked(n_blocks, 20, out);
		}
		case kGgmlTypeQ80: {
			if (num_elements % kQuantBlockSize != 0) {
				return false;
			}
			uint64_t n_blocks = num_elements / kQuantBlockSize;
			return MultiplyChecked(n_blocks, 34, out);
		}
		default:
			out = 0;
			return true;
	}
}

bool AddChecked(uint64_t a, uint64_t b, uint64_t& out) {
	if (b > std::numeric_limits<uint64_t>::max() - a) {
		return false;
	}
	out = a + b;
	return true;
}

// Read a scalar value and convert to string representation.
bool ReadValueAsString(std::ifstream& f, uint32_t type, std::string& out) {
	std::ostringstream oss;
	switch (type) {
		case kGgufTypeUint8: {
			uint8_t v = 0;
			if (!ReadLe(f, v)) {
				return false;
			}
			oss << static_cast<unsigned>(v);
			break;
		}
		case kGgufTypeInt8: {
			int8_t v = 0;
			if (!ReadLe(f, v)) {
				return false;
			}
			oss << static_cast<int>(v);
			break;
		}
		case kGgufTypeUint16: {
			uint16_t v = 0;
			if (!ReadLe(f, v)) {
				return false;
			}
			oss << v;
			break;
		}
		case kGgufTypeInt16: {
			int16_t v = 0;
			if (!ReadLe(f, v)) {
				return false;
			}
			oss << v;
			break;
		}
		case kGgufTypeUint32: {
			uint32_t v = 0;
			if (!ReadU32(f, v)) {
				return false;
			}
			oss << v;
			break;
		}
		case kGgufTypeInt32: {
			int32_t v = 0;
			if (!ReadLe(f, v)) {
				return false;
			}
			oss << v;
			break;
		}
		case kGgufTypeFloat32: {
			float v = 0.0f;
			if (!ReadLe(f, v)) {
				return false;
			}
			oss << v;
			break;
		}
		case kGgufTypeBool: {
			uint8_t v = 0;
			if (!ReadLe(f, v)) {
				return false;
			}
			oss << (v ? "true" : "false");
			break;
		}
		case kGgufTypeString: {
			if (!ReadString(f, out)) {
				return false;
			}
			return true;
		}
		case kGgufTypeUint64: {
			uint64_t v = 0;
			if (!ReadU64(f, v)) {
				return false;
			}
			oss << v;
			break;
		}
		case kGgufTypeInt64: {
			int64_t v = 0;
			if (!ReadLe(f, v)) {
				return false;
			}
			oss << v;
			break;
		}
		case kGgufTypeFloat64: {
			double v = 0.0;
			if (!ReadLe(f, v)) {
				return false;
			}
			oss << v;
			break;
		}
		// 对于数组类型需要特殊处理
		/**
		 *  GGUF 的每条元数据基本结构是：
		 *	[key：GGUF字符串][value_type：uint32][value：根据类型决定布局]
         *  当 value_type 是 kGgufTypeArray 时，value 的布局是：
		 *	[元素类型：uint32][元素数量：uint64][元素0][元素1]……
		 *  所以针对于数组类型，整体就是上面的2和4两行连起来：
		 *  [key：GGUF字符串][value_type：uint32][元素类型：uint32][元素数量：uint64][元素0][元素1]……
		 *	当前分支中只读取元素类型和数组长度
		 */
		case kGgufTypeArray: {
			uint32_t elem_type = 0;
			uint64_t len = 0;
			// key和value_type已经读完了，现在读取元素类型和数组长度
			// 因为数据类型底层使用uint32_t的编号来区分，所以调用ReadU32来读取编号值获得类型
			if (!ReadU32(f, elem_type) || !ReadU64(f, len)) {
				return false;
			}
			if (len > kMaxArrayElements) {
				return false;
			}
			oss << "[" << len << " x " << GgufValueTypeName(elem_type) << "]";
			for (uint64_t i = 0; i < len; ++i) {
				if (!SkipValue(f, elem_type)) {
					return false;
				}
			}
			break;
		}
		default:
			return false;
	}
	out = oss.str();
	return true;
}

/**
 * @brief 从记录的元数据数组中获取对齐字节数
 * 
 * @param metadata 
 * @return uint64_t 
 */
uint64_t ParseAlignment(const std::vector<GgufMetadataKv>& metadata) {
	// 首先默认是32字节
	uint64_t alignment = kDefaultAlignment;
	for (const auto& kv : metadata) {
		// 依次遍历，寻找对应的对齐字节数名次
		if (kv.key != "general.alignment") {
			continue;
		}
		try {                   
			size_t parsed = 0;
			// 对其string的值进行解析转换为uint64_t类型
			alignment = std::stoull(kv.value_str, &parsed);
			if (parsed != kv.value_str.size()) {
				// 这说明value_str字符串里面不只有数字字符，还有别的字符，导致parsed不能到最后
				return 0;
			}
		} catch (const std::exception&) {
			return 0;
		}
	}
	if (alignment == 0 || alignment > kMaxAlignment) {
		return 0;
	}

	// 判断对齐字节数是否为2的整数次幂，使用位运算加快速度
	// 因为2的整数次幂在二进制中只有一个bit位上是1，其它全0，减1后当前位为0，其他位全1
	if ((alignment & (alignment - 1)) != 0) {
		return 0;
	}
	// 返回解析后的对齐字节数
	return alignment;
}

}  // namespace

// ---------------------------------------------------------------------------
// GgufReader::Load
// ---------------------------------------------------------------------------
/**
 * @brief 加载gguf文件的主要函数
 * 
 * @param path 
 * @return true 
 * @return false 
 */
bool GgufReader::Load(const std::string& path) {
	version = 0;
	n_tensors = 0;
	n_metadata = 0;
	metadata.clear();
	tensors.clear();
	data_offset = 0;
	load_error.clear();

	std::ifstream f(path, std::ios::binary);
	if (!f.is_open()) {
		load_error = "failed to open: " + path;
		return false;
	}

	// Magic
	char magic[4];
	if (!f.read(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) {
		// 读取四个字节并与"gguf"进行比较，如果不是GGUF则直接报错
		load_error = "invalid GGUF magic";
		return false;
	}

	// Version
	if (!ReadU32(f, version)) {
		// 读取uint32_t的版本号信息
		load_error = "failed to read version";
		return false;
	}
	if (version != 3) {
		load_error = "unsupported GGUF version: " + std::to_string(version);
		return false;
	}

	// n_tensors, n_metadata
	// 读取张量个数和metadata键值对个数
	/**
	 * 这里对张量个数要说明一下：张量个数不是浮点元素个数，而是一个单独的张量对象单位(无论维度是一/二/三维)
	 * 比如在Transformer中，一层就有atten_norm、W_q、W_k、W_v、W_o、ffn_norm、W_fate、W_up、W_down
	 * 这9个张量，要是有N层的话，那么整个模型就是9N个张量
	 */
	if (!ReadU64(f, n_tensors) || !ReadU64(f, n_metadata)) {
		load_error = "failed to read header counts";
		return false;
	}
	// 限制张量个数和键值对数不可超过固定大小
	if (n_tensors > kMaxTensorEntries) {
		load_error = "too many tensors: " + std::to_string(n_tensors);
		return false;
	}
	if (n_metadata > kMaxMetadataEntries) {
		load_error = "too many metadata entries: " + std::to_string(n_metadata);
		return false;
	}

	// Metadata
	metadata.reserve(n_metadata);
	// 用一个set来记录已读取的key
	std::unordered_set<std::string> metadata_keys;
	for (uint64_t i = 0; i < n_metadata; ++i) {
		// 遍历整个metadata
		GgufMetadataKv kv;
		if (!ReadString(f, kv.key)) {
			load_error =
			    "failed to read metadata key at FlatIndex " + std::to_string(i);
			return false;
		}
		// 当实际插入时second才为true，否则为false
		// 没插入进去报错，说明重复了
		if (!metadata_keys.insert(kv.key).second) {
			load_error = "duplicate metadata key: " + kv.key;
			return false;
		}
		if (!ReadU32(f, kv.type)) {
			load_error = "failed to read metadata type for key: " + kv.key;
			return false;
		}
		// 根据类型来读数据，看属于哪种类型
		if (!ReadValueAsString(f, kv.type, kv.value_str)) {
			load_error = "failed to read metadata value for key: " + kv.key;
			return false;
		}
		metadata.push_back(std::move(kv));
	}

	// Tensor info
	/**
	 *  例如有一个 F32 权重矩阵：
		name       = "wq"
		维度数量    = 2
		dimensions = [32, 64]
		type       = 0，即 F32
		offset     = 0

		它的记录布局就是：
		[uint64：2]              ← 名称长度
		[字节：w q]              ← 名称内容
		[uint32：2]              ← 维度数量
		[uint64：32]             ← 第0维
		[uint64：64]             ← 第1维
		[uint32：0]              ← F32类型编号
		[uint64：0]              ← 数据偏移
		这条记录共：
		8 + 2 + 4 + 2×8 + 4 + 8 = 42 字节
	 * 
	 */
	tensors.reserve(n_tensors);
	for (uint64_t i = 0; i < n_tensors; ++i) {
		GgufTensorInfo info;
		if (!ReadString(f, info.name)) {
			load_error =
			    "failed to read tensor name at FlatIndex " + std::to_string(i);
			return false;
		}
		uint32_t n_dims = 0;
		if (!ReadU32(f, n_dims)) {
			load_error = "failed to read tensor n_dims for: " + info.name;
			return false;
		}
		if (n_dims == 0 || n_dims > kMaxTensorDims) {
			load_error = "invalid tensor n_dims for: " + info.name;
			return false;
		}
		info.shape.resize(n_dims);
		for (uint32_t d = 0; d < n_dims; ++d) {
			// 根据维度数依次读取每个维度的值
			if (!ReadU64(f, info.shape[d])) {
				load_error = "failed to read tensor shape for: " + info.name;
				return false;
			}
			if (info.shape[d] == 0) {
				load_error = "invalid zero tensor dimension for: " + info.name;
				return false;
			}
		}
		// 读取张量数据类型，也是以uint32_t作为编号的，取值只有0，1，2，3，8
		if (!ReadU32(f, info.type)) {
			load_error = "failed to read tensor type for: " + info.name;
			return false;
		}
		// 再读取偏移(一定是相对于data_section起始位置的相对偏移)
		if (!ReadU64(f, info.offset)) {
			load_error = "failed to read tensor offset for: " + info.name;
			return false;
		}
		tensors.push_back(std::move(info));
	}

	std::unordered_set<std::string> names;
	// names记录每个张量的名字
	for (const auto& tensor : tensors) {
		if (!names.insert(tensor.name).second) {
			load_error = "duplicate tensor name: " + tensor.name;
			return false;
		}
	}

	// Compute data offset (aligned to general.alignment, default 32).

	// 先获取当前文件指针所在位置
	std::streampos current = f.tellg();
	if (current < 0) {
		load_error = "failed to determine GGUF data offset";
		return false;
	}
	data_offset = static_cast<uint64_t>(current);
	uint64_t alignment = ParseAlignment(metadata);
	if (alignment == 0) {
		load_error = "invalid general.alignment";
		return false;
	}

	// 计算到下一个对齐
	/**
	 * data_offset % alignment：表示在对齐区间多走了多少
	 * 使用对齐字节数减去多走的就是对齐区间内剩余要补充的
	 * 那为什么还需要再与alignment求一次模呢，这种情况对应当前位置恰好就在对齐点上
	 * 如果不除以模则会多补充alignment个字节，浪费空间
	 * 
	 */
	uint64_t padding = (alignment - (data_offset % alignment)) % alignment;

	// 将当前位置往后移动padding个位置，作为Tensor数据区块的起始位置
	data_offset += padding;

	// 把文件读取指针移动到文件末尾
	f.seekg(0, std::ios::end);
	// 获取文件指针位置，此时在文件末尾
	std::streampos end = f.tellg();
	if (end < 0 || static_cast<uint64_t>(end) < data_offset) {
		load_error = "GGUF file ended before tensor data section";
		return false;
	}

	// 计算出整个文件大小
	uint64_t file_size = static_cast<uint64_t>(end);

	for (const auto& tensor : tensors) {
		// 遍历每个tesnor信息
		uint64_t tensor_size = 0;

		// 获取tensor所占的字节数
		if (!TensorDataSize(tensor, tensor_size)) {
			load_error = "invalid tensor data size for: " + tensor.name;
			return false;
		}
		if (tensor_size == 0) {
			continue;
		}

		// 计算当前这个tensor相对Tensor数据区块起始位置的存放位置
		// 就是用Tensor数据区块的起始位置加上相对偏移
		uint64_t tensor_start = 0;	
		if (!AddChecked(data_offset, tensor.offset, tensor_start)) {
			load_error = "tensor offset overflow for: " + tensor.name;
			return false;
		}

		// 计算tensor张量结束的位置
		uint64_t tensor_end = 0;
		// 用tensor张量起始位置加上张量字节数就是张量结束位置
		if (!AddChecked(tensor_start, tensor_size, tensor_end)) {
			load_error = "tensor data range overflow for: " + tensor.name;
			return false;
		}

		// 如果张量结束位置大于了文件大小，那么就报错
		if (tensor_end > file_size) {
			load_error = "tensor data outside file bounds: " + tensor.name;
			return false;
		}
	}

	return true;
}

// ---------------------------------------------------------------------------
// InspectGguf
// ---------------------------------------------------------------------------
void InspectGguf(const GgufReader& reader) {
	std::cout << "=== GGUF Info ===" << std::endl;
	std::cout << "  version:    " << reader.version << std::endl;
	std::cout << "  n_tensors:  " << reader.n_tensors << std::endl;
	std::cout << "  n_metadata: " << reader.n_metadata << std::endl;
	std::cout << "  data_offset: " << reader.data_offset << std::endl;

	if (!reader.metadata.empty()) {
		std::cout << std::endl;
		std::cout << "=== Metadata ===" << std::endl;
		for (const auto& kv : reader.metadata) {
			std::cout << "  " << kv.key << " (" << GgufValueTypeName(kv.type)
			          << "): " << kv.value_str << std::endl;
		}
	}

	if (!reader.tensors.empty()) {
		std::cout << std::endl;
		std::cout << "=== Tensors ===" << std::endl;
		for (const auto& t : reader.tensors) {
			std::cout << "  " << t.name << std::endl;
			std::cout << "    shape:  [";
			for (size_t i = 0; i < t.shape.size(); ++i) {
				if (i > 0) {
					std::cout << ", ";
				}
				std::cout << t.shape[i];
			}
			std::cout << "]" << std::endl;
			std::cout << "    dtype:  " << GgmlTypeName(t.type) << " ("
			          << t.type << ")" << std::endl;
			std::cout << "    offset: " << t.offset << std::endl;
		}
	}
}

// ---------------------------------------------------------------------------
// Tensor data reading (inside anonymous namespace for helper access)
// ---------------------------------------------------------------------------

bool GgufTensorDataSize(const GgufTensorInfo& tensor, uint64_t& out_size) {
	return TensorDataSize(tensor, out_size);
}

std::vector<uint8_t> ReadGgufTensorRaw(const std::string& path,
                                       const GgufTensorInfo& info,
                                       uint64_t data_offset) {
	std::vector<uint8_t> result;

	uint64_t tensor_size = 0;
	// 读取Tensor张量的字节数
	if (!TensorDataSize(info, tensor_size)) {
		return result;
	}
	if (tensor_size == 0) {
		return result;
	}

	std::ifstream f(path, std::ios::binary);
	if (!f.is_open()) {
		return result;
	}

	uint64_t tensor_start = 0;
	// 通过该函数将a和b的和作为当前tensor的起始位置
	if (!AddChecked(data_offset, info.offset, tensor_start)) {
		return result;
	}

	if (tensor_start >
	    static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
		return result;
	}

	// 从起始位置开始跳到当前tensor的起始位置
	f.seekg(static_cast<std::streamoff>(tensor_start), std::ios::beg);
	if (!f.good()) {
		return result;
	}

	// 数组元素类型是uint8_t

	// 这里也是分配的字节数，读的时候也是读tensoe_size和字节出来到resule中
	result.resize(tensor_size);
	f.read(reinterpret_cast<char*>(result.data()),
	       static_cast<std::streamsize>(tensor_size));
	if (static_cast<uint64_t>(f.gcount()) != tensor_size) {
		result.clear();
		return result;
	}

	return result;
}

// ---------------------------------------------------------------------------
// Internal: open GGUF, skip to metadata section, find key by name.
// On success, f is positioned after the key string, before the type uint32.
// ---------------------------------------------------------------------------

bool OpenAndFindMetadataKey(std::ifstream& f, const std::string& path,
                            const std::string& key, uint32_t& out_type) {
	f.open(path, std::ios::binary);
	if (!f.is_open()) {
		return false;
	}

	// Skip magic (4) + version (4)
	f.seekg(8, std::ios::beg);
	if (!f.good()) {
		return false;
	}

	uint64_t n_tensors = 0;
	uint64_t n_metadata = 0;
	if (!ReadU64(f, n_tensors) || !ReadU64(f, n_metadata)) {
		return false;
	}

	for (uint64_t i = 0; i < n_metadata; ++i) {
		std::string read_key;
		// 1、记录读到的key是什么
		if (!ReadString(f, read_key)) {
			return false;
		}

		// 2、这个key是一个什么类型
		uint32_t type = 0;
		if (!ReadU32(f, type)) {
			return false;
		}
		if (read_key == key) {
			out_type = type;
			return true;
		}
		// Skip value
		if (!SkipValue(f, type)) {
			return false;
		}
	}
	return false;  // key not found
}

// ---------------------------------------------------------------------------
// Typed metadata reading
// ---------------------------------------------------------------------------
bool GgufGetMetadataInt(const std::string& path, const std::string& key,
                        int64_t& out) {
	std::ifstream f;
	uint32_t type = 0;
	if (!OpenAndFindMetadataKey(f, path, key, type)) {
		return false;
	}
	switch (type) {
		case kGgufTypeUint8: {
			uint8_t v = 0;
			if (!ReadLe(f, v)) {
				return false;
			}
			out = static_cast<int64_t>(v);
			return true;
		}
		case kGgufTypeInt8: {
			int8_t v = 0;
			if (!ReadLe(f, v)) {
				return false;
			}
			out = static_cast<int64_t>(v);
			return true;
		}
		case kGgufTypeUint16: {
			uint16_t v = 0;
			if (!ReadLe(f, v)) {
				return false;
			}
			out = static_cast<int64_t>(v);
			return true;
		}
		case kGgufTypeInt16: {
			int16_t v = 0;
			if (!ReadLe(f, v)) {
				return false;
			}
			out = static_cast<int64_t>(v);
			return true;
		}
		case kGgufTypeUint32: {
			uint32_t v = 0;
			if (!ReadU32(f, v)) {
				return false;
			}
			out = static_cast<int64_t>(v);
			return true;
		}
		case kGgufTypeInt32: {
			int32_t v = 0;
			if (!ReadLe(f, v)) {
				return false;
			}
			out = static_cast<int64_t>(v);
			return true;
		}
		case kGgufTypeUint64: {
			uint64_t v = 0;
			if (!ReadU64(f, v)) {
				return false;
			}
			out = static_cast<int64_t>(v);
			return true;
		}
		case kGgufTypeInt64: {
			int64_t v = 0;
			if (!ReadLe(f, v)) {
				return false;
			}
			out = v;
			return true;
		}
		default:
			return false;
	}
}

bool GgufGetMetadataFloat(const std::string& path, const std::string& key,
                          double& out) {
	std::ifstream f;
	uint32_t type = 0;
	if (!OpenAndFindMetadataKey(f, path, key, type)) {
		return false;
	}
	switch (type) {
		case kGgufTypeFloat32: {
			float v = 0.0f;
			if (!ReadLe(f, v)) {
				return false;
			}
			out = static_cast<double>(v);
			return true;
		}
		case kGgufTypeFloat64: {
			double v = 0.0;
			if (!ReadLe(f, v)) {
				return false;
			}
			out = v;
			return true;
		}
		default:
			return false;
	}
}

bool GgufGetMetadataString(const std::string& path, const std::string& key,
                           std::string& out) {
	std::ifstream f;
	uint32_t type = 0;
	if (!OpenAndFindMetadataKey(f, path, key, type)) {
		return false;
	}
	if (type != kGgufTypeString) {
		return false;
	}
	return ReadString(f, out);
}

/**
 * @brief 从metadata中读取tokenizer字符串数组信息
 * 
 * @param path 
 * @param key 
 * @param out 
 * @return true 
 * @return false 
 */
bool GgufGetMetadataStringArray(const std::string& path, const std::string& key,
                                std::vector<std::string>& out) {
	std::ifstream f;
	uint32_t type = 0;
	if (!OpenAndFindMetadataKey(f, path, key, type)) {
		return false;
	}
	if (type != kGgufTypeArray) {
		return false;
	}

	// 到这里说明给定的key是数组类型
	uint32_t elem_type = 0;
	uint64_t len = 0;
	// 依次再读取数组的元素类型和元素个数
	if (!ReadU32(f, elem_type) || !ReadU64(f, len)) {
		return false;
	}
	if (elem_type != kGgufTypeString) {
		return false;
	}
	// 说明元素类型是string
	if (len > kMaxArrayElements) {
		return false;
	}
	// 调用reserve分配len个元素的容量空间
	out.clear();
	out.reserve(len);
	// 遍历len次，读取字符串到s中
	for (uint64_t i = 0; i < len; ++i) {
		std::string s;
		if (!ReadString(f, s)) {
			return false;
		}
		// 每读取一个就调用move做资源转移，然后进行push_back
		out.push_back(std::move(s));
	}
	return true;
}

bool GgufGetMetadataIntArray(const std::string& path, const std::string& key,
                             std::vector<int32_t>& out) {
	std::ifstream f;
	uint32_t type = 0;
	if (!OpenAndFindMetadataKey(f, path, key, type)) {
		return false;
	}
	if (type != kGgufTypeArray) {
		return false;
	}
	uint32_t elem_type = 0;
	uint64_t len = 0;
	if (!ReadU32(f, elem_type) || !ReadU64(f, len)) {
		return false;
	}
	if (elem_type != kGgufTypeInt32) {
		return false;
	}
	if (len > kMaxArrayElements) {
		return false;
	}
	out.clear();
	out.reserve(len);
	for (uint64_t i = 0; i < len; ++i) {
		int32_t v = 0;
		if (!ReadLe(f, v)) {
			return false;
		}
		out.push_back(v);
	}
	return true;
}

}  // namespace mini_llama
