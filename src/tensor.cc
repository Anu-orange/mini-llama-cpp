// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/tensor.h"

#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace mini_llama {

namespace {

const char* CallerName(const char* caller) {
	return caller == nullptr ? "Tensor" : caller;
}

std::string ShapeToString(const std::vector<int>& shape) {
	std::string s = "[";
	for (size_t i = 0; i < shape.size(); ++i) {
		if (i > 0) {
			s += ", ";
		}
		s += std::to_string(shape[i]);
	}
	s += "]";
	return s;
}

/**
 * @brief 计算张量的总元素数量，同时检查shape是否合法，乘法是否溢出
 *
 * @param shape
 * @param caller
 * @return size_t
 */
size_t CheckedNumel(const std::vector<int>& shape, const char* caller) {
	size_t total = 1;
	for (size_t axis = 0; axis < shape.size(); ++axis) {
		const int dim = shape[axis];
		if (dim <= 0) {
			// 维度不合法
			throw std::runtime_error(
			    std::string(CallerName(caller)) + ": dimension at axis " +
			    std::to_string(axis) + " must be positive, got " +
			    std::to_string(dim) + " in shape " + ShapeToString(shape));
		}
		const size_t dim_size = static_cast<size_t>(dim);
		// 当前维算上的话元素总数已经超过size_t的最大值了，此时报错
		if (total > std::numeric_limits<size_t>::max() / dim_size) {
			throw std::runtime_error(std::string(CallerName(caller)) +
			                         ": shape element count overflow for " +
			                         ShapeToString(shape));
		}
		total *= dim_size;
	}
	return total;
}

/**
 * @brief 检查张量的维度数量是否符合期待
 *
 * @param t
 * @param expected
 * @param caller
 */
void CheckRank(const Tensor& t, int expected, const char* caller) {
	if (t.num_dims() != expected) {
		throw std::runtime_error(std::string(caller) + " expected " +
		                         std::to_string(expected) + "D tensor, got " +
		                         t.ShapeStringShort());
	}
}

/**
 * @brief 检查传递进来张量对象的维度是否越界
 *
 * @param t
 * @param axis
 * @param flat_index
 * @param caller
 */
void CheckAxisIndex(const Tensor& t, int axis, int flat_index,
                    const char* caller) {
	const int dim = t.shape[axis];
	if (flat_index < 0 || flat_index >= dim) {
		// 超过了范围则报错
		throw std::out_of_range(
		    std::string(caller) + ": FlatIndex " + std::to_string(flat_index) +
		    " out of range for axis " + std::to_string(axis) + " with size " +
		    std::to_string(dim) + " in tensor " + t.ShapeStringShort());
	}
}

}  // namespace

Tensor::Tensor(const std::vector<int>& input_shape, float fill)
    : shape(input_shape) {
	const size_t total = CheckedNumel(shape, "Tensor constructor");
	// 使用fill填充total数量个元素
	data.resize(total, fill);
}

size_t Tensor::FlatIndex(const std::vector<int>& indices) const {
	if (indices.size() != shape.size()) {
		throw std::runtime_error("Tensor::FlatIndex expected " +
		                         std::to_string(shape.size()) +
		                         " indices for tensor " + ShapeStringShort() +
		                         ", got " + std::to_string(indices.size()));
	}
	// 检查传进来的每一维
	for (size_t axis = 0; axis < indices.size(); ++axis) {
		CheckAxisIndex(*this, static_cast<int>(axis), indices[axis],
		               "Tensor::FlatIndex");
	}

	size_t flat = 0;
	size_t stride = 1;
	// 从最后一维开始往前计算
	for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
		flat += static_cast<size_t>(indices[i]) * stride;
		stride *= static_cast<size_t>(shape[i]);
	}
	return flat;
}

float& Tensor::At(const std::vector<int>& indices) {
	return data[FlatIndex(indices)];
}

float Tensor::At(const std::vector<int>& indices) const {
	return data[FlatIndex(indices)];
}

// ---------------------------------------------------------------------------
// Convenience 1D accessor
// ---------------------------------------------------------------------------
float& Tensor::At1(int i) {
	CheckRank(*this, 1, "At1");
	CheckAxisIndex(*this, 0, i, "At1");
	return data[i];
}

float Tensor::At1(int i) const {
	CheckRank(*this, 1, "At1");
	CheckAxisIndex(*this, 0, i, "At1");
	return data[i];
}

// ---------------------------------------------------------------------------
// Convenience 2D accessor
// ---------------------------------------------------------------------------
float& Tensor::At2(int i, int j) {
	CheckRank(*this, 2, "At2");
	CheckAxisIndex(*this, 0, i, "At2");
	CheckAxisIndex(*this, 1, j, "At2");
	return data[i * shape[1] + j];
}

float Tensor::At2(int i, int j) const {
	CheckRank(*this, 2, "At2");
	CheckAxisIndex(*this, 0, i, "At2");
	CheckAxisIndex(*this, 1, j, "At2");
	return data[i * shape[1] + j];
}

// ---------------------------------------------------------------------------
// Convenience 3D accessor
// ---------------------------------------------------------------------------
float& Tensor::At3(int i, int j, int k) {
	CheckRank(*this, 3, "At3");
	CheckAxisIndex(*this, 0, i, "At3");
	CheckAxisIndex(*this, 1, j, "At3");
	CheckAxisIndex(*this, 2, k, "At3");
	return data[(i * shape[1] + j) * shape[2] + k];
}

float Tensor::At3(int i, int j, int k) const {
	CheckRank(*this, 3, "At3");
	CheckAxisIndex(*this, 0, i, "At3");
	CheckAxisIndex(*this, 1, j, "At3");
	CheckAxisIndex(*this, 2, k, "At3");
	return data[(i * shape[1] + j) * shape[2] + k];
}

// ---------------------------------------------------------------------------
// Convenience 4D accessor
// ---------------------------------------------------------------------------
float& Tensor::At4(int i, int j, int k, int l) {
	CheckRank(*this, 4, "At4");
	CheckAxisIndex(*this, 0, i, "At4");
	CheckAxisIndex(*this, 1, j, "At4");
	CheckAxisIndex(*this, 2, k, "At4");
	CheckAxisIndex(*this, 3, l, "At4");
	return data[((i * shape[1] + j) * shape[2] + k) * shape[3] + l];
}

float Tensor::At4(int i, int j, int k, int l) const {
	CheckRank(*this, 4, "At4");
	CheckAxisIndex(*this, 0, i, "At4");
	CheckAxisIndex(*this, 1, j, "At4");
	CheckAxisIndex(*this, 2, k, "At4");
	CheckAxisIndex(*this, 3, l, "At4");
	return data[((i * shape[1] + j) * shape[2] + k) * shape[3] + l];
}

// ---------------------------------------------------------------------------
// Row pointer for 2D tensors
// ---------------------------------------------------------------------------
float* Tensor::RowPtr(int row) {
	CheckRank(*this, 2, "row_ptr");
	CheckAxisIndex(*this, 0, row, "row_ptr");
	return data.data() + row * shape[1];
}

const float* Tensor::RowPtr(int row) const {
	CheckRank(*this, 2, "row_ptr");
	CheckAxisIndex(*this, 0, row, "row_ptr");
	return data.data() + row * shape[1];
}

// ---------------------------------------------------------------------------
// Shape assertion
// ---------------------------------------------------------------------------
void Tensor::AssertShape(const std::vector<int>& expected,
                         const char* caller) const {
	if (shape != expected) {
		throw std::runtime_error(
		    std::string(CallerName(caller)) + ": shape mismatch. expected " +
		    ShapeToString(expected) + ", got " + ShapeStringShort());
	}
}

// ---------------------------------------------------------------------------
// Reshape with element count validation
// ---------------------------------------------------------------------------
Tensor Tensor::ReshapeChecked(const std::vector<int>& new_shape,
                              const char* caller) const {
	// 改变shape形状，但是总元素个数要能对上
	const size_t new_total = CheckedNumel(new_shape, CallerName(caller));
	if (new_total != data.size()) {
		throw std::runtime_error(
		    std::string(CallerName(caller)) + ": cannot reshape tensor with " +
		    std::to_string(data.size()) + " elements into shape with " +
		    std::to_string(new_total) + " elements");
	}
	Tensor r = *this;
	r.shape = new_shape;
	return r;
}

std::string Tensor::ShapeStringShort() const { return ShapeToString(shape); }

void Tensor::Print(const std::string& name, bool print_data) const {
	if (!name.empty()) {
		std::cout << name << " ";
	}
	// 打印维度和元素个数信息
	std::cout << "shape=" << ShapeStringShort() << " size=" << size()
	          << std::endl;
	// 是否打印数据
	if (print_data) {
		for (size_t i = 0; i < data.size(); ++i) {
			std::cout << std::fixed << std::setprecision(6) << data[i];
			if (i + 1 < data.size()) {
				std::cout << " ";
			}
		}
		std::cout << std::endl;
	}
}

// 创建多维张量，使用fill填充
Tensor MakeTensor1D(int d0, float fill) { return Tensor({d0}, fill); }

Tensor MakeTensor2D(int d0, int d1, float fill) {
	return Tensor({d0, d1}, fill);
}

Tensor MakeTensor3D(int d0, int d1, int d2, float fill) {
	return Tensor({d0, d1, d2}, fill);
}

Tensor MakeTensor4D(int d0, int d1, int d2, int d3, float fill) {
	return Tensor({d0, d1, d2, d3}, fill);
}

}  // namespace mini_llama
