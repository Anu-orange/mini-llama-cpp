// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_RADIX_TREE_H_
#define INCLUDE_MINI_LLAMA_RADIX_TREE_H_

#include <map>
#include <memory>
#include <vector>

namespace mini_llama {

// Compressed prefix tree for token sequences.
/**
 * @brief 前缀树接口定义
 * 
 */
class RadixTree {
   public:
	RadixTree();
	~RadixTree();
	RadixTree(const RadixTree&) = delete; // 禁止拷贝构造
	RadixTree& operator=(const RadixTree&) = delete; // 禁止拷贝赋值
	RadixTree(RadixTree&&) noexcept; // 允许移动构造
	RadixTree& operator=(RadixTree&&) noexcept; // 允许移动赋值 

	void Clear();
	bool empty() const;
	size_t size() const;

	void Insert(const std::vector<int>& tokens);
	size_t LongestPrefix(const std::vector<int>& tokens) const;

   private:
	struct Node {
		std::vector<int> key; // 记录到当前token的整个token序列的id
		bool terminal = false; // 是否到达序列的尾部
		std::map<int, std::unique_ptr<Node>> children; // 当前Node的孩子节点，int表示的是token_id
	};

	static size_t CommonPrefixLength(const std::vector<int>& a, size_t a_offset,
	                                 const std::vector<int>& b);
	void InsertInto(Node* parent, std::vector<int> suffix);

	Node root_;
	size_t terminal_count_ = 0;
};

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_RADIX_TREE_H_
