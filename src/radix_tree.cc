// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/radix_tree.h"

#include <algorithm>
#include <utility>

namespace mini_llama {

RadixTree::RadixTree() = default;
RadixTree::~RadixTree() = default;
RadixTree::RadixTree(RadixTree&&) noexcept = default;
RadixTree& RadixTree::operator=(RadixTree&&) noexcept = default;

void RadixTree::Clear() {
	root_.children.clear();
	root_.terminal = false;
	root_.key.clear();
	terminal_count_ = 0;
}

bool RadixTree::empty() const { return terminal_count_ == 0; }

size_t RadixTree::size() const { return terminal_count_; }

size_t RadixTree::CommonPrefixLength(const std::vector<int>& a, size_t a_offset,
                                     const std::vector<int>& b) {
	size_t i = 0;
	while (a_offset + i < a.size() && i < b.size() && a[a_offset + i] == b[i]) {
		++i;
	}
	return i;
}

void RadixTree::Insert(const std::vector<int>& tokens) {
	if (tokens.empty()) {
		if (!root_.terminal) {
			root_.terminal = true;
			++terminal_count_;
		}
		return;
	}
	InsertInto(&root_, tokens);
}

/**
 * @brief 往压缩前缀树中插入数据
 * 该插入算法很重要，是压缩前缀树的巧妙操作，一个节点保存了不止一个token，而是一个序列
 * @param parent 
 * @param suffix 
 */
void RadixTree::InsertInto(Node* parent, std::vector<int> suffix) {
	// 最开始只有root节点，从root的孩子中寻找suffix中索引为0的元素
	auto child_it = parent->children.find(suffix[0]);
	if (child_it == parent->children.end()) {
		// 如果找不到，此时就创建一个Node
		auto child = std::make_unique<Node>();
		// 把整个suffix给key
		child->key = std::move(suffix);
		child->terminal = true;
		// 表示以child->key[0]开头的序列作为一个节点存到root下面
		parent->children[child->key[0]] = std::move(child);
		++terminal_count_;
		return;
	}

	// 说明找到了
	std::unique_ptr<Node>& child = child_it->second;
	// 找出公共部分的长度
	size_t common = CommonPrefixLength(suffix, 0, child->key);
	if (common == child->key.size()) {
		// 这里表示当前待插入的序列与压缩树中的某个公共部分已经在树中成为一个单独的节点
		// 或者说是插入相同的序列进去
		if (common == suffix.size()) {
			// 且公共部分和待插入序列长度又相同，说明不用再插入了
			if (!child->terminal) {
				child->terminal = true;
				++terminal_count_;
			}
			return;
		}
		// 把当前输入的suffix除开公共部分以外剩余的序列拿出来构造一个新的vector(新序列还有剩余)
		std::vector<int> remaining(suffix.begin() + static_cast<long>(common),
		                           suffix.end());
		// 然后以child作为新的根，递归调用寻找新的子节点插入
		InsertInto(child.get(), std::move(remaining));
		return;
	}

	// 先创建出一个公共节点出来
	auto split = std::make_unique<Node>();
	// 调用assign函数使用另一个vector的迭代器区间来给当前节点进行赋值
	split->key.assign(child->key.begin(),
	                  child->key.begin() + static_cast<long>(common));

	// 从child中删除已被复制的公共节点
	child->key.erase(child->key.begin(),
	                 child->key.begin() + static_cast<long>(common));
	// 此时就是删除元素后第一个元素的值
	int old_child_first = child->key[0];
	// 给split记录新的分支，split是公共部分
	split->children[old_child_first] = std::move(child);

	if (common == suffix.size()) {
		// 这种情况表示插入序列在公共部分就结束了
		split->terminal = true;
		++terminal_count_;
	} else {
		// 创建一个新的节点用来保存新插入序列中除了公共部分之外剩余的token序列
		auto new_child = std::make_unique<Node>();
		// 把剩余的序列放到key中
		new_child->key.assign(suffix.begin() + static_cast<long>(common),
		                      suffix.end());
		new_child->terminal = true;
		// 把这个新节点添加到公共节点的下面
		split->children[new_child->key[0]] = std::move(new_child);
		++terminal_count_;
	}
	// 然后更新根节点的子节点
	child_it->second = std::move(split);
}

/**
 * @brief 查询时返回最长命中长度
 * 
 * @param tokens 
 * @return size_t 
 */
size_t RadixTree::LongestPrefix(const std::vector<int>& tokens) const {
	if (tokens.empty()) {
		return 0;
	}

	const Node* node = &root_;
	size_t pos = 0;
	while (pos < tokens.size()) {
		auto child_it = node->children.find(tokens[pos]);
		if (child_it == node->children.end()) {
			// 说明根节点下面只有这一个节点，直接返回pos
			return pos;
		}

		const Node* child = child_it->second.get();
		size_t common = CommonPrefixLength(tokens, pos, child->key);
		pos += common;
		if (common < child->key.size()) {
			// 说明匹配到公共部分，且child有剩余，那么接下来的部分就不属于公共部分了
			return pos;
		}
		// 说明公共部分就是一个独立的节点，此时需要从这个独立的节点开始继续往下找
		node = child;
	}
	return pos;
}

}  // namespace mini_llama
