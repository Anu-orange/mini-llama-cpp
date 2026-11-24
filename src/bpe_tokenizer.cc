// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

// Primary header for the BPE tokenizer implementation.
// clang-format off
#include "mini_llama/tokenizer.h"
// clang-format on

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace mini_llama {

// ---------------------------------------------------------------------------
// UTF-8 helpers
// ---------------------------------------------------------------------------

static std::string CodepointToUtf8(char32_t cp) {
	std::string result;
	if (cp <= 0x7F) {
		result += static_cast<char>(cp);
	} else if (cp <= 0x7FF) {
		result += static_cast<char>(0xC0 | (cp >> 6));
		result += static_cast<char>(0x80 | (cp & 0x3F));
	} else if (cp <= 0xFFFF) {
		result += static_cast<char>(0xE0 | (cp >> 12));
		result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
		result += static_cast<char>(0x80 | (cp & 0x3F));
	} else {
		result += static_cast<char>(0xF0 | (cp >> 18));
		result += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
		result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
		result += static_cast<char>(0x80 | (cp & 0x3F));
	}
	return result;
}

// Extract next UTF-8 codepoint as a string; return empty if invalid
static std::string NextUtf8Codepoint(const std::string& s, size_t& pos) {
	if (pos >= s.size()) {
		return "";
	}
	unsigned char first = s[pos];
	size_t len = 1;
	if ((first & 0x80) == 0) {
		len = 1;
	} else if ((first & 0xE0) == 0xC0) {
		len = 2;
	} else if ((first & 0xF0) == 0xE0) {
		len = 3;
	} else if ((first & 0xF8) == 0xF0) {
		len = 4;
	} else {
		// Invalid UTF-8, skip one byte
		++pos;
		return "";
	}
	if (pos + len > s.size()) {
		++pos;
		return "";
	}
	std::string result = s.substr(pos, len);
	pos += len;
	return result;
}

// ---------------------------------------------------------------------------
// GPT-2 bytes_to_unicode mapping
// ---------------------------------------------------------------------------

static std::vector<std::string> BuildBytesToUnicode() {
	std::vector<std::string> map(256);

	std::vector<int> bs;

	// 厦下面这三种循环的字符都可以直接保存
	for (int c = '!'; c <= '~'; ++c) {
		bs.push_back(c);
	}
	for (int c = 0xA1; c <= 0xAC; ++c) {
		bs.push_back(c);
	}
	for (int c = 0xAE; c <= 0xFF; ++c) {
		bs.push_back(c);
	}

	std::vector<int> cs = bs;
	int n = 0;
	for (int b = 0; b < 256; ++b) {
		if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
			// bs存储原始字节
			bs.push_back(b);
			// cs存储目标码点
			cs.push_back(256 + n);
			++n;
		}
	}

	for (size_t i = 0; i < bs.size(); ++i) {
		// 把目标码点编码成uft-8存到map中，map的键还是这个字节的原始字节数
		// 然后对应的值则是编码后的的unicode
		map[bs[i]] = CodepointToUtf8(static_cast<char32_t>(cs[i]));
	}
	return map;
}

// ---------------------------------------------------------------------------
// Simple JSON number parser helper
// ---------------------------------------------------------------------------

static bool ParseJsonInt(const std::string& s, size_t& pos, int& out) {
	while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) {
		++pos;
	}
	if (pos >= s.size()) {
		return false;
	}
	size_t start = pos;
	if (s[pos] == '-') {
		++pos;
	}
	while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) {
		++pos;
	}
	if (start == pos) {
		return false;
	}
	try {
		out = std::stoi(s.substr(start, pos - start));
		return true;
	} catch (...) {
		return false;
	}
}

static bool ParseJsonString(const std::string& s, size_t& pos,
                            std::string& out) {
	while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) {
		++pos;
	}
	if (pos >= s.size() || s[pos] != '"') {
		return false;
	}
	++pos;  // skip opening quote
	out.clear();
	while (pos < s.size() && s[pos] != '"') {
		if (s[pos] == '\\' && pos + 1 < s.size()) {
			char esc = s[pos + 1];
			switch (esc) {
				case '"':
					out += '"';
					break;
				case '\\':
					out += '\\';
					break;
				case '/':
					out += '/';
					break;
				case 'b':
					out += '\b';
					break;
				case 'f':
					out += '\f';
					break;
				case 'n':
					out += '\n';
					break;
				case 'r':
					out += '\r';
					break;
				case 't':
					out += '\t';
					break;
				case 'u': {
					if (pos + 5 < s.size()) {
						std::string hex = s.substr(pos + 2, 4);
						try {
							char32_t cp = static_cast<char32_t>(
							    std::stoul(hex, nullptr, 16));
							out += CodepointToUtf8(cp);
						} catch (...) {
							out += '?';
						}
						pos += 4;
					}
					break;
				}
				default:
					out += esc;
					break;
			}
			pos += 2;
		} else {
			out += s[pos];
			++pos;
		}
	}
	if (pos < s.size() && s[pos] == '"') {
		++pos;
	}
	return true;
}

// ---------------------------------------------------------------------------
// BpeTokenizer
// ---------------------------------------------------------------------------

void BpeTokenizer::BuildByteMappings() {
	b2u_ = BuildBytesToUnicode();
	u2b_.clear();
	// 利用for循环反向构建unicode到字节的
	for (int b = 0; b < 256; ++b) {
		u2b_[b2u_[b]] = static_cast<uint8_t>(b);
	}
}

bool BpeTokenizer::Load(const std::string& vocab_path,
                        const std::string& merges_path,
                        const std::string& special_path) {
	BuildByteMappings();

	// Load vocab.json: { "token": id, ... }
	{
		std::ifstream f(vocab_path);
		if (!f.is_open()) {
			std::cerr << "Failed to open vocab: " << vocab_path << std::endl;
			return false;
		}
		std::stringstream buf;
		buf << f.rdbuf();
		std::string content = buf.str();

		size_t pos = 0;
		while (pos < content.size() &&
		       std::isspace(static_cast<unsigned char>(content[pos]))) {
			++pos;
		}
		if (pos < content.size() && content[pos] == '{') {
			++pos;
		}
		while (pos < content.size()) {
			while (pos < content.size() &&
			       std::isspace(static_cast<unsigned char>(content[pos]))) {
				++pos;
			}
			if (pos < content.size() && content[pos] == '}') {
				break;
			}
			std::string key;
			if (!ParseJsonString(content, pos, key)) {
				break;
			}
			while (pos < content.size() &&
			       std::isspace(static_cast<unsigned char>(content[pos]))) {
				++pos;
			}
			if (pos >= content.size() || content[pos] != ':') {
				break;
			}
			++pos;
			int val = 0;
			if (!ParseJsonInt(content, pos, val)) {
				break;
			}
			vocab_[key] = val;
			while (pos < content.size() &&
			       std::isspace(static_cast<unsigned char>(content[pos]))) {
				++pos;
			}
			if (pos < content.size() && content[pos] == ',') {
				++pos;
			}
		}

		if (vocab_.empty()) {
			std::cerr << "Vocab is empty" << std::endl;
			return false;
		}

		int max_id = 0;
		for (const auto& kv : vocab_) {
			if (kv.second > max_id) {
				max_id = kv.second;
			}
		}
		id_to_token_.resize(max_id + 1);
		for (const auto& kv : vocab_) {
			id_to_token_[kv.second] = kv.first;
		}

		// Extract special tokens (e.g. <|im_start|>) from vocab for
		// longest-match Encode.
		for (const auto& kv : vocab_) {
			if (kv.first.size() >= 4 && kv.first.front() == '<' &&
			    kv.first[1] == '|' && kv.first[kv.first.size() - 2] == '|' &&
			    kv.first.back() == '>') {
				special_tokens_.push_back({kv.first, kv.second});
			}
		}
		std::sort(special_tokens_.begin(), special_tokens_.end(),
		          [](const auto& a, const auto& b) {
			          return a.first.size() > b.first.size();
		          });
	}

	// Load merges.txt: "token1 token2" per line
	{
		std::ifstream f(merges_path);
		if (!f.is_open()) {
			std::cerr << "Failed to open merges: " << merges_path << std::endl;
			return false;
		}
		std::string line;
		int rank = 0;
		while (std::getline(f, line)) {
			std::stringstream ss(line);
			std::string a, b;
			if (ss >> a >> b) {
				merge_ranks_[{a, b}] = rank++;
			}
		}
	}

	// Load special_tokens.json
	{
		std::ifstream f(special_path);
		if (f.is_open()) {
			std::stringstream buf;
			buf << f.rdbuf();
			std::string content = buf.str();
			auto find_int = [&](const std::string& key) -> int {
				size_t p = content.find('"' + key + '"');
				if (p == std::string::npos) {
					return -1;
				}
				p = content.find(':', p);
				if (p == std::string::npos) {
					return -1;
				}
				size_t start = p + 1;
				while (
				    start < content.size() &&
				    std::isspace(static_cast<unsigned char>(content[start]))) {
					++start;
				}
				size_t end = start;
				while (
				    end < content.size() &&
				    (std::isdigit(static_cast<unsigned char>(content[end])) ||
				     content[end] == '-')) {
					++end;
				}
				if (start == end) {
					return -1;
				}
				try {
					return std::stoi(content.substr(start, end - start));
				} catch (...) {
					return -1;
				}
			};
			bos_id_ = find_int("bos_id");
			eos_id_ = find_int("eos_id");
			unk_id_ = find_int("unk_id");
		}
	}

	std::cout << "BPE tokenizer loaded: vocab=" << vocab_.size()
	          << " merges=" << merge_ranks_.size() << " bos=" << bos_id_
	          << " eos=" << eos_id_ << " unk=" << unk_id_ << std::endl;
	return true;
}

/**
 * @brief 将一个字符串编码形成token id列表的过程
 * 
 * @param text 
 * @return std::vector<int> 
 */
std::vector<int> BpeTokenizer::Encode(const std::string& text) const {
	if (text.empty()) {
		return {};
	}

	std::vector<int> result;
	size_t pos = 0;

	while (pos < text.size()) {
		// Try to match a special token first (longest match)
		bool matched = false;
		// 1、拿到一段文本最开始的时候从pos为0的位置开始，先尝试看是否能匹配到特殊token
		for (const auto& st : special_tokens_) {
			// 遍历每个特殊token，然后取出它
			const std::string& token_str = st.first;
			// 如果起始位置前面一段字节和特殊token完全相等
			// 那说明找到了特殊token
			if (pos + token_str.size() <= text.size() &&
			    std::memcmp(text.data() + pos, token_str.data(),
			                token_str.size()) == 0) {
				// 将特殊token的id加到result数组中
				result.push_back(st.second);
				// 然后遍历位置pos跳到特殊token尾部的后一位上
				pos += token_str.size();
				// 标记特殊token匹配上
				matched = true;
				break;
			}
		}
		if (matched) {
			// 如果说要是匹配上了，就进入下一轮循环
			continue;
		}

		// Find the next special token or end of string
		// 下面这个循环主要是找后面的特殊token，看普通文本在哪里结束
		// 如果前面匹配上了特殊token，那么此时的pos是普通文本的起始位置，就从这里找
		// 如果前面没匹配上特殊token，那么此时的pos同样是普通文本的起始位置
		size_t end = text.size();
		for (const auto& st : special_tokens_) {
			// 在text中从pos位置开始找st.first
			size_t p = text.find(st.first, pos);
			if (p != std::string::npos && p < end) {
				end = p;
			}
		}

		// Encode the plain text segment [pos, end) with BPE

		// 把pos到end这中间的普通文本给切出来进行BPE merge操作
		std::string segment = text.substr(pos, end - pos);
		pos = end;

		// 2、 转换原始字节到unicode字符
		std::vector<std::string> word;
		for (unsigned char c : segment) {
			if (c >= b2u_.size()) {
				std::cerr << "ERROR: byte " << static_cast<int>(c)
				          << " >= b2u_ size " << b2u_.size() << "\n";
				return {};
			}
			word.push_back(b2u_[c]);
		}

		// 3、根据word中的单个字符利用加载的merge等级表进行merge合并
		// merge操作的开始
		while (word.size() > 1) {
			// 获取int最大值作为初始优先级，这个rank越小，优先级越高
			int best_rank = std::numeric_limits<int>::max();
			size_t best_idx = word.size();  // invalid
			for (size_t i = 0; i + 1 < word.size(); ++i) {
				// 从我们的合并优先级map中看是否能找到当前尝试合并的两个字符串
				auto it = merge_ranks_.find({word[i], word[i + 1]});
				if (it != merge_ranks_.end() && it->second < best_rank) {
					// 如果存在且优先级值要小于best_rank
					// 那么就更新最优的优先级以及合并的位置索引
					best_rank = it->second;
					// 把最优的索引位置保存下来
					best_idx = i;
				}
			}

			if (best_idx >= word.size()) {
				// 如果说best_idx等于word.size的话，那么与之要合并的best_idx+1
				// 就超过了word.size，这是不符合下标的
				break;  // no more merges possible
			}
			// 到这里说明已经找到了最好的合并位置，此时将best_idx 和 best_idx + 1位置上的两个字符串进行合并
			word[best_idx] += word[best_idx + 1];
			// 然后把后面那个字符串删除
			word.erase(word.begin() + best_idx + 1);
		}

		// 3、根据合并完后的word词表，利用加载的vocab词表进行映射，将token变成id
		for (const auto& w : word) {
			// 遍历每个字符串，从词表中找对应的word，然后获取里面的token_id
			auto it = vocab_.find(w);
			if (it != vocab_.end()) {
				result.push_back(it->second);
			} else {
				result.push_back(unk_id_);
			}
		}
	}
	// 返回最后的result，这个result记录的就是我们进行分词后的token对应的token id
	// 接下来就是拿着这个id去找embedding表获取对应的向量了
	return result;
}

std::string BpeTokenizer::DecodeToken(int token) const {
	if (token >= 0 && token < static_cast<int>(id_to_token_.size())) {
		return id_to_token_[token];
	}
	return "";
}

std::string BpeTokenizer::Decode(const std::vector<int>& tokens) const {
	// 1. Concatenate token strings
	std::string text;
	for (int id : tokens) {
		if (id >= 0 && id < static_cast<int>(id_to_token_.size())) {
			text += id_to_token_[id];
		}
	}

	// 2. Convert unicode chars back to bytes
	std::string result;
	size_t pos = 0;
	while (pos < text.size()) {
		size_t start = pos;
		std::string cp = NextUtf8Codepoint(text, pos);
		if (cp.empty()) {
			// Invalid UTF-8, skip
			if (pos == start) {
				++pos;
			}
			continue;
		}
		auto it = u2b_.find(cp);
		if (it != u2b_.end()) {
			result += static_cast<char>(it->second);
		} else {
			// Not a byte token; pass through as-is (shouldn't happen for
			// standard BPE)
			result += cp;
		}
	}

	return result;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<ITokenizer> CreateBpeTokenizer(
    const std::string& vocab_path, const std::string& merges_path,
    const std::string& special_path) {
	auto tok = std::make_unique<BpeTokenizer>();
	if (tok->Load(vocab_path, merges_path, special_path)) {
		return tok;
	}
	return nullptr;
}

}  // namespace mini_llama
