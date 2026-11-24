// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_CHAT_H_
#define INCLUDE_MINI_LLAMA_CHAT_H_

#include <string>
#include <vector>

#include "mini_llama/context.h"
#include "mini_llama/radix_tree.h"
#include "mini_llama/sampler.h"

namespace mini_llama {

/**
 * @brief 该结构体很重要，表示谁说了什么内容
 * 
 */
struct ChatMessage {
	std::string role;  // 角色
	std::string content; // 输出内容
};

// Manages the state of an interactive chat session.
class ChatSession {
   public:
	std::vector<ChatMessage> messages; // 带有角色信息的对话历史，就类似于张三一句、李四一句、然后张三再说一句。。。
	std::vector<int> token_history; // 记录kv cache中已经计算过的扁平token id
	RadixTree prefix_cache; // 压缩前缀树缓存
	SamplingParams sampling_params; // 采样参数

	// Stats
	int total_prompt_tokens = 0;
	int total_generated_tokens = 0;
	double total_time_ms = 0.0;

	ChatSession() = default;

	// Reset messages and stats.
	void Clear();

	// Add a message to the history.
	void AddMessage(const std::string& role, const std::string& content);

	// Increment stats after a turn.
	void RecordTurn(int prompt_tokens, int generated_tokens, double time_ms);

	// Replace the current token history for the latest evaluated context.
	void SetTokenHistory(const std::vector<int>& tokens);

	// Append one generated token to the tracked context.
	void AppendToken(int token);

	// Record a token sequence whose prefix has been evaluated.
	void RecordPrefix(const std::vector<int>& tokens);

	// Return the longest cached prefix for a token sequence.
	size_t LongestCachedPrefix(const std::vector<int>& tokens) const;
};

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_CHAT_H_
