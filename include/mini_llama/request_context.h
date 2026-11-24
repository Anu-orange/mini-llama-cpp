// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_MINI_LLAMA_REQUEST_CONTEXT_H_
#define INCLUDE_MINI_LLAMA_REQUEST_CONTEXT_H_

#include <chrono>
#include <string>
#include <vector>

namespace mini_llama {

using RequestClock = std::chrono::steady_clock;

struct TraceEvent {
	std::string stage; // 阶段过程名
	double elapsed_ms = 0.0; // 耗费时间
	int tokens = 0; // 处理token数
	std::string detail; // 详情
};

// Request-level context for tracing one user request from entry to finish.
struct RequestContext {
	// 这部分是请求身份和运行配置
	std::string trace_id; // 唯一的请求id
	std::string mode; // 本次的运行模式，是run还是generate
	std::string backend; // 执行的后端，是cpu还是gpu
	std::string model_path; // 模型路径

	RequestClock::time_point start_time; // 请求开始时间
	double total_ms = 0.0; // 请求总耗时

	// 这部分是token记录
	int prompt_tokens = 0; // 本轮请求完整提示词token数
	int prefill_tokens = 0; // 本轮实际执行prefill的token数
	int decode_tokens = 0; // 本轮在生成过程中，实际送入前向计算的token数
	int generated_tokens = 0; // 本轮采样生成的token数

	// 各阶段耗时，单位是ms
	double model_load_ms = 0.0; // 模型及分词器加载阶段耗时
	double tokenize_ms = 0.0; // 将完整prompt编码为token id的耗时
	double prefill_ms = 0.0; // 本轮需要计算的prompt token执行前向的耗时
	double decode_ms = 0.0; // 生成过程中各次decode前向的累计耗时
	double sample_ms = 0.0; // 从logits中采样token的累计耗时

	std::string error; // 保存本次请求的错误信息
	std::vector<TraceEvent> events; // events记录明细

	void Start();
	void Finish();
	void RecordEvent(const std::string& stage, double elapsed_ms,
	                 int tokens = 0, const std::string& detail = "");
	void SetError(const std::string& message);
	bool ok() const;
};

std::string NewTraceId();
double ElapsedMs(RequestClock::time_point start);
RequestContext StartRequest(const std::string& mode, const std::string& backend,
                            const std::string& model_path);
std::string FormatRequestTraceSummary(const RequestContext& request);
std::vector<std::string> FormatRequestTraceEvents(
    const RequestContext& request);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_REQUEST_CONTEXT_H_
