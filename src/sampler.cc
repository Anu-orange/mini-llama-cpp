// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/sampler.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mini_llama/ops.h"

namespace mini_llama {

namespace {

constexpr float kGreedyTemperature = 1e-6f;

void ValidateLogits(const Tensor& logits) {
	if (logits.size() == 0) {
		throw std::runtime_error("sampler: cannot sample from empty logits");
	}
	for (float value : logits.data) {
		if (!std::isfinite(value)) {
			// 判断是否为有限小数
			throw std::runtime_error("sampler: logits must be finite");
		}
	}
}

void ValidateTemperature(float temperature) {
	if (!std::isfinite(temperature) || temperature < 0.0f) {
		throw std::runtime_error(
		    "sampler: temperature must be finite and non-negative");
	}
}

void ValidateTopK(int top_k) {
	if (top_k < 0) {
		throw std::runtime_error("sampler: top_k must be non-negative");
	}
}

/**
 * @brief 概率累计采样算法
 * 原理是：将候选范围的分数进行归一化之后使得它们加起来为1，然后在[0,1)中选取一个随机数
 * 归一化后的数组中从左到右进行分数值累加(归一化后的)，当累加值达到这个随机数时，就取最后一个加进来的
 * 那个分数在候选集中的下标索引
 * 
 * @param candidates 
 * @param temperature 
 * @param rng 
 * @return int 
 */
int SampleFromCandidates(const std::vector<std::pair<float, int>>& candidates,
                         float temperature, std::mt19937& rng) {
	if (temperature < kGreedyTemperature) {
		int best = candidates[0].second;
		float best_val = candidates[0].first;
		for (size_t i = 1; i < candidates.size(); ++i) {
			if (candidates[i].first > best_val) {
				best = candidates[i].second;
				best_val = candidates[i].first;
			}
		}
		return best;
	}
	// 获取候选分集合中最大分数的那个
	float max_val = candidates[0].first;
	for (size_t i = 1; i < candidates.size(); ++i) {
		if (candidates[i].first > max_val) {
			max_val = candidates[i].first;
		}
	}

	// 使用一个数组用来记录每个分数除以温度后的值
	std::vector<float> probs(candidates.size(), 0.0f);
	float sum = 0.0f;
	for (size_t i = 0; i < candidates.size(); ++i) {
		probs[i] = std::exp((candidates[i].first - max_val) / temperature);
		// 累加起来作为sum
		sum += probs[i];
	}

	std::uniform_real_distribution<float> dist(0.0f, 1.0f);
	// 随机抽取[0.1)之间一个小数
	float r = dist(rng);
	float cumulative = 0.0f;
	for (size_t i = 0; i < probs.size(); ++i) {
		// 然后用除以温度的分数
		cumulative += probs[i] / sum;
		if (r <= cumulative) {
			// 只要累加起来的cmulative大于等于了r，那就选择当前新算进来的这个(这里有点top-p的意思)
			return candidates[i].second;
		}
	}
	// 作为最终兜底，防止float计算时可能最终的和小于1，理论上是等于1的，但是float有精度限制
	return candidates.back().second;
}

}  // namespace

MiniSampler::MiniSampler(unsigned int seed) {
	if (seed != 0) {
		rng_.seed(seed);
	} else {
		std::random_device rd;
		rng_.seed(rd());
	}
}

MiniSampler::MiniSampler(const SamplingParams& params)
    : MiniSampler(params.seed) {}

/**
 * @brief 贪婪采样，每次都选择得分最高的那个
 * 
 * @param logits 
 * @return int 
 */
int MiniSampler::SampleGreedy(const Tensor& logits) {
	ValidateLogits(logits);
	return ArgMax(logits);
}

/**
 * @brief 加入温度系数，对得到的分数进行计算
 * 
 * @param logits 
 * @param temperature 
 * @return int 
 */
int MiniSampler::SampleTemperature(const Tensor& logits, float temperature) {
	ValidateLogits(logits);
	ValidateTemperature(temperature);
	if (temperature < kGreedyTemperature) {
		// 温度过低时，退回greedy
		return SampleGreedy(logits);
	}

	std::vector<std::pair<float, int>> candidates;
	candidates.reserve(logits.data.size());
	for (size_t i = 0; i < logits.data.size(); ++i) {
		candidates.emplace_back(logits.data[i], static_cast<int>(i));
	}
	return SampleFromCandidates(candidates, temperature, rng_);
}

/**
 * @brief 选取得分最高的前k个
 * 
 * @param logits 
 * @param temperature 
 * @param top_k 
 * @return int 
 */
int MiniSampler::SampleTopK(const Tensor& logits, float temperature,
                            int top_k) {
	ValidateLogits(logits);
	ValidateTemperature(temperature);
	ValidateTopK(top_k);
	if (top_k == 0) {
		throw std::runtime_error(
		    "sampler: top_k must be positive when calling SampleTopK");
	}
	if (top_k == 1) {
		return SampleGreedy(logits);
	}
	if (top_k > static_cast<int>(logits.data.size())) {
		top_k = static_cast<int>(logits.data.size());
	}

	std::vector<std::pair<float, int>> candidates;
	candidates.reserve(logits.data.size());
	for (size_t i = 0; i < logits.data.size(); ++i) {
		candidates.emplace_back(logits.data[i], static_cast<int>(i));
	}
	std::sort(candidates.begin(), candidates.end(),
	          [](const auto& a, const auto& b) {
		          if (a.first == b.first) {
					// 如果得分一样，那么谁的id小就是水
			          return a.second < b.second;
		          }
				  // 否则就按照分数从大到小进行排序
		          return a.first > b.first;
	          });
	// 将前top-k个候选传给随机概率采样函数进行采样
	candidates.resize(static_cast<size_t>(top_k));
	return SampleFromCandidates(candidates, temperature, rng_);
}

int MiniSampler::Sample(const Tensor& logits, const SamplingParams& params) {
	ValidateTemperature(params.temperature);
	ValidateTopK(params.top_k);
	if (params.temperature < kGreedyTemperature || params.top_k == 1) {
		return SampleGreedy(logits);
	}
	if (params.top_k > 1) {
		return SampleTopK(logits, params.temperature, params.top_k);
	}
	return SampleTemperature(logits, params.temperature);
}

// Legacy free function
int SampleGreedy(const Tensor& logits) {
	return MiniSampler::SampleGreedy(logits);
}

}  // namespace mini_llama
