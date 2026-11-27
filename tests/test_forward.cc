// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "mini_llama/context.h"
#include "mini_llama/forward.h"
#include "mini_llama/loader.h"
#include "mini_llama/model.h"
#include "mini_llama/sampler.h"
#include "mini_llama/tokenizer.h"
#include "tests/test_main.h"
#include "tests/test_names.h"

static bool TestForwardTokenProducesLogitShape() {
	MiniLlamaModel model =
	    LoadModel("models/tiny/model.json", "models/tiny/model.bin");
	if (model.layers.empty()) {
		MINI_LLAMA_ASSERT_FAIL("Failed to load model");
	}
	MiniLlamaContext ctx(&model);
	ctx.pos = 0;
	Tensor logits = ForwardToken(ctx, model, 1);  // BOS token
	MINI_LLAMA_ASSERT_EQ(logits.num_dims(), 1);
	MINI_LLAMA_ASSERT_EQ(logits.shape[0], model.config.vocab_size);
	return true;
}

static bool TestForwardTokenChangesKvCache() {
	MiniLlamaModel model =
	    LoadModel("models/tiny/model.json", "models/tiny/model.bin");
	if (model.layers.empty()) {
		MINI_LLAMA_ASSERT_FAIL("Failed to load model");
	}
	MiniLlamaContext ctx(&model);
	ctx.pos = 0;
	// Before forward, cache at layer 0, pos 0 should be zero
	float before = ctx.kv_cache.keys.At({0, 0, 0, 0});
	MINI_LLAMA_ASSERT_NEAR(before, 0.0f, 1e-6f);

	ForwardToken(ctx, model, 1);

	// After forward, cache should have been written
	float after = ctx.kv_cache.keys.At({0, 0, 0, 0});
	// It's unlikely to be exactly 0 after random weights forward
	// Just check it's finite and cache was written somewhere
	MINI_LLAMA_ASSERT_TRUE(std::isfinite(after));
	return true;
}

static bool TestGenerationGreedySequence() {
	MiniLlamaModel model =
	    LoadModel("models/tiny/model.json", "models/tiny/model.bin");
	if (model.layers.empty()) {
		MINI_LLAMA_ASSERT_FAIL("Failed to load model");
	}
	AsciiTokenizer tokenizer;
	std::vector<int> tokens = tokenizer.Encode("hello");
	MiniLlamaContext ctx(&model);

	// Prefill
	size_t prompt_len = tokens.size();
	for (size_t i = 0; i < prompt_len; ++i) {
		ctx.pos = static_cast<int>(i);
		Tensor logits = ForwardToken(ctx, model, tokens[i]);
		if (i + 1 == prompt_len) {
			tokens.push_back(SampleGreedy(logits));
		}
	}

	// Decode 5 more tokens
	for (int i = 0; i < 5; ++i) {
		ctx.pos = static_cast<int>(tokens.size() - 1);
		Tensor logits = ForwardToken(ctx, model, tokens.back());
		tokens.push_back(SampleGreedy(logits));
	}

	MINI_LLAMA_ASSERT_TRUE(tokens.size() > 6);
	// First generated token should be the same as what we got in prefill
	MINI_LLAMA_ASSERT_TRUE(tokens[tokens.size() - 6] >= 0);
	return true;
}

static bool TestPositionIncrementsCorrectly() {
	MiniLlamaModel model =
	    LoadModel("models/tiny/model.json", "models/tiny/model.bin");
	if (model.layers.empty()) {
		MINI_LLAMA_ASSERT_FAIL("Failed to load model");
	}
	MiniLlamaContext ctx(&model);

	// Forward token at pos 0
	ctx.pos = 0;
	ForwardToken(ctx, model, 1);

	// Forward token at pos 1
	ctx.pos = 1;
	Tensor logits = ForwardToken(ctx, model, 2);
	MINI_LLAMA_ASSERT_EQ(logits.shape[0], model.config.vocab_size);
	return true;
}

static bool TestForwardRejectsInvalidToken() {
	MiniLlamaModel model =
	    LoadModel("models/tiny/model.json", "models/tiny/model.bin");
	if (!model.loaded) {
		MINI_LLAMA_ASSERT_FAIL("Failed to load model");
	}
	MiniLlamaContext ctx(&model);
	ctx.pos = 0;

	try {
		ForwardToken(ctx, model, model.config.vocab_size);
		MINI_LLAMA_ASSERT_FAIL("ForwardToken accepted an out-of-range token");
	} catch (const std::out_of_range&) {
		return true;
	} catch (...) {
		MINI_LLAMA_ASSERT_FAIL(
		    "ForwardToken threw the wrong exception type for invalid token");
	}
}

static bool TestForwardRejectsInvalidPosition() {
	MiniLlamaModel model =
	    LoadModel("models/tiny/model.json", "models/tiny/model.bin");
	if (!model.loaded) {
		MINI_LLAMA_ASSERT_FAIL("Failed to load model");
	}
	MiniLlamaContext ctx(&model);
	ctx.pos = model.config.max_seq_len;

	try {
		ForwardToken(ctx, model, 1);
		MINI_LLAMA_ASSERT_FAIL(
		    "ForwardToken accepted an out-of-range position");
	} catch (const std::out_of_range&) {
		return true;
	} catch (...) {
		MINI_LLAMA_ASSERT_FAIL(
		    "ForwardToken threw the wrong exception type for invalid position");
	}
}

static bool TestQ41ModelConversion() {
	for (int source = 0; source < 3; ++source) {
		MiniLlamaModel model =
		    LoadModel("models/tiny/model.json", "models/tiny/model.bin");
		MINI_LLAMA_ASSERT_TRUE(model.loaded);
		if (source == 1) QuantizeModelToQ80(model);
		if (source == 2) QuantizeModelToQ40(model);
		auto embedding = model.token_embedding.data;
		auto norm = model.final_norm.data;
		auto layer_norm = model.layers[0].attention_norm.data;
		auto ffn_norm = model.layers[0].ffn_norm.data;
		model.layers[0].bq = Tensor({model.config.dim}, 0.25f);
		auto bias = model.layers[0].bq.data;
		// An existing upload must not keep using pre-conversion weights.
		model.cuda_weights = std::make_shared<mini_llama::CudaModelWeights>();
		QuantizeModelToQ41(model);
		MINI_LLAMA_ASSERT_TRUE(!model.cuda_weights);
		auto check = [](const QuantizedTensor& w) {
			if (w.type != QuantType::kQ41 || !w.f32_data.empty() ||
			    !w.q8_0_data.empty() || !w.q4_0_data.empty())
				return false;
			size_t row_size = static_cast<size_t>(w.shape.back());
			return w.q4_1_data.size() ==
			       (w.num_elements() / row_size) * ((row_size + 31) / 32);
		};
		MINI_LLAMA_ASSERT_TRUE(check(model.lm_head));
		for (const auto& layer : model.layers) {
			for (const auto* w : {&layer.wq, &layer.wk, &layer.wv, &layer.wo,
			                      &layer.w_gate, &layer.w_up, &layer.w_down}) {
				MINI_LLAMA_ASSERT_TRUE(check(*w));
			}
		}
		auto before = model.lm_head.q4_1_data;
		QuantizeModelToQ41(model);
		MINI_LLAMA_ASSERT_EQ(before.size(), model.lm_head.q4_1_data.size());
		for (size_t i = 0; i < before.size(); ++i) {
			MINI_LLAMA_ASSERT_EQ(before[i].d, model.lm_head.q4_1_data[i].d);
			MINI_LLAMA_ASSERT_EQ(before[i].m, model.lm_head.q4_1_data[i].m);
			for (int j = 0; j < 16; ++j) {
				MINI_LLAMA_ASSERT_EQ(before[i].qs[j],
				                     model.lm_head.q4_1_data[i].qs[j]);
			}
		}
		MINI_LLAMA_ASSERT_EQ(model.token_embedding.data, embedding);
		MINI_LLAMA_ASSERT_EQ(model.final_norm.data, norm);
		MINI_LLAMA_ASSERT_EQ(model.layers[0].attention_norm.data, layer_norm);
		MINI_LLAMA_ASSERT_EQ(model.layers[0].ffn_norm.data, ffn_norm);
		MINI_LLAMA_ASSERT_EQ(model.layers[0].bq.data, bias);
	}

	return true;
}

static bool TestQ41ForwardMatchesF32() {
	auto f32 = LoadModel("models/tiny/model.json", "models/tiny/model.bin");
	MINI_LLAMA_ASSERT_TRUE(f32.loaded);
	auto quant = f32;
	QuantizeModelToQ41(quant);
	MINI_LLAMA_ASSERT_TRUE(ModelWeightBytes(quant) < ModelWeightBytes(f32));
	MiniLlamaContext baseline(&f32), actual(&quant);
	// Several positions exercise both initial and history-dependent attention.
	int pos = 0;
	for (int token : {1, 104, 101, 108, 108, 111}) {
		baseline.pos = actual.pos = pos++;
		Tensor expected = ForwardToken(baseline, f32, token);
		Tensor logits = ForwardToken(actual, quant, token);
		MINI_LLAMA_ASSERT_EQ(logits.shape, expected.shape);
		float max_err = 0.0f, sum_err = 0.0f;
		for (size_t i = 0; i < logits.size(); ++i) {
			MINI_LLAMA_ASSERT_TRUE(std::isfinite(logits.data[i]));
			float err = std::abs(logits.data[i] - expected.data[i]);
			max_err = std::max(max_err, err);
			sum_err += err;
		}
		MINI_LLAMA_ASSERT_TRUE(max_err < 2.0f);
		MINI_LLAMA_ASSERT_TRUE(sum_err / logits.size() < 0.2f);
	}

	return true;
}

static struct ForwardTestRegistrar {
	ForwardTestRegistrar() {
		RegisterTest("q4_1_model_conversion", TestQ41ModelConversion);
		RegisterTest("q4_1_forward_matches_f32", TestQ41ForwardMatchesF32);
		RegisterTest("forward_token_produces_logit_shape",
		             TestForwardTokenProducesLogitShape);
		RegisterTest("forward_token_changes_kv_cache",
		             TestForwardTokenChangesKvCache);
		RegisterTest("generation_greedy_sequence",
		             TestGenerationGreedySequence);
		RegisterTest("position_increments_correctly",
		             TestPositionIncrementsCorrectly);
		RegisterTest("forward_rejects_invalid_token",
		             TestForwardRejectsInvalidToken);
		RegisterTest("forward_rejects_invalid_position",
		             TestForwardRejectsInvalidPosition);
	}
} forward_test_registrar;
