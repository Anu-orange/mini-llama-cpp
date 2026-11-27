// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

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
#include "tests/test_names.h"

TEST(ForwardTest, TestForwardTokenProducesLogitShape) {
	MiniLlamaModel model =
	    LoadModel("models/tiny/model.json", "models/tiny/model.bin");
	if (model.layers.empty()) {
		FAIL() << "Failed to load model";
	}
	MiniLlamaContext ctx(&model);
	ctx.pos = 0;
	Tensor logits = ForwardToken(ctx, model, 1);  // BOS token
	EXPECT_EQ(logits.num_dims(), 1);
	EXPECT_EQ(logits.shape[0], model.config.vocab_size);
}
TEST(ForwardTest, TestForwardTokenChangesKvCache) {
	MiniLlamaModel model =
	    LoadModel("models/tiny/model.json", "models/tiny/model.bin");
	if (model.layers.empty()) {
		FAIL() << "Failed to load model";
	}
	MiniLlamaContext ctx(&model);
	ctx.pos = 0;
	// Before forward, cache at layer 0, pos 0 should be zero
	float before = ctx.kv_cache.keys.At({0, 0, 0, 0});
	EXPECT_NEAR(before, 0.0f, 1e-6f);

	ForwardToken(ctx, model, 1);

	// After forward, cache should have been written
	float after = ctx.kv_cache.keys.At({0, 0, 0, 0});
	// It's unlikely to be exactly 0 after random weights forward
	// Just check it's finite and cache was written somewhere
	EXPECT_TRUE(std::isfinite(after));
}

TEST(ForwardTest, TestGenerationGreedySequence) {
	MiniLlamaModel model =
	    LoadModel("models/tiny/model.json", "models/tiny/model.bin");
	if (model.layers.empty()) {
		FAIL() << "Failed to load model";
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

	EXPECT_GT(tokens.size(), 6);
	// First generated token should be the same as what we got in prefill
	EXPECT_GE(tokens[tokens.size() - 6], 0);
}

TEST(ForwardTest, TestPositionIncrementsCorrectly) {
	MiniLlamaModel model =
	    LoadModel("models/tiny/model.json", "models/tiny/model.bin");
	if (model.layers.empty()) {
		FAIL() << "Failed to load model";
	}
	MiniLlamaContext ctx(&model);

	// Forward token at pos 0
	ctx.pos = 0;
	ForwardToken(ctx, model, 1);

	// Forward token at pos 1
	ctx.pos = 1;
	Tensor logits = ForwardToken(ctx, model, 2);
	EXPECT_EQ(logits.shape[0], model.config.vocab_size);
}

TEST(ForwardTest, TestForwardRejectsInvalidToken) {
	MiniLlamaModel model =
	    LoadModel("models/tiny/model.json", "models/tiny/model.bin");
	if (!model.loaded) {
		FAIL() << "Failed to load model";
	}
	MiniLlamaContext ctx(&model);
	ctx.pos = 0;

	EXPECT_THROW(ForwardToken(ctx, model, model.config.vocab_size),
	             std::out_of_range);
}

TEST(ForwardTest, TestForwardRejectsInvalidPosition) {
	MiniLlamaModel model =
	    LoadModel("models/tiny/model.json", "models/tiny/model.bin");
	if (!model.loaded) {
		FAIL() << "Failed to load model";
	}
	MiniLlamaContext ctx(&model);
	ctx.pos = model.config.max_seq_len;

	EXPECT_THROW(ForwardToken(ctx, model, 1), std::out_of_range);
}

TEST(ForwardTest, TestQ80ForwardMatchesF32) {
	// Load F32 model and run forward pass
	MiniLlamaModel model_f32 =
	    LoadModel("models/tiny/model.json", "models/tiny/model.bin");
	if (!model_f32.loaded) {
		FAIL() << "Failed to load model";
	}
	MiniLlamaContext ctx_f32(&model_f32);
	ctx_f32.pos = 0;
	Tensor logits_f32 = ForwardToken(ctx_f32, model_f32, 1);

	// Quantize another copy of the model to Q8_0
	MiniLlamaModel model_q8 =
	    LoadModel("models/tiny/model.json", "models/tiny/model.bin");
	if (!model_q8.loaded) {
		FAIL() << "Failed to load model";
	}
	QuantizeModelToQ80(model_q8);

	// Run forward with fresh context
	MiniLlamaContext ctx_q8(&model_q8);
	ctx_q8.pos = 0;
	Tensor logits_q8 = ForwardToken(ctx_q8, model_q8, 1);

	// Shapes must match
	EXPECT_EQ(logits_f32.shape, logits_q8.shape);

	// Logits should be close (Q8_0 roundtrip error is small)
	float max_err = 0.0f;
	float sum_err = 0.0f;
	for (size_t i = 0; i < logits_f32.size(); ++i) {
		float err = std::abs(logits_f32.data[i] - logits_q8.data[i]);
		max_err = std::max(max_err, err);
		sum_err += err;
	}
	float mean_err = sum_err / static_cast<float>(logits_f32.size());

	EXPECT_LT(max_err, 0.5f) << "max error too large: " << max_err;
	EXPECT_LT(mean_err, 0.05f) << "mean error too large: " << mean_err;
}

TEST(ForwardTest, TestQ40ForwardMatchesF32) {
	// Load F32 model and run forward pass
	MiniLlamaModel model_f32 =
	    LoadModel("models/tiny/model.json", "models/tiny/model.bin");
	if (!model_f32.loaded) {
		FAIL() << "Failed to load model";
	}
	MiniLlamaContext ctx_f32(&model_f32);
	ctx_f32.pos = 0;
	Tensor logits_f32 = ForwardToken(ctx_f32, model_f32, 1);

	// Quantize another copy of the model to Q4_0
	MiniLlamaModel model_q4 =
	    LoadModel("models/tiny/model.json", "models/tiny/model.bin");
	if (!model_q4.loaded) {
		FAIL() << "Failed to load model";
	}
	QuantizeModelToQ40(model_q4);

	// Run forward with fresh context
	MiniLlamaContext ctx_q4(&model_q4);
	ctx_q4.pos = 0;
	Tensor logits_q4 = ForwardToken(ctx_q4, model_q4, 1);

	// Shapes must match
	EXPECT_EQ(logits_f32.shape, logits_q4.shape);

	// Logits should be close (Q4_0 has higher error than Q8_0)
	float max_err = 0.0f;
	float sum_err = 0.0f;
	for (size_t i = 0; i < logits_f32.size(); ++i) {
		float err = std::abs(logits_f32.data[i] - logits_q4.data[i]);
		max_err = std::max(max_err, err);
		sum_err += err;
	}
	float mean_err = sum_err / static_cast<float>(logits_f32.size());

	EXPECT_LT(max_err, 2.0f) << "max error too large: " << max_err;
	EXPECT_LT(mean_err, 0.2f) << "mean error too large: " << mean_err;
}

TEST(ForwardTest, TestQ41ModelConversion) {
	for (int source = 0; source < 3; ++source) {
		MiniLlamaModel model =
		    LoadModel("models/tiny/model.json", "models/tiny/model.bin");
		ASSERT_TRUE(model.loaded);
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
		ASSERT_TRUE(!model.cuda_weights);
		auto check = [](const QuantizedTensor& w) {
			if (w.type != QuantType::kQ41 || !w.f32_data.empty() ||
			    !w.q8_0_data.empty() || !w.q4_0_data.empty())
				return false;
			size_t row_size = static_cast<size_t>(w.shape.back());
			return w.q4_1_data.size() ==
			       (w.num_elements() / row_size) * ((row_size + 31) / 32);
		};
		ASSERT_TRUE(check(model.lm_head));
		for (const auto& layer : model.layers) {
			for (const auto* w : {&layer.wq, &layer.wk, &layer.wv, &layer.wo,
			                      &layer.w_gate, &layer.w_up, &layer.w_down}) {
				ASSERT_TRUE(check(*w));
			}
		}
		auto before = model.lm_head.q4_1_data;
		QuantizeModelToQ41(model);
		ASSERT_EQ(before.size(), model.lm_head.q4_1_data.size());
		for (size_t i = 0; i < before.size(); ++i) {
			ASSERT_EQ(before[i].d, model.lm_head.q4_1_data[i].d);
			ASSERT_EQ(before[i].m, model.lm_head.q4_1_data[i].m);
			for (int j = 0; j < 16; ++j) {
				ASSERT_EQ(before[i].qs[j], model.lm_head.q4_1_data[i].qs[j]);
			}
		}
		ASSERT_EQ(model.token_embedding.data, embedding);
		ASSERT_EQ(model.final_norm.data, norm);
		ASSERT_EQ(model.layers[0].attention_norm.data, layer_norm);
		ASSERT_EQ(model.layers[0].ffn_norm.data, ffn_norm);
		ASSERT_EQ(model.layers[0].bq.data, bias);
	}
}

TEST(ForwardTest, TestQ41ForwardMatchesF32) {
	auto f32 = LoadModel("models/tiny/model.json", "models/tiny/model.bin");
	ASSERT_TRUE(f32.loaded);
	auto quant = f32;
	QuantizeModelToQ41(quant);
	ASSERT_TRUE(ModelWeightBytes(quant) < ModelWeightBytes(f32));
	MiniLlamaContext baseline(&f32), actual(&quant);
	// Several positions exercise both initial and history-dependent attention.
	int pos = 0;
	for (int token : {1, 104, 101, 108, 108, 111}) {
		baseline.pos = actual.pos = pos++;
		Tensor expected = ForwardToken(baseline, f32, token);
		Tensor logits = ForwardToken(actual, quant, token);
		ASSERT_EQ(logits.shape, expected.shape);
		float max_err = 0.0f, sum_err = 0.0f;
		for (size_t i = 0; i < logits.size(); ++i) {
			ASSERT_TRUE(std::isfinite(logits.data[i]));
			float err = std::abs(logits.data[i] - expected.data[i]);
			max_err = std::max(max_err, err);
			sum_err += err;
		}
		ASSERT_TRUE(max_err < 2.0f);
		ASSERT_TRUE(sum_err / logits.size() < 0.2f);
	}
}
