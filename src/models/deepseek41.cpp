#include "llama-kv-cache-dsv4.h"
#include "models.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <vector>

void llama_model_deepseek41::load_arch_hparams(llama_model_loader & ml) {
    llama_model_deepseek4::load_arch_hparams(ml);

    hparams.n_embd_out_impl = hparams.n_embd;
    if (hparams.dsv4_hc_mult < 2 || hparams.dsv4_o_group_count == 0 ||
        hparams.n_head() % hparams.dsv4_o_group_count != 0 ||
        hparams.n_embd_head_k() != hparams.n_embd_head_v() || hparams.n_embd_head_k() < hparams.n_rot() ||
        hparams.n_rot() % 2 != 0 || hparams.n_swa == 0) {
        throw std::runtime_error("DeepSeek-V4.1 attention dimensions are invalid");
    }

    auto read_layers = [&](llm_kv key, std::bitset<LLAMA_MAX_LAYERS> & result) {
        std::vector<uint32_t> layers;
        ml.get_arr(key, layers);
        if (!std::is_sorted(layers.begin(), layers.end())) {
            throw std::runtime_error("DeepSeek-V4.1 layers must be ordered");
        }
        result.reset();
        for (uint32_t il : layers) {
            if (il >= hparams.n_layer()) {
                throw std::runtime_error(format("%s layer %u is out of range", ml.llm_kv(key).c_str(), il));
            }
            if (result.test(il)) {
                throw std::runtime_error("DeepSeek-V4.1 source layers must be unique and ordered");
            }
            result.set(il);
        }
        return layers;
    };

    read_layers(LLM_KV_ATTENTION_KV_SOURCE_LAYERS, hparams.dsv41_kv_sources);
    read_layers(LLM_KV_ATTENTION_INDEXER_SOURCE_LAYERS, hparams.dsv41_index_sources);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_CANDIDATE_SOURCE_LAYER, hparams.dsv41_candidate_source_layer);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_CANDIDATE_BLOCK_SIZE, hparams.dsv41_candidate_block_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_CANDIDATE_TOP_K_BLOCKS, hparams.dsv41_candidate_top_k_blocks);

    const auto engram_layers         = read_layers(LLM_KV_ENGRAM_LAYERS, hparams.dsv41_engram_layers);
    hparams.dsv41_engram_layer_count = engram_layers.size();
    ml.get_key(LLM_KV_ENGRAM_MAX_NGRAM_SIZE, hparams.dsv41_engram_max_ngram_size);
    ml.get_key(LLM_KV_ENGRAM_HEAD_COUNT, hparams.dsv41_engram_head_count);
    ml.get_key(LLM_KV_ENGRAM_HEAD_DIM, hparams.dsv41_engram_head_dim);
    ml.get_key(LLM_KV_ENGRAM_PAD_TOKEN_ID, hparams.dsv41_engram_pad_token_id);
    ml.get_key(LLM_KV_ENGRAM_COMPRESSED_VOCAB_SIZE, hparams.dsv41_engram_compressed_vocab_size);

    std::vector<uint32_t> table_rows;
    ml.get_arr(LLM_KV_ENGRAM_TABLE_ROWS, table_rows);
    if (table_rows.size() != engram_layers.size()) {
        throw std::runtime_error("DeepSeek-V4.1 Engram table row count does not match its layer count");
    }
    for (size_t i = 0; i < engram_layers.size(); ++i) {
        hparams.dsv41_engram_table_rows[engram_layers[i]] = table_rows[i];
    }

    const uint64_t ngram = hparams.dsv41_engram_max_ngram_size;
    const uint64_t heads = hparams.dsv41_engram_head_count;
    const uint64_t modules = hparams.dsv41_engram_layer_count;
    if (ngram < 2 || ngram > LLAMA_MAX_PLE_NGRAM || heads == 0 || heads > LLAMA_MAX_PLE_HEADS ||
        modules == 0 || ngram * modules > LLAMA_MAX_PLE_NGRAM ||
        (ngram - 1) * heads * modules > LLAMA_MAX_PLE_HEADS || hparams.dsv41_engram_head_dim == 0) {
        throw std::runtime_error("DeepSeek-V4.1 Engram metadata exceeds loader limits");
    }
    const uint64_t hash_heads = (ngram - 1) * heads;
    const uint64_t total_hash_heads = hash_heads * modules;

    const auto check_length = [&](llm_kv key, uint64_t expected) {
        uint32_t count = 0;
        ml.get_arr_n(key, count);
        if (count != expected) {
            throw std::runtime_error(format("%s size does not match Engram configuration", ml.llm_kv(key).c_str()));
        }
    };
    check_length(LLM_KV_ENGRAM_HASH_MULTIPLIERS, ngram * modules);
    check_length(LLM_KV_ENGRAM_HEAD_OFFSETS, total_hash_heads);
    check_length(LLM_KV_ENGRAM_HEAD_BUCKET_SIZES, total_hash_heads);
    std::array<uint64_t, LLAMA_MAX_PLE_HEADS> offsets = {};
    std::array<uint64_t, LLAMA_MAX_PLE_HEADS> bucket_sizes = {};
    ml.get_arr(LLM_KV_ENGRAM_HASH_MULTIPLIERS, hparams.dsv41_engram_hash_multipliers);
    ml.get_arr(LLM_KV_ENGRAM_HEAD_OFFSETS, offsets);
    ml.get_arr(LLM_KV_ENGRAM_HEAD_BUCKET_SIZES, bucket_sizes);
    for (uint64_t i = 0; i < total_hash_heads; ++i) {
        const uint64_t rows = table_rows[i / hash_heads];
        if (rows == 0 || rows > INT32_MAX || bucket_sizes[i] == 0 || offsets[i] >= rows ||
            bucket_sizes[i] > rows - offsets[i]) {
            throw std::runtime_error("DeepSeek-V4.1 Engram hash bucket exceeds its embedding table");
        }
        hparams.dsv41_engram_head_offsets[i] = offsets[i];
        hparams.dsv41_engram_head_bucket_sizes[i] = bucket_sizes[i];
    }

    ml.get_arr(LLM_KV_ENGRAM_TOKEN_MAP, engram_token_map);
    if (hparams.dsv41_engram_compressed_vocab_size == 0 ||
        hparams.dsv41_engram_pad_token_id >= engram_token_map.size()) {
        throw std::runtime_error("DeepSeek-V4.1 Engram vocabulary or padding token is invalid");
    }
    for (auto token : engram_token_map) {
        if ((uint64_t) token >= hparams.dsv41_engram_compressed_vocab_size) {
            throw std::runtime_error("DeepSeek-V4.1 Engram token map contains an invalid token");
        }
    }

    int kv_source = -1;
    int index_source = -1;
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        const auto ratio = hparams.dsv4_compress_ratios[il];
        if (hparams.dsv41_kv_sources.test(il)) {
            kv_source = il;
            if (!hparams.dsv41_index_sources.test(il)) {
                throw std::runtime_error("DeepSeek-V4.1 KV sources must also own an indexer");
            }
        }
        if (hparams.dsv41_index_sources.test(il)) {
            index_source = il;
        }
        if (ratio > 2 || (ratio == 0 && (kv_source == (int) il || index_source == (int) il)) ||
            (ratio != 0 && (kv_source < 0 || index_source < kv_source ||
                hparams.dsv4_compress_ratios[kv_source] != ratio || hparams.dsv4_compress_ratios[index_source] != ratio))) {
            throw std::runtime_error("DeepSeek-V4.1 compressed attention source layout is invalid");
        }
    }
    const auto candidate = hparams.dsv41_candidate_source_layer;
    if (candidate >= hparams.n_layer() || !hparams.dsv41_index_sources.test(candidate) ||
        hparams.dsv41_candidate_block_size == 0 || 256 % hparams.dsv41_candidate_block_size != 0 ||
        hparams.dsv41_candidate_top_k_blocks == 0 || hparams.indexer_top_k == 0 ||
        hparams.indexer_n_head == 0 || hparams.indexer_head_size < hparams.n_rot()) {
        throw std::runtime_error("DeepSeek-V4.1 candidate indexer configuration is invalid");
    }
    for (uint32_t il = candidate; il < hparams.n_layer(); ++il) {
        if (hparams.dsv4_compress_ratios[il] != hparams.dsv4_compress_ratios[candidate]) {
            throw std::runtime_error("DeepSeek-V4.1 candidate layers must use the same compression ratio");
        }
    }

    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.set_swa_pattern(0);
    type = LLM_TYPE_UNKNOWN;
}

void llama_model_deepseek41::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;
    GGML_UNUSED(ml);

    if (engram_token_map.size() != (size_t) n_vocab) {
        throw std::runtime_error("DeepSeek-V4.1 Engram token map size does not match vocabulary size");
    }

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t n_ff_exp        = hparams.n_ff_exp();
    const int64_t n_expert_shared = hparams.n_expert_shared;
    const int64_t n_embd_head     = hparams.n_embd_head_k();
    const int64_t o_groups        = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank     = hparams.dsv4_o_lora_rank;
    const int64_t hc              = hparams.dsv4_hc_mult;
    const int64_t hc_dim          = hc * n_embd;
    const int64_t hc_mix_dim      = (2 + hc) * hc;
    const int64_t hash_heads      = (hparams.dsv41_engram_max_ngram_size - 1) * hparams.dsv41_engram_head_count;

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, 0);

    for (int il = 0; il < n_layer; ++il) {
        auto & layer = layers[il];

        layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", il), { n_embd }, 0);
        layer.attn_sinks    = create_tensor(tn(LLM_TENSOR_ATTN_SINKS, "weight", il), { n_head }, 0);
        layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", il), { n_embd, q_lora_rank }, 0);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", il), { q_lora_rank }, 0);
        layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", il), { q_lora_rank, n_head * n_embd_head }, 0);
        layer.wkv  = create_tensor(tn(LLM_TENSOR_ATTN_KV, "weight", il), { n_embd, n_embd_head }, 0);
        layer.attn_kv_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_NORM, "weight", il), { n_embd_head }, 0);
        layer.wo_a         = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A, "weight", il),
                                           { n_head * n_embd_head / o_groups, o_lora_rank, o_groups }, TENSOR_ALLOW_RESHAPE);
        layer.wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B, "weight", il), { o_groups * o_lora_rank, n_embd }, 0);

        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN, "weight", il), { hc_dim, hc_mix_dim }, 0);
        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE, "weight", il), { hc_mix_dim }, 0);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", il), { 3 }, 0);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN, "weight", il), { hc_dim, hc_mix_dim }, 0);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE, "weight", il), { hc_mix_dim }, 0);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE, "weight", il), { 3 }, 0);

        if (hparams.dsv41_kv_sources.test(il)) {
            layer.attn_comp_wkv =
                create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WKV, "weight", il), { n_embd, n_embd_head }, 0);
            layer.attn_comp_norm = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_NORM, "weight", il), { n_embd_head }, 0);
            if (hparams.dsv4_compress_ratios[il] > 1) {
                layer.attn_comp_wgate =
                    create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WGATE, "weight", il), { n_embd, n_embd_head }, 0);
            }
        }

        if (hparams.dsv41_index_sources.test(il)) {
            layer.indexer_proj =
                create_tensor(tn(LLM_TENSOR_INDEXER_PROJ, "weight", il), { n_embd, hparams.indexer_n_head }, 0);
            layer.indexer_attn_q_b =
                create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", il),
                              { q_lora_rank, hparams.indexer_n_head * hparams.indexer_head_size }, 0);
            if (hparams.dsv41_kv_sources.test(il)) {
                layer.indexer_attn_k = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K, "weight", il),
                                                     { n_embd_head, hparams.indexer_head_size }, 0);
                layer.indexer_k_norm =
                    create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", il), { hparams.indexer_head_size }, 0);
            }
        }

        if (hparams.dsv41_engram_layers.test(il)) {
            layer.engram_embd =
                create_tensor(tn(LLM_TENSOR_ENGRAM_EMBD, "weight", il),
                              { hparams.dsv41_engram_head_dim, hparams.dsv41_engram_table_rows[il] }, TENSOR_READ_LAZY);
            layer.engram_wkv = create_tensor(tn(LLM_TENSOR_ENGRAM_WKV, "weight", il),
                                             { hash_heads * hparams.dsv41_engram_head_dim, n_embd * (hc + 1) }, 0);
            layer.engram_q   = create_tensor(tn(LLM_TENSOR_ENGRAM_Q, "weight", il), { n_embd, hc }, 0);
            layer.engram_k   = create_tensor(tn(LLM_TENSOR_ENGRAM_K, "weight", il), { n_embd, hc }, 0);
        }

        layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", il), { n_embd, n_expert }, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", il), { n_expert }, 0);
        layer.ffn_exp_probs_b_vl =
            create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B_VL, "bias", il), { n_expert }, TENSOR_NOT_REQUIRED);
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", il), { n_embd }, 0);
        layer.ffn_gate_exps =
            create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", il), { n_embd, n_ff_exp, n_expert }, 0);
        layer.ffn_down_exps =
            create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, 0);
        layer.ffn_up_exps = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "weight", il), { n_embd, n_ff_exp, n_expert }, 0);
        layer.ffn_gate_shexp =
            create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", il), { n_embd, n_ff_exp * n_expert_shared }, 0);
        layer.ffn_down_shexp =
            create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", il), { n_ff_exp * n_expert_shared, n_embd }, 0);
        layer.ffn_up_shexp =
            create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "weight", il), { n_embd, n_ff_exp * n_expert_shared }, 0);
    }
}

class llm_graph_input_dsv41_engram : public llm_graph_input_i {
  public:
    llm_graph_input_dsv41_engram(const llama_model_deepseek41 & model,
                                 const llama_kv_cache_dsv4_raw_context * mctx,
                                 uint32_t                       module) :
        model(model),
        mctx(mctx),
        module(module) {}

    void set_input(const llama_ubatch * ubatch) override {
        const auto &             hp              = model.hparams;
        const int64_t            n_tokens        = ubatch->n_tokens;
        const int64_t            ngram           = hp.dsv41_engram_max_ngram_size;
        const int64_t            heads_per_ngram = hp.dsv41_engram_head_count;
        const int64_t            hash_heads      = (ngram - 1) * heads_per_ngram;
        const int64_t            pad             = model.engram_token_map.at(hp.dsv41_engram_pad_token_id);
        std::vector<int32_t>     idx(hash_heads * n_tokens);
        std::vector<llama_token> prev;
        mctx->get_prev_tokens(*ubatch, ngram - 1, prev);

        for (int64_t i = 0; i < n_tokens; ++i) {
            GGML_ASSERT(ubatch->token != nullptr && "DeepSeek-V4.1 requires token IDs for Engram");
            GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "DeepSeek-V4.1 Engram does not support shared tokens");
            std::vector<int64_t> tokens(ngram, pad);
            tokens[0]    = model.engram_token_map.at(ubatch->token[i]);
            bool blocked = false;
            for (int64_t shift = 1; shift < ngram; ++shift) {
                const llama_token token = blocked ? LLAMA_TOKEN_NULL : prev[i * (ngram - 1) + (ngram - 1 - shift)];
                blocked                 = blocked || token < 0;
                tokens[shift]           = blocked ? pad : model.engram_token_map.at(token);
            }

            const int64_t mult_base = module * ngram;
            const int64_t head_base = module * hash_heads;
            uint64_t      mixed     = (uint64_t) tokens[0] * hp.dsv41_engram_hash_multipliers[mult_base];
            for (int64_t n = 2; n <= ngram; ++n) {
                mixed ^= (uint64_t) tokens[n - 1] * hp.dsv41_engram_hash_multipliers[mult_base + n - 1];
                for (int64_t head = 0; head < heads_per_ngram; ++head) {
                    const int64_t local_head         = (n - 2) * heads_per_ngram + head;
                    const int64_t global_head        = head_base + local_head;
                    idx[i * hash_heads + local_head] = mixed % hp.dsv41_engram_head_bucket_sizes[global_head] +
                                                       hp.dsv41_engram_head_offsets[global_head];
                }
            }
        }

        ggml_backend_tensor_set(rows, idx.data(), 0, idx.size() * sizeof(int32_t));
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_kv_cache_dsv4_context *>(params.mctx)->get_raw();
        return rows->ne[0] == (int64_t) model.hparams.dsv41_engram_head_count *
                                  (model.hparams.dsv41_engram_max_ngram_size - 1) * params.ubatch.n_tokens;
    }

    ggml_tensor *                  rows = nullptr;
    const llama_model_deepseek41 & model;
    const llama_kv_cache_dsv4_raw_context * mctx;
    const uint32_t                 module;
};

class llm_graph_input_dsv41_index : public llm_graph_input_i {
  public:
    llm_graph_input_dsv41_index(const llama_kv_cache_dsv4_context * mctx, uint32_t ratio, uint32_t block_size) :
        mctx(mctx), ratio(ratio), block_size(block_size) {}

    void set_input(const llama_ubatch * ubatch) override {
        const auto & plan = ratio == 2 ? mctx->get_csa_plan(*ubatch) : mctx->get_hca_plan(*ubatch);
        if (write_idxs) {
            const int64_t source_size = ratio == 2 ? mctx->get_csa()->get_n_kv() : mctx->get_hca()->get_n_kv();
            const int64_t target_size = mctx->get_lid()->get_n_kv();
            std::vector<int64_t> idxs = plan.state_write_idxs;
            for (auto & idx : idxs) {
                idx = (idx / source_size) * target_size + idx % source_size;
            }
            ggml_backend_tensor_set(write_idxs, idxs.data(), 0, idxs.size() * sizeof(int64_t));
        }
        if (latest_block) {
            const int64_t blocks = latest_block->ne[0];
            std::vector<float> bias(blocks * ubatch->n_tokens, 0.0f);
            for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
                const int64_t visible = (ubatch->pos[i] + 1) / ratio;
                if (visible > 0) {
                    bias[i * blocks + (visible - 1) / block_size] = INFINITY;
                }
            }
            ggml_backend_tensor_set(latest_block, bias.data(), 0, bias.size() * sizeof(float));
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_kv_cache_dsv4_context *>(params.mctx);
        const auto & plan = ratio == 2 ? mctx->get_csa_plan(params.ubatch) : mctx->get_hca_plan(params.ubatch);
        return (!write_idxs || write_idxs->ne[0] == (int64_t) plan.state_write_idxs.size()) &&
            (!latest_block || (latest_block->ne[0] == plan.n_kv / block_size &&
                latest_block->ne[1] == params.ubatch.n_tokens / plan.n_stream && latest_block->ne[3] == plan.n_stream));
    }

    const llama_kv_cache_dsv4_context * mctx;
    const uint32_t ratio;
    const uint32_t block_size;
    ggml_tensor * write_idxs = nullptr;
    ggml_tensor * latest_block = nullptr;
};

ggml_tensor * llama_model_deepseek41::graph::build_attention41(
        const llama_model & model, llm_graph_input_dsv4 * inp,
        ggml_tensor * cur, ggml_tensor * inp_pos, int il) {
    const auto & layer = model.layers[il];
    const int64_t dim = hparams.n_embd_head_k();
    const int64_t nt = cur->ne[1];
    const uint32_t ratio = hparams.dsv4_compress_ratios[il];
    const float base = ratio ? hparams.dsv4_compress_rope_base : freq_base;
    const float scale = ratio ? freq_scale : 1.0f;
    const float ext = ratio ? ext_factor : 0.0f;
    const float attn_factor = ext == 0.0f ? 1.0f : 1.0f / (1.0f + 0.1f * logf(1.0f / scale));
    const auto rope = [&](ggml_tensor * x, ggml_tensor * pos, bool inverse = false) {
        auto fn = inverse ? ggml_rope_ext_back : ggml_rope_ext;
        x = fn(ctx0, x, pos, nullptr, hparams.n_rot(), rope_type, ratio ? n_ctx_orig : 0,
                base, scale, ext, attn_factor, ratio ? beta_fast : 0.0f, ratio ? beta_slow : 0.0f);
        return ggml_rope_set_offset(x, x->ne[0] - hparams.n_rot());
    };
    const auto slice_k = [&](ggml_tensor * k, int64_t count) {
        GGML_ASSERT(count <= k->ne[2]);
        return ggml_view_4d(ctx0, k, k->ne[0], k->ne[1], count, k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);
    };

    ggml_tensor * qr = build_norm(build_lora_mm(layer.wq_a, cur), layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * q = ggml_reshape_3d(ctx0, build_lora_mm(layer.wq_b, qr), dim, n_head, nt);
    q = rope(q, inp_pos);
    ggml_tensor * kv = build_norm(build_lora_mm(layer.wkv, cur), layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
    kv = rope(ggml_reshape_3d(ctx0, kv, dim, 1, nt), inp_pos);

    ggml_tensor * out = nullptr;
    if (ratio == 0) {
        out = build_raw_attention(inp->get_raw(), q, kv, layer.attn_sinks, 1.0f / sqrtf(float(dim)), il);
    } else {
        const auto & comp = ratio == 2 ? inp->get_csa() : inp->get_hca();
        const auto * cache = ratio == 2 ? inp->mctx->get_csa() : inp->mctx->get_hca();
        const auto * state = ratio == 2 ? inp->mctx->get_csa_state() : inp->mctx->get_hca_state();
        const auto * lid = inp->mctx->get_lid();
        const int64_t count = comp.kq_mask->ne[0];
        const int64_t ns = comp.kq_mask->ne[3];
        const int64_t index_dim = hparams.indexer_head_size;
        const int64_t index_heads = hparams.indexer_n_head;
        const bool owns_kv = hparams.dsv41_kv_sources.test(il);
        const bool candidate_source = (uint32_t) il == hparams.dsv41_candidate_source_layer;
        auto aux = std::make_unique<llm_graph_input_dsv41_index>(inp->mctx, ratio, hparams.dsv41_candidate_block_size);

        if (owns_kv) {
            kv_source = il;
            ggml_tensor * values = build_lora_mm(layer.attn_comp_wkv, cur);
            ggml_tensor * scores = ratio == 1 ? ggml_fill(ctx0, values, 0.0f) : build_lora_mm(layer.attn_comp_wgate, cur);
            ggml_tensor * latent = build_compressed_latent(comp, state, values, scores, layer.attn_comp_norm, il);
            if (latent) {
                ggml_tensor * compressed = rope(latent, comp.state_write_pos);
                if (comp.k_rot) {
                    compressed = llama_mul_mat_hadamard(ctx0, compressed, comp.k_rot);
                }
                ggml_build_forward_expand(gf, cache->cpy_k(ctx0, compressed, comp.state_write_idxs, il));

                ggml_tensor * key = build_lora_mm(layer.indexer_attn_k,
                        ggml_reshape_2d(ctx0, latent, dim, latent->ne[2]));
                key = build_norm(key, layer.indexer_k_norm, nullptr, LLM_NORM_RMS, il);
                key = rope(ggml_reshape_3d(ctx0, key, index_dim, 1, key->ne[1]), comp.state_write_pos);
                if (inp->get_lid().k_rot) {
                    key = llama_mul_mat_hadamard(ctx0, key, inp->get_lid().k_rot);
                }
                aux->write_idxs = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, comp.state_write_idxs->ne[0]);
                ggml_set_input(aux->write_idxs);
                ggml_build_forward_expand(gf, lid->cpy_k(ctx0, key, aux->write_idxs, il));
            }
        }

        if (hparams.dsv41_index_sources.test(il)) {
            ggml_tensor * iq = build_lora_mm(layer.indexer_attn_q_b, qr);
            iq = rope(ggml_reshape_3d(ctx0, iq, index_dim, index_heads, nt), inp_pos);
            if (inp->get_lid().k_rot) {
                iq = llama_mul_mat_hadamard(ctx0, iq, inp->get_lid().k_rot);
            }
            iq = ggml_reshape_4d(ctx0, iq, index_dim, index_heads, nt / ns, ns);
            ggml_tensor * ik = slice_k(lid->get_k(ctx0, kv_source), count);
            ggml_tensor * weights = build_lora_mm(layer.indexer_proj, cur);
            weights = ggml_scale(ctx0, weights, 1.0f / sqrtf(float(index_dim * index_heads)));
            weights = ggml_reshape_4d(ctx0, weights, index_heads, nt / ns, 1, ns);
            ggml_tensor * scores = ggml_mul_mat(ctx0,
                    ggml_permute(ctx0, ik, 0, 2, 1, 3), ggml_permute(ctx0, iq, 0, 2, 1, 3));
            scores = ggml_cont(ctx0, ggml_permute(ctx0, scores, 2, 1, 0, 3));
            scores = ggml_sum_rows(ctx0, ggml_mul(ctx0, ggml_relu(ctx0, scores), weights));
            scores = ggml_cont(ctx0, ggml_permute(ctx0, scores, 2, 1, 0, 3));
            scores = ggml_add(ctx0, scores, ggml_cast(ctx0, comp.kq_mask, GGML_TYPE_F32));

            if (candidate_source) {
                const int64_t block_size = hparams.dsv41_candidate_block_size;
                const int64_t blocks = count / block_size;
                ggml_tensor * block_scores = ggml_pool_1d(ctx0, scores, GGML_OP_POOL_MAX, block_size, block_size, 0);
                aux->latest_block = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, blocks, nt / ns, 1, ns);
                ggml_set_input(aux->latest_block);
                block_scores = ggml_add(ctx0, block_scores, aux->latest_block);
                ggml_tensor * selected = ggml_cont(ctx0, ggml_top_k(ctx0, block_scores,
                        std::min<int64_t>(blocks, hparams.dsv41_candidate_top_k_blocks)));
                ggml_tensor * mask = ggml_fill(ctx0, ggml_cast(ctx0, block_scores, comp.kq_mask->type), 0.0f);
                mask = build_top_k_mask(mask, selected, "index_candidate_blocks", il);
                mask = ggml_reshape_4d(ctx0, mask, 1, blocks, nt / ns, ns);
                mask = ggml_repeat_4d(ctx0, mask, block_size, blocks, nt / ns, ns);
                candidate_mask = ggml_reshape_4d(ctx0, mask, count, nt / ns, 1, ns);
            } else if ((uint32_t) il > hparams.dsv41_candidate_source_layer) {
                GGML_ASSERT(candidate_mask);
                scores = ggml_add(ctx0, scores, ggml_cast(ctx0, candidate_mask, GGML_TYPE_F32));
            }
            ggml_tensor * selected = ggml_cont(ctx0, ggml_top_k(ctx0, scores,
                    std::min<int64_t>(count, hparams.indexer_top_k)));
            index_mask = build_top_k_mask(comp.kq_mask, selected, "index_top_k_mask", il);
        }
        if (aux->write_idxs || aux->latest_block) {
            res->add_input(std::move(aux));
        }
        GGML_ASSERT(index_mask && kv_source >= 0);

        const auto * raw = inp->get_raw();
        if (raw->self_k_rot) {
            q = llama_mul_mat_hadamard(ctx0, q, raw->self_k_rot);
            kv = llama_mul_mat_hadamard(ctx0, kv, raw->self_k_rot);
        }
        ggml_build_forward_expand(gf, q);
        ggml_build_forward_expand(gf, raw->mctx->cpy_k(ctx0, kv, raw->get_k_idxs(), il));
        ggml_tensor * keys = ggml_concat(ctx0, raw->mctx->get_k(ctx0, il), slice_k(cache->get_k(ctx0, kv_source), count), 2);
        ggml_tensor * mask = ggml_concat(ctx0, raw->get_kq_mask(), index_mask, 0);
        out = build_attn_mha(q, keys, keys, nullptr, mask, layer.attn_sinks, nullptr, 0, 1.0f / sqrtf(float(dim)), il);
        if (raw->self_k_rot) {
            out = llama_mul_mat_hadamard(ctx0, out, raw->self_k_rot);
        }
    }

    out = rope(ggml_reshape_3d(ctx0, out, dim, n_head, nt), inp_pos, true);
    const int64_t groups = hparams.dsv4_o_group_count;
    out = ggml_reshape_3d(ctx0, out, n_head * dim / groups, groups, nt);
    out = ggml_mul_mat(ctx0, layer.wo_a, ggml_permute(ctx0, out, 0, 2, 1, 3));
    out = ggml_cont_2d(ctx0, ggml_permute(ctx0, out, 0, 2, 1, 3), hparams.dsv4_o_lora_rank * groups, nt);
    out = build_lora_mm(layer.wo_b, out);
    cb(out, "attn_out", il);
    return out;
}

llama_model_deepseek41::graph::graph(const llama_model & model_base, const llm_graph_params & params) :
    llama_model_deepseek4::graph(params) {
    const auto & model = static_cast<const llama_model_deepseek41 &>(model_base);

    ggml_tensor *                 inp         = build_inp_embd(model.tok_embd);
    ggml_tensor *                 inp_pos     = build_inp_pos();
    ggml_tensor *                 inp_out_ids = build_inp_out_ids();
    llm_graph_input_dsv4 * inp_attn = build_inp_dsv4();

    const int64_t hc         = hparams.dsv4_hc_mult;
    const int64_t hash_heads = (hparams.dsv41_engram_max_ngram_size - 1) * hparams.dsv41_engram_head_count;
    ggml_tensor * inpL = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens), n_embd, hc, n_tokens, 1);

    ggml_tensor * one     = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens), 1.0f);
    ggml_tensor * zero    = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc - 1, n_tokens), 0.0f);
    ggml_tensor * pre_mix = ggml_concat(ctx0, one, zero, 0);

    uint32_t engram_module = 0;
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        if (hparams.dsv41_engram_layers.test(il)) {
            auto engram_inp =
                std::make_unique<llm_graph_input_dsv41_engram>(model, inp_attn->mctx->get_raw(), engram_module++);
            engram_inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, hash_heads * n_tokens);
            ggml_set_input(engram_inp->rows);
            ggml_tensor * rows = engram_inp->rows;
            res->add_input(std::move(engram_inp));

            ggml_tensor * emb   = ggml_get_rows(ctx0, layer.engram_embd, rows);
            emb                 = ggml_reshape_2d(ctx0, emb, hash_heads * hparams.dsv41_engram_head_dim, n_tokens);
            ggml_tensor * kv    = build_lora_mm(layer.engram_wkv, emb);
            kv                  = ggml_reshape_3d(ctx0, kv, n_embd, hc + 1, n_tokens);
            ggml_tensor * key   = ggml_view_3d(ctx0, kv, n_embd, hc, n_tokens, kv->nb[1], kv->nb[2], 0);
            ggml_tensor * value = ggml_view_2d(ctx0, kv, n_embd, n_tokens, kv->nb[2], kv->nb[1] * hc);
            key                 = ggml_rms_norm(ctx0, key, norm_rms_eps);
            key                 = ggml_mul(ctx0, key, ggml_cast(ctx0, layer.engram_k, GGML_TYPE_F32));
            ggml_tensor * query = ggml_rms_norm(ctx0, inpL, norm_rms_eps);
            query               = ggml_mul(ctx0, query, ggml_cast(ctx0, layer.engram_q, GGML_TYPE_F32));
            ggml_tensor * score =
                ggml_scale(ctx0, ggml_sum_rows(ctx0, ggml_mul(ctx0, query, key)), 1.0f / sqrtf((float) n_embd));
            ggml_tensor * magnitude = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, score), 1e-6f, INFINITY));
            ggml_tensor * gate      = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, score), magnitude));
            value                   = ggml_cont(ctx0, value);
            value = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, value, n_embd, 1, n_tokens), n_embd, hc, n_tokens, 1);
            inpL  = ggml_add(ctx0, inpL, ggml_mul(ctx0, value, gate));
            cb(inpL, "engram_out", il);
        }

        if ((size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) {
            ggml_tensor * weights = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc, n_tokens), 1.0f / hc);
            res->t_layer_inp[il] = build_hc_pre(inpL, weights, il);
            cb(res->t_layer_inp[il], "layer_inp", il);
            ggml_build_forward_expand(gf, res->t_layer_inp[il]);
        }

        ggml_tensor * residual  = inpL;
        ggml_tensor * attn_pre  = nullptr;
        ggml_tensor * attn_post = nullptr;
        ggml_tensor * attn_comb = nullptr;
        build_hc_pre(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, &attn_post, &attn_comb, il,
                     &attn_pre);
        ggml_tensor * cur = build_hc_pre(inpL, pre_mix, il);
        cur               = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cur               = build_attention41(model, inp_attn, cur, inp_pos, il);
        inpL              = build_hc_post(cur, residual, attn_post, attn_comb, il);

        residual               = inpL;
        ggml_tensor * ffn_pre  = nullptr;
        ggml_tensor * ffn_post = nullptr;
        ggml_tensor * ffn_comb = nullptr;
        build_hc_pre(inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, &ffn_post, &ffn_comb, il, &ffn_pre);
        cur = build_hc_pre(inpL, attn_pre, il);
        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);

        ggml_tensor * moe = build_moe_ffn(cur, layer.ffn_gate_inp, layer.ffn_up_exps, layer.ffn_gate_exps,
                                          layer.ffn_down_exps, layer.ffn_exp_probs_b, n_expert, hparams.n_expert_used(),
                                          LLM_FFN_SILU, hparams.expert_weights_norm, hparams.expert_weights_scale,
                                          (llama_expert_gating_func_type) hparams.expert_gating_func, il);
        ggml_tensor * shared =
            build_ffn(cur, layer.ffn_up_shexp, nullptr, nullptr, layer.ffn_gate_shexp, nullptr, nullptr,
                      layer.ffn_down_shexp, nullptr, nullptr, nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cur     = ggml_add(ctx0, moe, shared);
        inpL    = build_hc_post(cur, residual, ffn_post, ffn_comb, il);
        inpL    = build_cvec(inpL, il);
        pre_mix = ffn_pre;
    }

    ggml_tensor * cur = build_hc_pre(inpL, pre_mix, -1);
    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }
    cur           = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    res->t_embd   = cur;
    cur           = ggml_mul_mat(ctx0, model.output, cur);
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}

std::unique_ptr<llm_graph_context> llama_model_deepseek41::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}
