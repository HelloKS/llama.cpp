#include "models.h"

void llama_model_nemotron_h_mtp::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp,        false);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp,      false);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,               hparams.n_expert_shared, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_MOE_LATENT_SIZE,                   hparams.moe_latent_size, false);

    GGML_ASSERT(hparams.n_layer_nextn > 0
                && "NEMOTRON_H_MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn <= hparams.n_layer_all);

}

void llama_model_nemotron_h_mtp::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;
    GGML_UNUSED(ml);

    const int64_t moe_n_embd = hparams.moe_latent_size > 0 ? hparams.moe_latent_size : n_embd;

    // embeddings
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

    // output
    {
        output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, TENSOR_NOT_REQUIRED);
        output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
        // if output is NULL, init from the input tok embed, duplicated to allow offloading
        if (output == NULL) {
            output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED | TENSOR_NOT_REQUIRED);
        }
    }

    // MTP model: skip trunk layers, only load the MTP block tensors
    const uint32_t n_main = n_layer;

    for (int i = 0; i < n_layer_all; ++i) {
        auto & layer = layers[i];

        // Skip trunk layers - they are not executed in the MTP graph
        if (static_cast<uint32_t>(i) < n_main) {
            continue;
        }

        // MTP block: attention layer tensors
        layer.attn_norm  = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        const int64_t n_head_i = hparams.n_head(i);
        const int64_t n_embd_k_gqa_i = hparams.n_embd_k_gqa(i);
        const int64_t n_embd_v_gqa_i = hparams.n_embd_v_gqa(i);
        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head_i, n_embd_k_gqa_i, n_embd_v_gqa_i, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head_i, n_embd}, 0);
        layer.wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);

        // MoE sub-layer norm (FFN entry norm before MoE)
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);

        // FFN/MoE layer tensors
        if (n_expert != 0) {
            const int64_t n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff / n_expert_used;
            const int64_t n_ff_shexp = hparams.n_ff_shexp;

            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), { n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert         }, 0);

            // MoE branch
            layer.ffn_latent_down = create_tensor(tn(LLM_TENSOR_FFN_LATENT_DOWN, "weight", i), {n_embd, moe_n_embd}, TENSOR_NOT_REQUIRED);
            layer.ffn_latent_up   = create_tensor(tn(LLM_TENSOR_FFN_LATENT_UP,   "weight", i), {moe_n_embd, n_embd}, TENSOR_NOT_REQUIRED);

            layer.ffn_down_exps   = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   moe_n_embd, n_expert}, 0);
            layer.ffn_up_exps     = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {moe_n_embd, n_ff_exp, n_expert}, 0);

            // Shared expert branch
            layer.ffn_down_shexp  = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd}, 0);
            layer.ffn_up_shexp    = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_shexp}, 0);
        } else {
            // mlp layers
            layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  hparams.n_ff(i), n_embd}, 0);
            layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   hparams.n_ff(i)}, 0);
            layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias",   i), {n_embd}, TENSOR_NOT_REQUIRED);
            layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias",   i), {hparams.n_ff(i)}, TENSOR_NOT_REQUIRED);
        }

        // NextN/MTP fusion tensors
        layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", i), { 2 * n_embd, n_embd }, 0);
        layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", i), { n_embd },            0);
        layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", i), { n_embd },            0);
        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", i), { n_embd, n_vocab },   TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), { n_embd, n_vocab },   TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), { n_embd },            TENSOR_NOT_REQUIRED);
    }
}

std::unique_ptr<llm_graph_context> llama_model_nemotron_h_mtp::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_nemotron_h_mtp::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    GGML_ASSERT(hparams.n_layer_nextn > 0 && "NEMOTRON_H_MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "NEMOTRON_H_MTP currently only supports a single MTP block");

    // The MTP layer is stored immediately after the main layers in model.layers[].
    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");

    auto inp = std::make_unique<llm_graph_input_embd>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->embd);
    ggml_set_name(inp->embd, "mtp_h_input");

    ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;

    ggml_tensor * h_input  = inp->embd;
    ggml_tensor * tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    cb(tok_embd, "mtp_tok_embd", il);

    res->add_input(std::move(inp));

    build_inp_pos();
    auto * inp_attn       = build_attn_inp_kv();

    // === Fusion: norm → concat → project ===
    ggml_tensor * h_norm = build_norm(h_input, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat);
    cb(cur, "mtp_eh_proj", il);

    // === Attention (standard MHA - no gated attention) ===
    // vLLM pre-norm residual pattern: residual = fused (before norm), added in next layer
    ggml_tensor * residual = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    auto [Qcur, Kcur, Vcur] = build_qkv(layer, cur, n_embd_head, hparams.n_head(il), hparams.n_head_kv(il), il);

    const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp_attn,
            layer.wo, layer.wo_b, layer.wo_s,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "mtp_attn_out", il);
    // NOTE: no residual add here — follows vLLM AttentionDecoderLayer pattern

    // === FFN/MoE layer ===
    // vLLM pre-norm residual: hidden_states += residual (on layer entry)
    cur = ggml_add(ctx0, cur, residual);  // attn_output + fused
    cb(cur, "mtp_attn_add_residual", il);

    // Save summed value as new residual for end_norm
    residual = cur;

    // MoE sub-layer norm (FFN entry norm before MoE)
    if (layer.ffn_norm) {
        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "mtp_ffn_norm", il);
    }

    if (layer.ffn_gate_inp == nullptr) {
        cur = build_ffn(cur,
                layer.ffn_up,   layer.ffn_up_b,   layer.ffn_up_s,
                NULL,           NULL,              NULL,
                layer.ffn_down, layer.ffn_down_b, layer.ffn_down_s,
                NULL,
                LLM_FFN_RELU_SQR, LLM_FFN_PAR, il);
        cb(cur, "mtp_ffn_out", il);
    } else {
        ggml_tensor * inp_emb    = cur;
        ggml_tensor * inp_latent = cur;

        if (layer.ffn_latent_down) {
            inp_latent = ggml_mul_mat(ctx0, layer.ffn_latent_down, cur);
        }

        ggml_tensor * router_logits = build_lora_mm(layer.ffn_gate_inp, cur);
        cb(router_logits, "mtp_ffn_moe_logits", il);

        ggml_tensor * moe_out =
            build_moe_ffn(inp_latent,
                    layer.ffn_gate_inp,
                    layer.ffn_up_exps,
                    nullptr, // no gate
                    layer.ffn_down_exps,
                    layer.ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_RELU_SQR, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID,
                    il,
                    router_logits, nullptr,
                    layer.ffn_up_exps_s,
                    nullptr, // no gate
                    layer.ffn_down_exps_s);
        cb(moe_out, "mtp_ffn_moe_out", il);

        if (layer.ffn_latent_up) {
            moe_out = ggml_mul_mat(ctx0, layer.ffn_latent_up, moe_out);
        }

        ggml_tensor * ffn_shexp = build_ffn(inp_emb,
                    layer.ffn_up_shexp,   NULL, layer.ffn_up_shexp_s,
                    NULL /* no gate */   , NULL, NULL,
                    layer.ffn_down_shexp, NULL, layer.ffn_down_shexp_s,
                    NULL,
                    LLM_FFN_RELU_SQR, LLM_FFN_PAR, il);
        cb(ffn_shexp, "mtp_ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "mtp_ffn_out", il);
    }

    // End norm: cur += residual, then final_layernorm (shared_head_norm)
    // This matches vLLM's AttentionMoEDecoderLayer has_end_norm pattern:
    //   cur = cur + residual  (MoE output + pre-norm-summed)
    //   cur = final_layernorm(cur)
    cur = ggml_add(ctx0, cur, residual);  // MoE output + (attn+fusion)
    cb(cur, "mtp_end_add_residual", il);

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w && "NEMOTRON_H_MTP: missing both nextn.shared_head_norm and output_norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_end_norm", il);

    // Pre-norm hidden state: used by the AR draft loop to seed the next MTP step.
    cb(cur, "h_pre_norm", -1);
    res->t_h_nextn = cur;

    // === LM head ===
    // No additional norm here — end_norm (shared_head_norm) already applied above.
    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    GGML_ASSERT(head_w && "NEMOTRON_H_MTP: missing LM head (nextn.shared_head_head or model.output)");
    cur = build_lora_mm(head_w, cur);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
