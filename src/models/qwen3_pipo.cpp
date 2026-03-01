#include "models.h"
#include <regex>

bool need_offload(const std::vector<std::regex>& regex, std::string name){
    for(const auto & pattern: regex){
        if (std::regex_search(name, pattern)){
            return true;
        }
    }
    return false;
}

void init_regex(std::vector<std::regex>& regex, const std::vector<const char*>& patterns){
    regex.clear();
    for(const auto &p:patterns){
        if (p) {
            regex.emplace_back(p);
        }
    }
}

llm_build_qwen3_pipo::llm_build_qwen3_pipo(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * attn_norm;
    ggml_tensor * wq;
    ggml_tensor * wk;
    ggml_tensor * wv;
    ggml_tensor * attn_q_norm;
    ggml_tensor * attn_k_norm;
    ggml_tensor * wo;
    ggml_tensor * bo;
    ggml_tensor * ffn_norm;
    ggml_tensor * ffn_up;
    ggml_tensor * ffn_gate;
    ggml_tensor * ffn_down;
    std::vector<std::regex> regex;
    if(params.gtype == LLM_GRAPH_TYPE_DEFAULT_PREFILL){
        init_regex(regex, model.p_offload_weights);
    }
    else{
        GGML_ASSERT(params.gtype == LLM_GRAPH_TYPE_DEFAULT_DECODE);
        init_regex(regex, model.d_offload_weights);
    }

    for (int il = 0; il < n_layer; ++il) {

        auto get_offloaded = [&](ggml_tensor * t) {
            if (t && need_offload(regex, std::string(t->name))) {
                struct ggml_tensor * dynamic_tensor = ggml_backend_sched_get_pipo_tensor(params.sched, model.name_weight_map.at(std::string(t->name)));
                res->dynamic_src_tensor_list[dynamic_tensor->name].push_back(t);
                res->dynamic_dst_tensor_list[dynamic_tensor->name].push_back(dynamic_tensor);
                return dynamic_tensor;
            }
            return t;
        };

        const llama_layer& layer = model.layers[il];
        attn_norm   = get_offloaded(layer.attn_norm);
        wq          = get_offloaded(layer.wq);
        wk          = get_offloaded(layer.wk);
        wv          = get_offloaded(layer.wv);
        attn_q_norm = get_offloaded(layer.attn_q_norm);
        attn_k_norm = get_offloaded(layer.attn_k_norm);
        wo          = get_offloaded(layer.wo);
        bo          = get_offloaded(layer.bo);
        ffn_norm    = get_offloaded(layer.ffn_norm);
        ffn_up      = get_offloaded(layer.ffn_up);
        ffn_gate    = get_offloaded(layer.ffn_gate);
        ffn_down    = get_offloaded(layer.ffn_down);

        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q and K and RoPE them
            ggml_tensor * Qcur = build_lora_mm(wq, cur);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = build_lora_mm(wk, cur);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = build_lora_mm(wv, cur);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = build_norm(Qcur, attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = build_norm(Kcur, attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    wo, bo,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = build_norm(ffn_inp,
                ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                ffn_up,   NULL, NULL,
                ffn_gate, NULL, NULL,
                ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
