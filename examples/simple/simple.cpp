#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include "../vendor/nlohmann/json.hpp"

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [-pipo pipo_alg_config] [prompt]\n", argv[0]);
    printf("\n");
}


#define QK_K 256
#define K_SCALE_SIZE 12

typedef struct {
    union {
        struct {
            ggml_fp16_t d;
            ggml_fp16_t dmin;
        } s;
        uint32_t dm;
    } u;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K/2];
} block_q4_K;

typedef struct {
    uint8_t ql[QK_K/2];
    uint8_t qh[QK_K/4];
    int8_t  scales[QK_K/16];
    ggml_fp16_t d;
} block_q6_K;

extern "C" {
    void dequantize_row_q4_K(const block_q4_K * x, float * y, int64_t k);
    void dequantize_row_q6_K(const block_q6_K * x, float * y, int64_t k);
}

static void dump_tensor(ggml_tensor* input){
    // return;
    if(!input) return;
    FILE * fp = fopen("logs/input_dump.log", "a");
    if (fp) {
        fprintf(fp, "input: %s type: %s shape: %ld %ld %ld %ld\n", input->name, ggml_type_name(input->type), input->ne[0], input->ne[1], input->ne[2], input->ne[3]);
        int64_t n_elements = ggml_nelements(input);
        int n_print = n_elements < 20 ? (int)n_elements : 20;

        if (input->type == GGML_TYPE_F32) {
            float data[20];
            ggml_backend_tensor_get(input, data, 0, sizeof(float) * n_print);
            for (int i = 0; i < n_print; i++) {
                fprintf(fp, "%f ", data[i]);
            }
            fprintf(fp, "\n");
        } else if (input->type == GGML_TYPE_Q4_K) {
            block_q4_K data;
            ggml_backend_tensor_get(input, &data, 0, sizeof(block_q4_K));
            float out[QK_K];
            dequantize_row_q4_K(&data, out, QK_K);
            for (int i = 0; i < n_print; i++) {
                fprintf(fp, "%f ", out[i]);
            }
            fprintf(fp, "\n");
        } else if (input->type == GGML_TYPE_Q6_K) {
            block_q6_K data;
            ggml_backend_tensor_get(input, &data, 0, sizeof(block_q6_K));
            float out[QK_K];
            dequantize_row_q6_K(&data, out, QK_K);
            for (int i = 0; i < n_print; i++) {
                fprintf(fp, "%f ", out[i]);
            }
            fprintf(fp, "\n");
        } else if (input->type == GGML_TYPE_I32) {
            int32_t data[20];
            ggml_backend_tensor_get(input, data, 0, sizeof(int32_t) * n_print);
            for (int i = 0; i < n_print; i++) {
                fprintf(fp, "%d ", data[i]);
            }
            fprintf(fp, "\n");
        } else if (input->type == GGML_TYPE_I64) {
            int64_t data[20];
            ggml_backend_tensor_get(input, data, 0, sizeof(int64_t) * n_print);
            for (int i = 0; i < n_print; i++) {
                fprintf(fp, "%ld ", data[i]);
            }
            fprintf(fp, "\n");
        } else if (input->type == GGML_TYPE_F16) {
            ggml_fp16_t data[20];
            ggml_backend_tensor_get(input, data, 0, sizeof(ggml_fp16_t) * n_print);
            for (int i = 0; i < n_print; i++) {
                fprintf(fp, "%f ", ggml_fp16_to_fp32(data[i]));
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
    }
        
}

static bool my_eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        // Return true to observe this tensor
        // You can filter by name here, e.g.:
        // return false;
        return strstr(t->name, "ffn_up")!=NULL || strstr(t->name, "ffn_out")!=NULL;
        // return true; 
    }
    
    dump_tensor(t);
    dump_tensor(t->src[0]);
    dump_tensor(t->src[1]);
    
    return true;
}

// struct llama_model_tensor_buft_override {
//     const char * pattern;
//     ggml_backend_buffer_type_t buft;
// };

void pipo_tensor_layout(std::vector<llama_model_tensor_buft_override>& overrides,
                        ggml_backend_buffer_type_t cuda, ggml_backend_buffer_type_t cuda_host){

    overrides.clear();

    overrides.push_back({ "blk\\.([0-9]|[1-3][0-9])\\.ffn_down\\.weight", cuda_host });
    overrides.push_back({ "blk\\.([0-9]|[1-3][0-9])\\.ffn_up\\.weight", cuda_host });

    overrides.push_back({ "^token_embd\\.weight$", cuda_host });
    // overrides.push_back({ "^output\\.weight$", cuda_host });
    // overrides.push_back({ "^output_norm\\.weight$", cuda_host });
    overrides.push_back({ ".*", cuda });

    // Terminate with nullptr
    overrides.push_back({ nullptr, nullptr });
}

void pipo_assign_offload(std::vector<const char*>& prefill_offload, std::vector<const char*>& decode_offload) {
    prefill_offload.clear();
    decode_offload.clear();
    
    prefill_offload.push_back("blk\\.([0-9]|[1-3][0-9])\\.ffn_down\\.weight");
    prefill_offload.push_back("blk\\.([0-9]|[1-3][0-9])\\.ffn_up\\.weight");

    decode_offload.push_back("blk\\.(3|7|11|15|19|23|27|31|35|39)\\.ffn_down\\.weight");
    // decode_offload.push_back("blk\\.([0-9]|[1-3][0-9])\\.ffn_up\\.weight");
}

int main(int argc, char ** argv) {
    // path to the model gguf file
    std::string model_path;
    // prompt to generate text from
    std::string prompt = "Once upon a time, in a land far, far away, there was a small village nestled between two great mountains. The villagers were known for their kindness and their peculiar habit of singing to the stars every night. One day, a mysterious traveler arrived at the village gates, carrying nothing but a worn-out leather bag and a wooden staff. The traveler claimed to have come from the other side of the world, seeking a legendary artifact said to be hidden deep within the caves of the northern mountain.";
    // number of layers to offload to the GPU
    int ngl = 99;
    // number of tokens to predict
    int n_predict = 32;
    bool enable_pipo = false;
    // path to pipo perf file
    std::string pipo_alg_result_path;
    // parse command line arguments

    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    model_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-n") == 0) {
                if (i + 1 < argc) {
                    try {
                        n_predict = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-ngl") == 0) {
                if (i + 1 < argc) {
                    try {
                        ngl = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-pipo") == 0) {
                enable_pipo = true;
                if (i + 1 < argc){
                    pipo_alg_result_path = argv[++i];
                }else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else {
                // prompt starts here
                break;
            }
        }
        if (model_path.empty()) {
            print_usage(argc, argv);
            return 1;
        }
        if (i < argc) {
            prompt = argv[i++];
            for (; i < argc; i++) {
                prompt += " ";
                prompt += argv[i];
            }
        }
    }
    // int n_cpu_layers_per_split = 0;
    std::vector<std::string> overrides_list, decode_offloads_list;
    // load alg result
    if (enable_pipo){
        std::ifstream conf_file(pipo_alg_result_path, std::ios_base::in);
        auto j = nlohmann::json::parse(conf_file);
        overrides_list.assign(j["overrides"].begin(), j["overrides"].end());
        decode_offloads_list.assign(j["offloads"].begin(), j["offloads"].end());
    }
    // load dynamic backends
    ggml_backend_load_all();

    // initialize the model

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;
    model_params.enable_pipo = enable_pipo;
    // model_params.no_host = false;
    model_params.use_mmap = true;
    // model_params.n_cpu_layers_per_split = n_cpu_layers_per_split;
    // model_params.use_extra_bufts = false;
    std::vector<llama_model_tensor_buft_override> overrides;

    if(enable_pipo){
        ggml_backend_buffer_type_t cuda,cuda_host;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto * dev = ggml_backend_dev_get(i);
            auto * buft = ggml_backend_dev_buffer_type(dev);
            if (buft) {
                auto name = ggml_backend_buft_name(buft);
                if (strstr(name, "CUDA")){
                    cuda = buft;
                    cuda_host = ggml_backend_dev_host_buffer_type(dev);
                    break;
                }
            }
        }
        // pipo_tensor_layout(overrides, cuda, cuda_host);
        for (auto& override : overrides_list){
            overrides.push_back({override.c_str(), cuda_host});
        }
        overrides.push_back({".*", cuda});
        overrides.push_back({nullptr, nullptr});
        model_params.tensor_buft_overrides = overrides.data();
    }

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    // tokenize the prompt

    // find the number of tokens in the prompt
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);

    // allocate space for the tokens and tokenize the prompt
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    // pre-assign op_offload
    std::vector<const char*> p_offload, d_offload;
    if(enable_pipo){
        // pipo_assign_offload(p_offload, d_offload);
        for (auto & offload : decode_offloads_list){
            d_offload.push_back(offload.c_str());
        }
        // offload all
        for (auto& override: overrides_list){
            p_offload.push_back(override.c_str());
        }
        llama_model_set_offload(model, p_offload.data(), d_offload.data(), p_offload.size(), d_offload.size());
    }

    // initialize the context

    llama_context_params ctx_params = llama_context_default_params();
    // n_ctx is the context size
    ctx_params.n_ctx = n_prompt + n_predict - 1;
    // n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
    ctx_params.n_batch = n_prompt;
    // enable performance counters
    ctx_params.no_perf = false;

    // if(enable_pipo) ctx_params.op_offload = false;

    ctx_params.enable_pipo = enable_pipo;
    // ctx_params.n_cpu_layers_per_split = n_cpu_layers_per_split;

    // ctx_params.cb_eval = my_eval_callback;
    // ctx_params.cb_eval_user_data = NULL;

    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // initialize the sampler

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    // llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8));
    // llama_sampler_chain_add(smpl, llama_sampler_init_dist(1234));
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // print the prompt token-by-token

    for (auto id : prompt_tokens) {
        char buf[128];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            return 1;
        }
        std::string s(buf, n);
        printf("%s", s.c_str());
    }

    // prepare a batch for the prompt

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    if (llama_model_has_encoder(model)) {
        if (llama_encode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return 1;
        }

        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
            decoder_start_token_id = llama_vocab_bos(vocab);
        }

        batch = llama_batch_get_one(&decoder_start_token_id, 1);
    }

    // main loop

    const auto t_main_start = ggml_time_us();
    int n_decode = 0;
    llama_token new_token_id;

    std::vector<std::string> tokens;

    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }

        n_pos += batch.n_tokens;

        // sample the next token
        {
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // is it an end of generation?
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }
            std::string s(buf, n);
            tokens.push_back(s);
            // printf("%s", s.c_str());
            // fflush(stdout);

            // prepare the next batch with the sampled token
            batch = llama_batch_get_one(&new_token_id, 1);

            n_decode += 1;
        }
    }
    printf("\n");
    for(auto &s: tokens){
        printf("%s", s.c_str());
        fflush(stdout);
    }
    printf("\n");
    fflush(stdout);

    const auto t_main_end = ggml_time_us();

    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    fprintf(stderr, "\n");
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);
    fprintf(stderr, "\n");

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
