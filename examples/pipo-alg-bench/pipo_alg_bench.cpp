#include "llama.h"
#include "llama-context.h"
#include "llama-model.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include "../vendor/nlohmann/json.hpp"

static void print_usage(int argc, char ** argv) {
    (void)argc;
    printf("\nusage:\n");
    printf("\n    %s -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [-pipo pipo_alg_config] [-p n_prompt] [-r random] [-run n_runs] [prompt]\n", argv[0]);
    printf("\n");
}

static nlohmann::json* pipo_memory_breakdown(const struct llama_context * ctx) {
    const std::vector<ggml_backend_dev_t> & devices = ctx->get_model().devices;
    nlohmann::json* mem_usage = new nlohmann::json();
    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> memory_breakdown = ctx->memory_breakdown();

    constexpr size_t MiB = 1024 * 1024;
    const std::vector<std::string> desc_prefixes_strip = {"NVIDIA ", "GeForce ", "Tesla ", "AMD ", "Radeon ", "Instinct "};

    // track seen buffer types to avoid double counting:
    std::set<ggml_backend_buffer_type_t> seen_buffer_types;

    // accumulative memory breakdown for each device and for host:
    std::vector<llama_memory_breakdown_data> mb_dev(devices.size());
    llama_memory_breakdown_data              mb_host;

    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const llama_memory_breakdown_data & mb   = buft_mb.second;
        if (ggml_backend_buft_is_host(buft)) {
            mb_host.model   += mb.model;
            mb_host.context += mb.context;
            mb_host.compute += mb.compute;
            seen_buffer_types.insert(buft);
            continue;
        }
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (dev) {
            int i_dev = -1;
            for (size_t i = 0; i < devices.size(); i++) {
                if (devices[i] == dev) {
                    i_dev = i;
                    break;
                }
            }
            if (i_dev != -1) {
                mb_dev[i_dev].model   += mb.model;
                mb_dev[i_dev].context += mb.context;
                mb_dev[i_dev].compute += mb.compute;
                seen_buffer_types.insert(buft);
                continue;
            }
        }
    }

    // print memory breakdown for each device:
    for (size_t i = 0; i < devices.size(); i++) {
        ggml_backend_dev_t          dev = devices[i];
        llama_memory_breakdown_data mb  = mb_dev[i];

        const std::string name = ggml_backend_dev_name(dev);
        std::string desc = ggml_backend_dev_description(dev);
        for (const std::string & prefix : desc_prefixes_strip) {
            if (desc.length() >= prefix.length() && desc.substr(0, prefix.length()) == prefix) {
                desc = desc.substr(prefix.length());
            }
        }

        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);

        const size_t self = mb.model + mb.context + mb.compute;
        nlohmann::json gpu_break;
        gpu_break["total"] = total / MiB;
        gpu_break["free"] = free / MiB;
        gpu_break["self"] = self / MiB;
        gpu_break["model"] = mb.model / MiB;
        gpu_break["context"] = mb.context / MiB;
        gpu_break["compute"] = mb.compute / MiB;

        (*mem_usage)[name] = gpu_break;
    }

    // print memory breakdown for host:
    {
        const size_t self = mb_host.model + mb_host.context + mb_host.compute;
        nlohmann::json cpu_break;
        cpu_break["total"] = "";
        cpu_break["free"] = "";
        cpu_break["self"] = self / MiB;
        cpu_break["model"] = mb_host.model / MiB;
        cpu_break["context"] = mb_host.context / MiB;
        cpu_break["compute"] = mb_host.compute / MiB;

        (*mem_usage)["Host"] = cpu_break;
    }

    // print memory breakdown for all remaining buffer types:
    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const llama_memory_breakdown_data & mb   = buft_mb.second;
        if (seen_buffer_types.count(buft) == 1) {
            continue;
        }
        const std::string name = ggml_backend_buft_name(buft);
        const size_t self = mb.model + mb.context + mb.compute;
        nlohmann::json ext_buf_break;
        ext_buf_break["total"] = "";
        ext_buf_break["free"] = "";
        ext_buf_break["self"] = self / MiB;
        ext_buf_break["model"] = mb.model / MiB;
        ext_buf_break["context"] = mb.context / MiB;
        ext_buf_break["compute"] = mb.compute / MiB;

        (*mem_usage)[name] = ext_buf_break;
        seen_buffer_types.insert(buft);
    }
    return mem_usage;
}

// prompt to generate text from
extern std::string prompt;

int main(int argc, char ** argv) {
    std::string model_path;
    // number of layers to offload to the GPU
    int ngl = 99;
    // number of tokens to predict
    int n_predict = 32;
    int n_prompt_target = -1; // -1 means use prompt length, otherwise fill to this length
    bool enable_random = false;
    bool enable_pipo = false;
    int n_runs = 5;
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
            } else if (strcmp(argv[i], "-p") == 0) {
                if (i + 1 < argc) {
                    try {
                        n_prompt_target = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-r") == 0) {
                enable_random = true;
            } else if (strcmp(argv[i], "-run") == 0) {
                if (i + 1 < argc) {
                    try {
                        n_runs = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            }
            else {
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
        #if 0
        pipo_tensor_layout(overrides, cuda, cuda_host);
        #else
        for (auto& override : overrides_list){
            overrides.push_back({override.c_str(), cuda_host});
        }
        overrides.push_back({".*", cuda});
        overrides.push_back({nullptr, nullptr});
        #endif

        model_params.tensor_buft_overrides = overrides.data();
    }

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    // tokenize the prompt
    std::vector<llama_token> prompt_tokens;

    if (enable_random && n_prompt_target > 0) {
        fprintf(stderr, "Generating %d random tokens for prompt\n", n_prompt_target);
        prompt_tokens.resize(n_prompt_target);
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        int start_idx = 0;
        if (llama_vocab_get_add_bos(vocab)) {
            prompt_tokens[0] = llama_vocab_bos(vocab);
            start_idx = 1;
        }
        for (int i = start_idx; i < n_prompt_target; i++) {
            prompt_tokens[i] = std::rand() % n_vocab;
        }
    } else {
        if (n_prompt_target > 0) {
            // repeat prompt until we roughly hit the target length
             fprintf(stderr, "Repeating base prompt to roughly hit %d tokens\n", n_prompt_target);
             std::string base_prompt = prompt;
             int current_len = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
             while (current_len < n_prompt_target) {
                 prompt += " " + base_prompt;
                 current_len = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
             }
        }
        
        // find the number of tokens in the prompt
        const int n_prompt_actual = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
        
        // limit to target length if specified
        int final_n_prompt = (n_prompt_target > 0 && n_prompt_target < n_prompt_actual) ? n_prompt_target : n_prompt_actual;

        // allocate space for the tokens and tokenize the prompt
        prompt_tokens.resize(n_prompt_actual);
        if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
            fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
            return 1;
        }
        prompt_tokens.resize(final_n_prompt); // truncate if we generated too many
    }
    
    int n_prompt = prompt_tokens.size();

    // pre-assign op_offload
    std::vector<const char*> p_offload, d_offload;
    if(enable_pipo){
        #if 0
        pipo_assign_offload(p_offload, d_offload);
        #else
        for (auto & offload : decode_offloads_list){
            d_offload.push_back(offload.c_str());
        }
        // offload all
        for (auto& override: overrides_list){
            if(override.find("blk") != override.npos)
            p_offload.push_back(override.c_str());
        }
        #endif
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

    if(enable_pipo) ctx_params.op_offload = false;

    ctx_params.enable_pipo = enable_pipo;
    // ctx_params.n_cpu_layers_per_split = n_cpu_layers_per_split;

    // ctx_params.cb_eval = my_eval_callback;
    // ctx_params.cb_eval_user_data = NULL;

    

    std::vector<double> decode_time_per_token(n_runs); 
    std::vector<double> prefill_time_per_token(n_runs);
    std::vector<double> total_time_per_token(n_runs);
    nlohmann::json* mem_usage = nullptr;
    // bench loop
    for (int run_idx = 0; run_idx < n_runs; run_idx++) {
        fprintf(stderr, "[=== bench run %d begin ===]\n", run_idx);
        llama_context * ctx = llama_init_from_model(model, ctx_params);

        if (ctx == NULL) {
            fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
            return 1;
        }
        // initialize the sampler
        auto sparams         = llama_sampler_chain_default_params();
        sparams.no_perf      = false;
        llama_sampler * smpl = llama_sampler_chain_init(sparams);

        // llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8));
        // llama_sampler_chain_add(smpl, llama_sampler_init_dist(1234));
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

        // print the prompt token-by-token

        if (!enable_random) {
            for (auto id : prompt_tokens) {
                char buf[128];
                int  n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
                if (n < 0) {
                    fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                    return 1;
                }
                std::string s(buf, n);
                fprintf(stderr, "%s", s.c_str());
            }
        } else {
            fprintf(stderr, "[Random prompt initialized, omitted from output]\n");
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

        const auto  t_main_start = ggml_time_us();
        int         n_decode     = 0;
        llama_token new_token_id;

        std::vector<std::string> tokens;

        for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict;) {
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
                int  n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
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

        if (!enable_random) {
            fprintf(stderr, "\n");
            for (auto & s : tokens) {
                fprintf(stderr, "%s", s.c_str());
            }
            fprintf(stderr, "\n");
        } else {
            fprintf(stderr, "[Decoding completed, output omitted in random mode]\n");
        }

        const auto t_main_end = ggml_time_us();

        fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n", __func__, n_decode,
                (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));
        total_time_per_token[run_idx] = ((t_main_end - t_main_start) / 1000.0f);

        fprintf(stderr, "\n");
        llama_perf_sampler_print(smpl);
        llama_perf_context_print(ctx);
        fprintf(stderr, "\n");

        const auto data = llama_perf_context(ctx);
        prefill_time_per_token[run_idx] = data.t_p_eval_ms / data.n_p_eval;
        decode_time_per_token[run_idx] = data.t_eval_ms / data.n_eval;
        llama_sampler_free(smpl);
        if (!mem_usage){
            mem_usage = pipo_memory_breakdown(ctx);
        }
        llama_free(ctx);
    }
    
    llama_model_free(model);

    auto avg = [](const std::vector<double>& arr){
        double sum = 0;
        for (double num : arr) sum += num;
        return sum / (double)arr.size();
    };
    nlohmann::json bench_result;
    bench_result["decode_time_per_token_arr"] = decode_time_per_token;
    bench_result["decode_time_per_token_avg"] = avg(decode_time_per_token);
    bench_result["prefill_time_per_token_arr"] = prefill_time_per_token; 
    bench_result["perfill_time_per_token_avg"] = avg(prefill_time_per_token);
    bench_result["total_time_per_token_arr"] = total_time_per_token;
    bench_result["total_time_per_token_avg"] = avg(total_time_per_token);
    bench_result["mem_usage"] = *mem_usage;
    delete mem_usage;
    fprintf(stdout, "%s", bench_result.dump(4).c_str());
    return 0;
}

std::string prompt = R"(The Gift of Maggie
　　One dollar and eighty-seven cents. That was all. And sixty cents of it was in pennies. Pennies saved one and two at a time by bulldozing the grocer and the vegetable man and the butcher until one's cheeks burned with the silent imputation of parsimony that such close dealing implied. Three times Della counted it. One dollar and eighty- seven cents. And the next day would be Christmas.
　　There was clearly nothing to do but flop down on the shabby little couch and howl. So Della did it. Which instigates the moral reflection that life is made up of sobs, sniffles, and smiles, with sniffles predominating.
　　While the mistress of the home is gradually subsiding from the first stage to the second, take a look at the home. A furnished flat at $8 per week. It did not exactly beggar description, but it certainly had that word on the lookout for the mendicancy squad.
　　In the vestibule below was a letter-box into which no letter would go, and an electric button from which no mortal finger could coax a ring. Also appertaining thereunto was a card bearing the name "Mr. James Dillingham Young."
　　The "Dillingham" had been flung to the breeze during a former period of prosperity when its possessor was being paid $30 per week. Now, when the income was shrunk to $20, though, they were thinking seriously of contracting to a modest and unassuming D. But whenever Mr. James Dillingham Young came home and reached his flat above he was called "Jim" and greatly hugged by Mrs. James Dillingham Young, already introduced to you as Della. Which is all very good.
　　Della finished her cry and attended to her cheeks with the powder rag. She stood by the window and looked out dully at a gray cat walking a gray fence in a gray backyard. Tomorrow would be Christmas Day, and she had only $1.87 with which to buy Jim a present. She had been saving every penny she could for months, with this result. Twenty dollars a week doesn't go far. Expenses had been greater than she had calculated. They always are. Only $1.87 to buy a present for Jim. Her Jim. Many a happy hour she had spent planning for something nice for him. Something fine and rare and sterling--something just a little bit near to being worthy of the honor of being owned by Jim.
　　There was a pier-glass between the windows of the room. Perhaps you have seen a pier-glass in an $8 flat. A very thin and very agile person may, by observing his reflection in a rapid sequence of longitudinal strips, obtain a fairly accurate conception of his looks. Della, being slender, had mastered the art.
　　Suddenly she whirled from the window and stood before the glass. her eyes were shining brilliantly, but her face had lost its color within twenty seconds. Rapidly she pulled down her hair and let it fall to its full length.
　　Now, there were two possessions of the James Dillingham Youngs in which they both took a mighty pride. One was Jim's gold watch that had been his father's and his grandfather's. The other was Della's hair. Had the queen of Sheba lived in the flat across the airshaft, Della would have let her hair hang out the window some day to dry just to depreciate Her Majesty's jewels and gifts. Had King Solomon been the janitor, with all his treasures piled up in the basement, Jim would have pulled out his watch every time he passed, just to see him pluck at his beard from envy.
　　So now Della's beautiful hair fell about her rippling and shining like a cascade of brown waters. It reached below her knee and made itself almost a garment for her. And then she did it up again nervously and quickly. Once she faltered for a minute and stood still while a tear or two splashed on the worn red carpet.
　　On went her old brown jacket; on went her old brown hat. With a whirl of skirts and with the brilliant sparkle still in her eyes, she fluttered out the door and down the stairs to the street.
　　Where she stopped the sign read: "Mne. Sofronie. Hair Goods of All Kinds." One flight up Della ran, and collected herself, panting. Madame, large, too white, chilly, hardly looked the "Sofronie."
　　"Will you buy my hair?" asked Della.
　　"I buy hair," said Madame. "Take yer hat off and let's have a sight at the looks of it."
　　Down rippled the brown cascade.
　　"Twenty dollars," said Madame, lifting the mass with a practised hand.
　　"Give it to me quick," said Della.
　　Oh, and the next two hours tripped by on rosy wings. Forget the hashed metaphor. She was ransacking the stores for Jim's present.
　　She found it at last. It surely had been made for Jim and no one else. There was no other like it in any of the stores, and she had turned all of them inside out. It was a platinum fob chain simple and chaste in design, properly proclaiming its value by substance alone and not by meretricious ornamentation--as all good things should do. It was even worthy of The Watch. As soon as she saw it she knew that it must be Jim's. It was like him. Quietness and value--the description applied to both. Twenty-one dollars they took from her for it, and she hurried home with the 87 cents. With that chain on his watch Jim might be properly anxious about the time in any company. Grand as the watch was, he sometimes looked at it on the sly on account of the old leather strap that he used in place of a chain.
　　When Della reached home her intoxication gave way a little to prudence and reason. She got out her curling irons and lighted the gas and went to work repairing the ravages made by generosity added to love. Which is always a tremendous task, dear friends--a mammoth task.
　　Within forty minutes her head was covered with tiny, close-lying curls that made her look wonderfully like a truant schoolboy. She looked at her reflection in the mirror long, carefully, and critically.
　　"If Jim doesn't kill me," she said to herself, "before he takes a second look at me, he'll say I look like a Coney Island chorus girl. But what could I do--oh! what could I do with a dollar and eighty- seven cents?"
　　At 7 o'clock the coffee was made and the frying-pan was on the back of the stove hot and ready to cook the chops.
　　Jim was never late. Della doubled the fob chain in her hand and sat on the corner of the table near the door that he always entered. Then she heard his step on the stair away down on the first flight, and she turned white for just a moment. She had a habit for saying little silent prayer about the simplest everyday things, and now she whispered: "Please God, make him think I am still pretty."
　　The door opened and Jim stepped in and closed it. He looked thin and very serious. Poor fellow, he was only twenty-two--and to be burdened with a family! He needed a new overcoat and he was without gloves.
　　Jim stopped inside the door, as immovable as a setter at the scent of quail. His eyes were fixed upon Della, and there was an expression in them that she could not read, and it terrified her. It was not anger, nor surprise, nor disapproval, nor horror, nor any of the sentiments that she had been prepared for. He simply stared at her fixedly with that peculiar expression on his face.
　　Della wriggled off the table and went for him.
　　"Jim, darling," she cried, "don't look at me that way. I had my hair cut off and sold because I couldn't have lived through Christmas without giving you a present. It'll grow out again--you won't mind, will you? I just had to do it. My hair grows awfully fast. Say `Merry Christmas!' Jim, and let's be happy. You don't know what a nice-- what a beautiful, nice gift I've got for you."
　　"You've cut off your hair?" asked Jim, laboriously, as if he had not arrived at that patent fact yet even after the hardest mental labor.
　　"Cut it off and sold it," said Della. "Don't you like me just as well, anyhow? I'm me without my hair, ain't I?"
　　Jim looked about the room curiously.
　　"You say your hair is gone?" he said, with an air almost of idiocy.
　　"You needn't look for it," said Della. "It's sold, I tell you--sold and gone, too. It's Christmas Eve, boy. Be good to me, for it went for you. Maybe the hairs of my head were numbered," she went on with sudden serious sweetness, "but nobody could ever count my love for you. Shall I put the chops on, Jim?"
　　Out of his trance Jim seemed quickly to wake. He enfolded his Della. For ten seconds let us regard with discreet scrutiny some inconsequential object in the other direction. Eight dollars a week or a million a year--what is the difference? A mathematician or a wit would give you the wrong answer. The magi brought valuable gifts, but that was not among them. This dark assertion will be illuminated later on.
　　Jim drew a package from his overcoat pocket and threw it upon the table.
　　"Don't make any mistake, Dell," he said, "about me. I don't think there's anything in the way of a haircut or a shave or a shampoo that could make me like my girl any less. But if you'll unwrap that package you may see why you had me going a while at first."
　　White fingers and nimble tore at the string and paper. And then an ecstatic scream of joy; and then, alas! a quick feminine change to hysterical tears and wails, necessitating the immediate employment of all the comforting powers of the lord of the flat.
　　For there lay The Combs--the set of combs, side and back, that Della had worshipped long in a Broadway window. Beautiful combs, pure tortoise shell, with jewelled rims--just the shade to wear in the beautiful vanished hair. They were expensive combs, she knew, and her heart had simply craved and yearned over them without the least hope of possession. And now, they were hers, but the tresses that should have adorned the coveted adornments were gone.
　　But she hugged them to her bosom, and at length she was able to look up with dim eyes and a smile and say: "My hair grows so fast, Jim!"
　　And them Della leaped up like a little singed cat and cried, "Oh, oh!"
　　Jim had not yet seen his beautiful present. She held it out to him eagerly upon her open palm. The dull precious metal seemed to flash with a reflection of her bright and ardent spirit.
　　"Isn't it a dandy, Jim? I hunted all over town to find it. You'll have to look at the time a hundred times a day now. Give me your watch. I want to see how it looks on it."
　　Instead of obeying, Jim tumbled down on the couch and put his hands under the back of his head and smiled.
　　"Dell," said he, "let's put our Christmas presents away and keep 'em a while. They're too nice to use just at present. I sold the watch to get the money to buy your combs. And now suppose you put the chops on."
　　The magi, as you know, were wise men--wonderfully wise men--who brought gifts to the Babe in the manger. They invented the art of giving Christmas presents. Being wise, their gifts were no doubt wise ones, possibly bearing the privilege of exchange in case of duplication. And here I have lamely related to you the uneventful chronicle of two foolish children in a flat who most unwisely sacrificed for each other the greatest treasures of their house. But in a last word to the wise of these days let it be said that of all who give gifts these two were the wisest. O all who give and receive gifts, such as they are wisest. Everywhere they are wisest. They are the magi.)";
