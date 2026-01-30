#include "base64.hpp"
#include "llama.h"
#include "pipo_op_perf.h"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>
using namespace std;

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

static std::vector<std::string> escape_patterns_manual(const std::vector<std::string>& patterns) {
    std::vector<std::string> escaped_patterns;
    escaped_patterns.reserve(patterns.size());
    
    // 正则表达式特殊字符集合
    static const std::unordered_set<char> special_chars = {
        '.', '*', '+', '?', '^', '$',
        '{', '}', '[', ']', '(', ')',
        '|', '\\'
    };
    
    for (const auto& pattern : patterns) {
        std::string escaped;
        escaped.reserve(pattern.size() * 2 + 2);
        escaped.push_back('^'); 
        for (char c : pattern) {
            if (special_chars.count(c)) {
                escaped.push_back('\\');
            }
            escaped.push_back(c);
        }
        escaped.push_back('$');
        
        escaped_patterns.push_back(escaped);
    }
    
    return escaped_patterns;
}
/* tensor random utils
    refer to test-backend-ops.cpp
*/
#ifdef __EMSCRIPTEN__
#    define N_THREADS 1
#else
#    define N_THREADS std::thread::hardware_concurrency()
#endif
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <random>
#include <thread>

static void init_tensor_uniform(ggml_tensor * tensor,
                                float         min     = -1.0f,
                                float         max     = 1.0f,
                                int64_t       int_min = 0,
                                int64_t       int_max = 100) {
    size_t nels = ggml_nelements(tensor);

    // 处理整数类型
    if (tensor->type == GGML_TYPE_I8 || tensor->type == GGML_TYPE_I16 || tensor->type == GGML_TYPE_I32 ||
        tensor->type == GGML_TYPE_I64) {
        // 为整数类型创建对应大小的缓冲区
        size_t element_size = 0;
        switch (tensor->type) {
            case GGML_TYPE_I8:
                element_size = sizeof(int8_t);
                break;
            case GGML_TYPE_I16:
                element_size = sizeof(int16_t);
                break;
            case GGML_TYPE_I32:
                element_size = sizeof(int32_t);
                break;
            case GGML_TYPE_I64:
                element_size = sizeof(int64_t);
                break;
            default:
                break;
        }

        std::vector<uint8_t> data(nels * element_size);

        {
            // 并行初始化整数
            static const size_t                            n_threads  = N_THREADS;
            static std::vector<std::default_random_engine> generators = []() {
                std::random_device                      rd;
                std::vector<std::default_random_engine> vec;
                vec.reserve(n_threads);
                for (size_t i = 0; i < n_threads; i++) {
                    vec.emplace_back(rd());
                }
                return vec;
            }();

            auto init_thread = [&](size_t ith, size_t start, size_t end) {
                // 使用int64_t生成随机整数，然后转换为目标类型
                std::uniform_int_distribution<int64_t> distribution(int_min, int_max);
                auto &                                 gen = generators[ith];

                switch (tensor->type) {
                    case GGML_TYPE_I8:
                        {
                            int8_t * ptr = reinterpret_cast<int8_t *>(data.data());
                            for (size_t i = start; i < end; i++) {
                                ptr[i] = static_cast<int8_t>(distribution(gen));
                            }
                            break;
                        }
                    case GGML_TYPE_I16:
                        {
                            int16_t * ptr = reinterpret_cast<int16_t *>(data.data());
                            for (size_t i = start; i < end; i++) {
                                ptr[i] = static_cast<int16_t>(distribution(gen));
                            }
                            break;
                        }
                    case GGML_TYPE_I32:
                        {
                            int32_t * ptr = reinterpret_cast<int32_t *>(data.data());
                            for (size_t i = start; i < end; i++) {
                                ptr[i] = static_cast<int32_t>(distribution(gen));
                            }
                            break;
                        }
                    case GGML_TYPE_I64:
                        {
                            int64_t * ptr = reinterpret_cast<int64_t *>(data.data());
                            for (size_t i = start; i < end; i++) {
                                ptr[i] = distribution(gen);
                            }
                            break;
                        }
                    default:
                        break;
                }
            };

            if (n_threads == 1) {
                init_thread(0, 0, nels);
            } else {
                std::vector<std::future<void>> tasks;
                tasks.reserve(n_threads);
                for (size_t i = 0; i < n_threads; i++) {
                    size_t start = i * nels / n_threads;
                    size_t end   = (i + 1) * nels / n_threads;
                    tasks.push_back(std::async(std::launch::async, init_thread, i, start, end));
                }
                for (auto & t : tasks) {
                    t.get();
                }
            }
        }

        // 设置张量数据
        ggml_backend_tensor_set(tensor, data.data(), 0, data.size());
    }
    // 处理浮点数类型
    else if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> data(nels);
        {
            // parallel initialization
            static const size_t                            n_threads  = N_THREADS;
            static std::vector<std::default_random_engine> generators = []() {
                std::random_device                      rd;
                std::vector<std::default_random_engine> vec;
                vec.reserve(n_threads);
                for (size_t i = 0; i < n_threads; i++) {
                    vec.emplace_back(rd());
                }
                return vec;
            }();

            auto init_thread = [&](size_t ith, size_t start, size_t end) {
                std::uniform_real_distribution<float> distribution(min, max);
                auto &                                gen = generators[ith];
                for (size_t i = start; i < end; i++) {
                    data[i] = distribution(gen);
                }
            };

            if (n_threads == 1) {
                init_thread(0, 0, nels);
            } else {
                std::vector<std::future<void>> tasks;
                tasks.reserve(n_threads);
                for (size_t i = 0; i < n_threads; i++) {
                    size_t start = i * nels / n_threads;
                    size_t end   = (i + 1) * nels / n_threads;
                    tasks.push_back(std::async(std::launch::async, init_thread, i, start, end));
                }
                for (auto & t : tasks) {
                    t.get();
                }
            }
        }
        ggml_backend_tensor_set(tensor, data.data(), 0, nels * sizeof(float));
    }
    // 处理量化类型
    else if (ggml_is_quantized(tensor->type) || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) {
        GGML_ASSERT(nels % ggml_blck_size(tensor->type) == 0);

        std::vector<float> data(nels);
        {
            // parallel initialization
            static const size_t                            n_threads  = N_THREADS;
            static std::vector<std::default_random_engine> generators = []() {
                std::random_device                      rd;
                std::vector<std::default_random_engine> vec;
                vec.reserve(n_threads);
                for (size_t i = 0; i < n_threads; i++) {
                    vec.emplace_back(rd());
                }
                return vec;
            }();

            auto init_thread = [&](size_t ith, size_t start, size_t end) {
                std::uniform_real_distribution<float> distribution(min, max);
                auto &                                gen = generators[ith];
                for (size_t i = start; i < end; i++) {
                    data[i] = distribution(gen);
                }
            };

            if (n_threads == 1) {
                init_thread(0, 0, nels);
            } else {
                std::vector<std::future<void>> tasks;
                tasks.reserve(n_threads);
                for (size_t i = 0; i < n_threads; i++) {
                    size_t start = i * nels / n_threads;
                    size_t end   = (i + 1) * nels / n_threads;
                    tasks.push_back(std::async(std::launch::async, init_thread, i, start, end));
                }
                for (auto & t : tasks) {
                    t.get();
                }
            }
        }

        // 量化处理
        std::vector<float> imatrix(tensor->ne[0], 1.0f);
        const float *      im = imatrix.data();
        if (!ggml_quantize_requires_imatrix(tensor->type)) {
            if (data[0] > 0.5f * (min + max)) {
                im = nullptr;
            }
        }

        std::vector<uint8_t> dataq(ggml_row_size(tensor->type, nels));
        {
            // parallel quantization by block
            size_t blck_size = ggml_blck_size(tensor->type);
            size_t n_blocks  = nels / blck_size;

            auto quantize_thread = [&](size_t start, size_t end) {
                ggml_quantize_chunk(tensor->type, data.data(), dataq.data(), start * blck_size, end - start, blck_size,
                                    im);
            };

            const size_t min_blocks_per_thread = 1;
            const size_t n_quant_threads       = std::min<size_t>(std::max<size_t>(N_THREADS / 2, 1),
                                                                  std::max<size_t>(1, n_blocks / min_blocks_per_thread));

            if (n_quant_threads == 1) {
                quantize_thread(0, n_blocks);
            } else {
                std::vector<std::future<void>> tasks;
                tasks.reserve(n_quant_threads);
                for (size_t i = 0; i < n_quant_threads; i++) {
                    size_t start = i * n_blocks / n_quant_threads;
                    size_t end   = (i + 1) * n_blocks / n_quant_threads;
                    tasks.push_back(std::async(std::launch::async, quantize_thread, start, end));
                }
                for (auto & t : tasks) {
                    t.get();
                }
            }
        }
        ggml_backend_tensor_set(tensor, dataq.data(), 0, dataq.size());
    } else {
        GGML_ABORT("Unsupported tensor type in init_tensor_uniform");
    }
}

/* single test result */
struct SingleTestResult {
    const pipo_unique_op & op;
    ggml_backend_t         backend;
    double                 compute_ms;
};

static double run_single_bench(const pipo_unique_op & op, ggml_backend_t backend, int n_iter) {
    size_t                       ctx_size    = 1024 * 1024 * 64;
    struct ggml_init_params      init_params = { ctx_size, NULL, true };
    struct ggml_context *        ctx         = ggml_init(init_params);
    // 2. 创建 src tensors
    vector<struct ggml_tensor *> src_tensors;
    src_tensors.resize(op.src_types.size());
    for (size_t i = 0; i < op.src_types.size(); i++) {
        src_tensors[i] = ggml_new_tensor(ctx, op.src_types[i], op.src_nes[i].size(), op.src_nes[i].data());
    }

    // 3. 创建 result tensor
    struct ggml_tensor * result = ggml_new_tensor(ctx, op.node_type, op.op_shape.size(), op.op_shape.data());

    // 4. 构建计算图
    result->op = op.op_type;
    for (size_t i = 0; i < src_tensors.size(); i++) {
        result->src[i] = src_tensors[i];
    }
    for (size_t i = src_tensors.size(); i < GGML_MAX_SRC; i++) {
        result->src[i] = NULL;
    }
    memcpy(result->op_params, op.op_param_bytes.data(), op.op_param_bytes.size());
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);

    if (!ggml_backend_supports_op(backend, result)) {
        cerr << "op " << op.short_desc() << " not supported by backend " << ggml_backend_name(backend) << '\n';
        ggml_free(ctx);
        return -1.0;
    }

    // 5. 后端分配
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        cerr << "Failed to allocate buffer for " << op.op_type << '\n';
        ggml_free(ctx);
        return -1.0;
    }

    // GGML_OP_GET_ROWS 需要特殊处理, 要防止下标越界
    if (op.op_type == GGML_OP_GET_ROWS) {
        init_tensor_uniform(src_tensors.at(0));
        init_tensor_uniform(src_tensors.at(1), 0, 0, 0, src_tensors.at(0)->ne[1] - 1);
    } else {
        for (size_t i = 0; i < src_tensors.size(); i++) {
            init_tensor_uniform(src_tensors.at(i));
        }
    }

    // warmup
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);

    // 6. 执行计算图
    int64_t t_compute_start = ggml_time_us();
    for (int i = 0; i < n_iter; i++) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);

    int64_t t_compute_end = ggml_time_us();
    double  compute_ms    = (t_compute_end - t_compute_start) / 1000.0 / n_iter;

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return compute_ms;
}

static vector<string> override_stratagy(const pipo_graph_info * i, size_t free_mem) {
    // 认为 中间节点(batch_size未知) + kvcache 的大小不超过整体的 10%？
    free_mem = free_mem * 0.90;

    vector<pair<size_t, string>> arr;
    size_t                       total_size = 0;
    arr.reserve(i->weight_sizes.size());
    for (auto & [name, size] : i->weight_sizes) {
        arr.emplace_back(make_pair(size, name));
        total_size += size;
    }
    size_t need_override = total_size - free_mem;
    // cerr << "free_mem = " << free_mem << "; total_size = " << total_size << "; " << "need_override = " << need_override
    //      << "\n";

    sort(arr.begin(), arr.end(), [](const pair<size_t, string> & a, const pair<size_t, string> & b) {
        // tensor name 的字典序，如果大小相同会把靠前的 tensor 放到主存上
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    });

    vector<string> result;
    for (auto & [size, name] : arr) {
        result.push_back(name);
        if (size > need_override) {
            break;
        }
        need_override -= size;
    }
    return result;
}

static vector<string> offload_stratgy(const pipo_graph_info *  i, const unordered_map<string, unordered_map<string, double>>& op_perf_result, const char* cpu_backend_name_c, const char* gpu_backend_name_c, ggml_backend_t gpu_backend) {
    // 参数，设与 host -> cuda 并行的 cpu 计算会慢 alpha 倍
    double alpha = 1.0;
    // 设由于与cpu计算并发传输慢了多少
    double belta = 3;

    const string cpu_backend_name(cpu_backend_name_c);
    const string gpu_backend_name(gpu_backend_name_c);

    // 用于构造 ggml_tensor
    size_t                       ctx_size    = 1024 * 1024 * 64;
    struct ggml_init_params      init_params = { ctx_size, NULL, true };
    struct ggml_context *        ggml_ctx         = ggml_init(init_params);


    vector<string> offload_weights;
    double time = 0;
    for (auto& [tn, cur_node, mid_nodes]: i->override_tensors_interval){
        for (auto& node : mid_nodes){
            if (!op_perf_result.count(gpu_backend_name)){
                fprintf(stderr, "%s: cpu backend result isn't initilized.", __func__);
                continue;
            }
            if (!op_perf_result.at(gpu_backend_name).count(node)){
                fprintf(stderr, "%s: cpu not foune op key %s\n", __func__, node.c_str());
                continue;
            }
            time += op_perf_result.at(gpu_backend_name).at(node);
        } 
        double transfer_time = (double)i->weight_sizes.at(tn) / i->h2d_bandwidth * belta;

        fprintf(stderr, "Considering offload tensor %s [%lf MB]\nestimated transfer time = %lf ms, estimated async calculation time = %lf ms\n", tn.c_str(), (double)i->weight_sizes.at(tn) / 1024 / 1024, transfer_time, time);

        if (ggml_backend_supports_op(gpu_backend, pipo_unique_op(cur_node).to_tensor(ggml_ctx)) && time > transfer_time){
            offload_weights.push_back(tn);
            fprintf(stderr, "[offload]: %s\n", tn.c_str());
            time = 0;
        }
        else {
            if (!op_perf_result.count(cpu_backend_name)){
                fprintf(stderr, "%s: cpu backend result isn't initilized.", __func__);
                continue;
            }
            if (!op_perf_result.at(cpu_backend_name).count(cur_node)){
                fprintf(stderr, "%s: cpu not found op key %s\n", __func__, cur_node.c_str());
                continue;
            }
            time += op_perf_result.at(cpu_backend_name).at(cur_node) * alpha;
        }
    }
    return offload_weights;
}

/* main */
int main(int argc, char ** argv) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " -m <model>\n";
        return 1;
    }
    const char * model_path = argv[2];
    // load backends
    ggml_backend_load_all();
    // load model
    llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap           = true;
    llama_model * model             = llama_model_load_from_file(model_path, model_params);

    if (model == NULL) {
        cerr << __LINE__ << ": Failed to load model\n";
        return 1;
    }

    // initialize context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx                = 1;
    ctx_params.n_batch              = 1;
    ctx_params.no_perf              = true;

    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        cerr << __LINE__ << ": Failed to create llama_context\n";
        return 1;
    }

    auto graph_info = pipo_get_graph_info(ctx);

    // 为测试清理内存
    llama_free(ctx);
    llama_model_free(model);
    ctx = NULL;
    model = NULL;

    ggml_backend_t cpu_backend = ggml_backend_init_by_name("cpu", NULL);
    ggml_backend_t gpu_backend = NULL;
    size_t         dev_count   = ggml_backend_dev_count();
    for (size_t i = 0; i < dev_count; ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu_backend = ggml_backend_dev_init(d, NULL);
            break;
        }
    }
    if (gpu_backend == NULL) {
        cerr << __LINE__ << ": GPU backend not found\n";
        return 1;
    }
    auto &                                               ops = graph_info->unique_ops;
    unordered_map<string, unordered_map<string, double>> op_perf_results;
    const char *                                         cpu_backend_name = ggml_backend_name(cpu_backend);
    op_perf_results[cpu_backend_name]                                     = unordered_map<string, double>();

    const char * gpu_backend_name     = ggml_backend_name(gpu_backend);
    op_perf_results[gpu_backend_name] = unordered_map<string, double>();

    for (auto & op : ops) {
        cerr << "perf op: " << op.short_desc() << '\n' << "key = " << op.op_key() << "\n\n";
        op_perf_results[cpu_backend_name][op.op_key()] = run_single_bench(op, cpu_backend, 20);
        op_perf_results[gpu_backend_name][op.op_key()] = run_single_bench(op, gpu_backend, 50);
    }
    // test cpu -> gpu bandwidth

    size_t free_memory;
    {
        ggml_backend_dev_t dev = ggml_backend_get_device(gpu_backend);
        size_t             _;
        ggml_backend_dev_memory(dev, &free_memory, &_);
        if (free_memory < 128 * 1024 * 1024) {
            cerr << "Free memory on GPU is less than 128 MB, quit test\n";
            return 1;
        }
        ggml_init_params      init_params = { 1024 * 1024 * 10, NULL, true };
        ggml_context *        ctx         = ggml_init(init_params);
        size_t                tensor_size = 128 * 1024 * 1024;
        ggml_tensor *         gpu_tensor  = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, tensor_size);
        vector<uint8_t>       host_data(tensor_size);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, gpu_backend);
        if (!buffer) {
            cerr << __FILE__ << "[" << __LINE__ << "]: Failed to allocate buffer for GPU\n";
            return 1;
        }
        // warm up
        ggml_backend_tensor_set(gpu_tensor, host_data.data(), 0, tensor_size);
        ggml_backend_synchronize(gpu_backend);
        double transfer_time = 0;
        for (int i = 0; i < 5; i++) {
            int64_t t_start = ggml_time_us();
            ggml_backend_tensor_set(gpu_tensor, host_data.data(), 0, tensor_size);
            ggml_backend_synchronize(gpu_backend);
            int64_t t_end = ggml_time_us();
            transfer_time += (t_end - t_start);
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        transfer_time  = transfer_time / 5 / 1000;
        graph_info->h2d_bandwidth = (double) tensor_size / transfer_time;
    }

    vector<string> override_list = override_stratagy(graph_info, free_memory);
    
    fprintf(stderr, "override list\n[");
    for (auto& tn : override_list){
        fprintf(stderr, "%s, ", tn.c_str());
    }
    fprintf(stderr, "]\n");

    unordered_set<string> override_set(override_list.begin(), override_list.end());


    // reinitialize context and model
    model             = llama_model_load_from_file(model_path, model_params);

    if (model == NULL) {
        cerr << __LINE__ << ": Failed to load model\n";
        return 1;
    }
    ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        cerr << __LINE__ << ": Failed to create llama_context\n";
        return 1;
    }

    auto graph_info2 = pipo_get_graph_info(ctx, &override_set);
    llama_free(ctx);
    llama_model_free(model);

    graph_info2->h2d_bandwidth = graph_info->h2d_bandwidth;
    graph_info2->weight_sizes = std::move(graph_info->weight_sizes);

    vector<string> offload_list = offload_stratgy(graph_info2, op_perf_results, cpu_backend_name, gpu_backend_name, gpu_backend);
    
    auto override_list_regex = escape_patterns_manual(override_list);
    auto offload_list_regex = escape_patterns_manual(offload_list);
    // output json result
    nlohmann::json j;
    j["overrides"] = override_list_regex;
    j["offloads"]  = offload_list_regex;

    cout << j.dump(4);

    ggml_backend_free(cpu_backend);
    if (gpu_backend) {
        ggml_backend_free(gpu_backend);
    }
    delete graph_info;
    delete graph_info2;
    return 0;
}
