#include "llama.h"
#include "pipo_op_perf.h"

#include <fcntl.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using namespace std;

/*
    TODO: reserve cuda mem base on graph alloc and kv cache alloc instead of guessing
*/
static std::vector<std::string> escape_patterns_manual(const std::vector<std::string> & patterns) {
    std::vector<std::string> escaped_patterns;
    escaped_patterns.reserve(patterns.size());

    // 正则表达式特殊字符集合
    static const std::unordered_set<char> special_chars = { '.', '*', '+', '?', '^', '$', '{',
                                                            '}', '[', ']', '(', ')', '|', '\\' };

    for (const auto & pattern : patterns) {
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

/* single test result */
struct SingleTestResult {
    const pipo_unique_op & op;
    ggml_backend_t         backend;
    double                 compute_ms;
};

static vector<string> override_stratagy(const pipo_graph_info * i, size_t free_mem) {
    free_mem = free_mem * 0.8;

    vector<pair<size_t, string>> arr;
    size_t                       total_size = 0;
    arr.reserve(i->weight_sizes.size());
    for (auto & [name, size] : i->weight_sizes) {
        arr.emplace_back(make_pair(size, name));
        total_size += size;
    }
    size_t need_override = total_size - free_mem;

    sort(arr.begin(), arr.end(),
         [](const pair<size_t, string> & a, const pair<size_t, string> & b) { return a.first > b.first; });
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

static vector<string> offload_stratgy(const pipo_graph_info *                                      i,
                                      const unordered_map<string, unordered_map<string, double>> & op_perf_result,
                                      const char *                                                 cpu_backend_name_c,
                                      const char *                                                 gpu_backend_name_c,
                                      ggml_backend_t                                               gpu_backend, double h2d_bandwidth) {
    // 参数，设与 host -> cuda 并行的 cpu 计算会慢 alpha 倍
    double alpha = 2;
    // 设由于与cpu计算并发传输慢了多少
    double belta = 1.5;

    const string cpu_backend_name(cpu_backend_name_c);
    const string gpu_backend_name(gpu_backend_name_c);

    // 用于构造 ggml_tensor
    size_t                  ctx_size    = 1024 * 1024 * 64;
    struct ggml_init_params init_params = { ctx_size, NULL, true };
    struct ggml_context *   ggml_ctx    = ggml_init(init_params);

    vector<string> offload_weights;
    double         time           = 0;
    bool           last_offloaded = false;
    for (auto & [tn, cur_node, mid_nodes] : i->override_tensors_interval) {
        if (!last_offloaded) {
            double tmp = 0;
            for (auto & node : mid_nodes) {
                if (!op_perf_result.count(gpu_backend_name)) {
                    fprintf(stderr, "%s: gpu backend result isn't initilized.", __func__);
                    continue;
                }
                if (!op_perf_result.at(gpu_backend_name).count(node)) {
                    fprintf(stderr, "%s: gpu do not found op key %s\n", __func__, node.c_str());
                    continue;
                }
                tmp += op_perf_result.at(gpu_backend_name).at(node);
            }
            fprintf(stderr, "gpu spent %lf ms\n", tmp);
            time += tmp;
        }
        double transfer_time = (double) i->weight_sizes.at(tn) / h2d_bandwidth * belta;

        fprintf(stderr,
                "Considering offload tensor %s [%lf MB]\nestimated transfer time = %lf ms, estimated async calculation "
                "time = %lf ms\n",
                tn.c_str(), (double) i->weight_sizes.at(tn) / 1024 / 1024, transfer_time, time);

        if (ggml_backend_supports_op(gpu_backend, pipo_unique_op(cur_node).to_tensor(ggml_ctx)) &&
            time > transfer_time) {
            offload_weights.push_back(tn);
            fprintf(stderr, "[offload]: %s\n", tn.c_str());
            time           = 0;
            last_offloaded = true;
        } else {
            if (!op_perf_result.count(cpu_backend_name)) {
                fprintf(stderr, "%s: cpu backend result isn't initilized.", __func__);
                continue;
            }
            if (!op_perf_result.at(cpu_backend_name).count(cur_node)) {
                fprintf(stderr, "%s: cpu not found op key %s\n", __func__, cur_node.c_str());
                continue;
            }
            double tmp = op_perf_result.at(cpu_backend_name).at(cur_node) * alpha;
            time += tmp;
            fprintf(stderr, "cpu spent %lf ms\n", tmp);
            last_offloaded = false;
        }
    }
    return offload_weights;
}

static void print_usage(int _, char ** argv) {
    cerr << "Usage: " << argv[0] << "<model> [-r <op_perf_json>]";
    (void) _;
}

int main(int argc, char ** argv) {
    const char * op_perf_result_path = "examples/pipo-alg/perf_result.json";
    const char * model_path          = nullptr;
    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-r") == 0) {
                if (i + 1 < argc) {
                    op_perf_result_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else {
                model_path = argv[i];
            }
        }
        if (model_path == nullptr || op_perf_result_path == nullptr) {
            print_usage(argc, argv);
            return 1;
        }
    }
    ifstream                                             _op_perf_result_s(op_perf_result_path, ios::in);
    nlohmann::json                                       _j              = nlohmann::json::parse(_op_perf_result_s);
    unordered_map<string, unordered_map<string, double>> op_perf_results = _j["op_perf_result"];
    double                                               h2d_bandwidth   = _j["h2d_bandwidth"];

    fprintf(stderr, "Fetching compute graph info, disable stderr\n");
    
    int _dev_null = open("/dev/null", O_WRONLY);
    int _stderr_fd = dup(STDERR_FILENO);
    dup2(_dev_null, STDERR_FILENO);
    // load backends
    ggml_backend_load_all();
    // load model
    llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap           = false;
    model_params.no_alloc           = true;
    llama_model * model             = llama_model_load_from_file(model_path, model_params);

    if (model == NULL) {
        cout << "[Error]" << __LINE__ << ": Failed to load model\n";
        return 1;
    }

    // initialize context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx                = 1;
    ctx_params.n_batch              = 1;
    ctx_params.no_perf              = true;

    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        cout << "[Error]" << __LINE__ << ": Failed to create llama_context\n";
        return 1;
    }
    dup2(_stderr_fd, STDERR_FILENO);
    close(_dev_null);

    auto graph_info = pipo_get_graph_info(ctx);

    ggml_backend_t cpu_backend = ggml_backend_init_by_name("cpu", NULL);
    ggml_backend_t gpu_backend = NULL;

    size_t dev_count = ggml_backend_dev_count();
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
    const char * cpu_backend_name = ggml_backend_name(cpu_backend);
    const char * gpu_backend_name = ggml_backend_name(gpu_backend);

    size_t             free_memory;
    ggml_backend_dev_t dev = ggml_backend_get_device(gpu_backend);
    size_t             _;
    ggml_backend_dev_memory(dev, &free_memory, &_);

    vector<string> override_list = override_stratagy(graph_info, free_memory);

    fprintf(stderr, "override list\n[");
    for (auto & tn : override_list) {
        fprintf(stderr, "%s, ", tn.c_str());
    }
    fprintf(stderr, "]\n");

    unordered_set<string> override_set(override_list.begin(), override_list.end());

    auto graph_info2 = pipo_get_graph_info(ctx, &override_set);
    llama_free(ctx);
    llama_model_free(model);

    graph_info2->weight_sizes = std::move(graph_info->weight_sizes);

    vector<string> offload_list =
        offload_stratgy(graph_info2, op_perf_results, cpu_backend_name, gpu_backend_name, gpu_backend, h2d_bandwidth);

    auto           override_list_regex = escape_patterns_manual(override_list);
    auto           offload_list_regex  = escape_patterns_manual(offload_list);
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
