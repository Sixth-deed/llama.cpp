#include "llama-model.h"
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

// 简易高性能位数组
struct BitArray {
    vector<uint64_t> data;
    size_t           size_bits;

    BitArray(size_t n) : size_bits(n) { data.resize((n + 63) / 64, 0); }

    void set(size_t idx, bool val) {
        size_t word_idx = idx >> 6;  // / 64
        size_t bit_idx  = idx & 63;  // % 64
        if (val) {
            data[word_idx] |= (1ULL << bit_idx);
        } else {
            data[word_idx] &= ~(1ULL << bit_idx);
        }
    }

    bool get(size_t idx) const {
        size_t word_idx = idx >> 6;
        size_t bit_idx  = idx & 63;
        return (data[word_idx] >> bit_idx) & 1ULL;
    }
};

static pair<vector<string>, vector<string>> dp_strategy(
    ggml_cgraph *                                                gf,
    const vector<pair<string, ggml_tensor*>>&                                          tensors_by_name,
    const unordered_map<string, unordered_map<string, double>> & op_perf_results,
    const char *                                                 _cpu_backend_name,
    const char *                                                 _gpu_backend_name,
    size_t                                                       free_mem,
    double                                                       h2d_bandwidth,
    const double                                                 alpha = 1.0,
    const double                                                 theta = 0.5) {
    const string cpu_name(_cpu_backend_name);
    const string gpu_name(_gpu_backend_name);

    // node_index
    using n_id                   = int;
    // weight_index
    using w_id                   = int;
    unordered_map<n_id, w_id> n2w;
    unordered_map<w_id, n_id> w2n;
    {
        unordered_map<string, w_id> w2i;
        for (w_id i = 0; i < (int) tensors_by_name.size(); i++) {
            w2i[tensors_by_name[i].first] = i;
        }
        for (n_id node_id = 0; node_id < ggml_graph_n_nodes(gf); node_id += 1) {
            ggml_tensor * t = ggml_graph_node(gf, node_id);
            for (n_id src_id = 0; src_id < GGML_MAX_SRC; src_id++) {
                if (t->src[src_id] == nullptr) {
                    break;
                }
                if (!w2i.count(string(t->src[src_id]->name))) {
                    continue;
                }
                w_id weight_id = w2i[string(t->src[src_id]->name)];
                n2w[node_id]   = weight_id;
                w2n[weight_id] = node_id;
            }
        }
    }

    auto gpu_compute_time = [&](n_id node_id) -> double {
        const ggml_tensor * t = ggml_graph_node(gf, node_id);
        if (pipo_is_view_op(t->op)) {
            return 0;
        }
        if (!op_perf_results.count(gpu_name) || !op_perf_results.at(gpu_name).count(pipo_make_op_key(t)) ||
            op_perf_results.at(gpu_name).at(pipo_make_op_key(t)) == -1) {
            return INFINITY;
        }
        return op_perf_results.at(gpu_name).at(pipo_make_op_key(t));
    };
    auto cpu_compute_time = [&](n_id node_id) -> double {
        const ggml_tensor * t = ggml_graph_node(gf, node_id);
        if (pipo_is_view_op(t->op)) {
            return 0;
        }
        if (!op_perf_results.count(cpu_name) || !op_perf_results.at(cpu_name).count(pipo_make_op_key(t)) ||
            op_perf_results.at(cpu_name).at(pipo_make_op_key(t)) == -1) {
            fprintf(stderr, "cpu not support op\n%s\n", pipo_make_op_key(t).c_str());
            return INFINITY;
        }
        return op_perf_results.at(cpu_name).at(pipo_make_op_key(t)) * alpha;
    };
    auto weight_size = [&](w_id weight_id) -> size_t {
        return ggml_nbytes(tensors_by_name[weight_id].second);
    };
    // TODO: 也许可以通过对tensors分组来减少彻底搜索的计算量

    // 单位是字节
    constexpr size_t mem_bin_size  = 1024 * 1024;
    // 单位是毫秒
    constexpr double time_bin_size = 0.2;
    auto             mem_bin       = [](size_t size) -> int {
        return size / mem_bin_size + ((size % mem_bin_size) > (mem_bin_size / 2));
    };
    auto time_bin = [](double time) -> int {
        return (int) std::round(time / time_bin_size);
    };
    const w_id weight_cnt = tensors_by_name.size();

    vector<double> mid_node_sum_C(weight_cnt, 0.0);
    vector<double> mid_node_sum_G(weight_cnt, 0.0);
    for (w_id i = 0; i < weight_cnt; i++) {
        n_id l = w2n[i] + 1;
        n_id r = i == weight_cnt - 1 ? ggml_graph_n_nodes(gf) : w2n[i + 1];
        for (n_id j = l; j < r; j++) {
            mid_node_sum_C[i] += cpu_compute_time(j);
            mid_node_sum_G[i] += gpu_compute_time(j);
        }
    }

    // weight transfer time bin
    vector<int> weight_tt_bin(weight_cnt);
    vector<int> weight_size_bin(weight_cnt);
    int         ttf_bin_cnt = 0;
    int         mem_bin_cnt = mem_bin(free_mem);
    for (w_id i = 0; i < weight_cnt; i++) {
        weight_tt_bin[i]   = time_bin((double) weight_size(i) / h2d_bandwidth);
        ttf_bin_cnt        = max(ttf_bin_cnt, weight_tt_bin[i]);
        weight_size_bin[i] = mem_bin(weight_size(i));
    }
    ttf_bin_cnt += 1;
    assert(ttf_bin_cnt < 65536);
    // cache
    vector<double> cpu_compute_time_cache(weight_cnt);
    vector<double> gpu_compute_time_cache(weight_cnt);
    for (w_id i = 0; i < weight_cnt; i++) {
        cpu_compute_time_cache[i] = cpu_compute_time(w2n[i]);
        gpu_compute_time_cache[i] = gpu_compute_time(w2n[i]);
    }

    fprintf(stderr, "[INFO] dp arr take %.4lf MB\n", (double) (ttf_bin_cnt * mem_bin_cnt * 8) / 1024.0 / 1024.0);
    fprintf(stderr, "[INFO] dp trace arr take %.4lf MB\n",
            (double) (weight_cnt * ttf_bin_cnt * mem_bin_cnt) * 4.375 / 1024.0 / 1024.0);

    const int W = weight_cnt;
    const int T = ttf_bin_cnt;
    const int M = mem_bin_cnt;

    auto idx_3d = [&](int w, int t, int m) -> size_t {
        return ((size_t) w * T + t) * M + m;
    };

    auto idx_2d = [&](int t, int m) -> size_t {
        return (size_t) t * M + m;
    };
    vector<double>   dp_G(T * M);
    vector<double>   dp_C(T * M);
    BitArray         next_on_gpu_C(W * T * M);
    BitArray         next_on_gpu_G(W * T * M);
    BitArray         offload(W * T * M);
    vector<uint16_t> next_ttf_C(W * T * M, 0);
    vector<uint16_t> next_ttf_G(W * T * M, 0);

    w_id       progress_interval       = weight_cnt / 30;
    const auto begin                   = ggml_time_ms();
    auto       print_dp_progress_debug = [&](w_id wid) {
        auto now = ggml_time_ms();
        fprintf(stderr, "finished %d/%d weights search, taken %.2lfs.\n", weight_cnt - wid, weight_cnt,
                      (double) (now - begin) / 1000.0);
        fprintf(stderr, "ttf: ");
        for (int i = 0; i < 10; i++) {
            fprintf(stderr, "%-5d ", i);
        }
        fprintf(stderr, "\n# C: ");
        for (int i = 0; i < 10; i++) {
            fprintf(stderr, "%-5.4lg ", dp_C[idx_2d(i, mem_bin_cnt - 1)]);
        }
        fprintf(stderr, "\n# G: ");
        for (int i = 0; i < 10; i++) {
            fprintf(stderr, "%-5.4lg ", dp_G[idx_2d(i, mem_bin_cnt - 1)]);
        }
        fprintf(stderr, "\n");
        fflush(stderr);
    };
    auto print_progress_bar = [&](int current) {
        auto      now       = ggml_time_ms();
        double    elapsed   = (double) (now - begin) / 1000.0;
        const int bar_width = 50;
        float     progress  = (float) current / weight_cnt;
        int       pos       = (int) (bar_width * progress);
        fprintf(stderr, "\r[");
        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) {
                fprintf(stderr, "=");
            } else if (i == pos && current < weight_cnt) {
                fprintf(stderr, ">");
            } else {
                fprintf(stderr, " ");
            }
        }
        fprintf(stderr, "] %d%% elapsed: %.2fs", (int) (progress * 100), elapsed);
        if (current == weight_cnt) {
            fprintf(stderr, "\n");
        }
        fflush(stderr);
    };
    (void) print_dp_progress_debug;
    (void) print_progress_bar;
    fprintf(stderr, "dp computation level: %.4lfe9\n", (double) (weight_cnt * mem_bin_cnt * ttf_bin_cnt) / 1e9);
    for (w_id wid = weight_cnt - 1; wid >= 0; wid--) {
        // 当前 weight 对应的 oprand 在 cpu 上的计算时间
        double           t_cC      = cpu_compute_time_cache[wid];
        // 当前 weight 对应的 oprand 在 gpu 上的计算时间
        double           t_cG      = gpu_compute_time_cache[wid];
        // 当前 weight 与下一个 weight 之间的 oprands 在 cpu 上的计算时间
        double           t_cmidC   = mid_node_sum_C[wid];
        // 当前 weight 与下一个 weight 之间的 oprands 在 gpu 上的计算时间
        double           t_cmidG   = mid_node_sum_G[wid];
        // 当前 weight 传输所需的时间
        int              b_t_curTf = weight_tt_bin[wid];
        // 当前 weight 占用内存大小
        int              b_curMem  = weight_size_bin[wid];
        // tensor on cpu, offload compute to gpu
        // 因为这玩意要用到完整的前一层的状态，所以第一个算
        vector<double>   offload_time(mem_bin_cnt, INFINITY);
        BitArray         offload_next_on_g(mem_bin_cnt);
        vector<uint16_t> offload_next_ttf(mem_bin_cnt, 0);
        // t_nft 这里表达的是下一个传输剩余的时间，没传完要罚时
        for (int t_nft = 0; t_nft < ttf_bin_cnt; t_nft++) {
            for (int mem = 0; mem < mem_bin_cnt; mem++) {
                double t_next_C      = dp_C[idx_2d(t_nft, mem)] + theta;
                double t_next_G      = dp_G[idx_2d(t_nft, mem)];
                bool   cur_next_on_G = t_next_C > t_next_G;
                double t_total       = t_cG + t_cmidG + t_nft * time_bin_size + min(t_next_C, t_next_G);
                if (t_total < offload_time[mem]) {
                    offload_time[mem] = t_total;
                    offload_next_on_g.set(mem, cur_next_on_G);
                    offload_next_ttf[mem] = t_nft;
                }
            }
        }

        int ct_G_bin = time_bin(t_cG + t_cmidG);
        // 特判 transfer time = 0
        for (int mem = mem_bin_cnt - 1; mem >= 0; mem--) {
            // cur remain transfer time = 0 && cur on cpu
            bool   next_on_gpu = false;
            double t_nc        = INFINITY;
            int    t_ntf_min   = 0;
            for (int t_ntf = 0; t_ntf < min(time_bin(t_cC + t_cmidG + theta) + 1, ttf_bin_cnt); t_ntf++) {
                if (t_nc > t_cmidG + dp_G[idx_2d(t_ntf, mem)] + theta) {
                    t_nc        = t_cmidG + dp_G[idx_2d(t_ntf, mem)] + theta;
                    next_on_gpu = true;
                    t_ntf_min   = t_ntf;
                }
            }
            for (int t_ntf = 0; t_ntf < min(time_bin(t_cC + t_cmidC) + 1, ttf_bin_cnt); t_ntf++) {
                if (t_nc > t_cmidC + dp_C[idx_2d(t_ntf, mem)]) {
                    t_nc        = t_cmidC + dp_C[idx_2d(t_ntf, mem)];
                    next_on_gpu = false;
                    t_ntf_min   = t_ntf;
                }
            }
            dp_C[idx_2d(0, mem)] = t_cC + t_nc;
            next_on_gpu_C.set(idx_3d(wid, 0, mem), next_on_gpu);
            next_ttf_C[idx_3d(wid, 0, mem)] = t_ntf_min;
            // cur remain transfer time = 0 && cur on gpu
            t_nc                            = INFINITY;
            next_on_gpu                     = false;
            t_ntf_min                       = 0;
            if (mem < b_curMem) {
                dp_G[idx_2d(0, mem)] = INFINITY;
                next_on_gpu_G.set(idx_3d(wid, 0, mem), false);
                next_ttf_G[idx_3d(wid, 0, mem)] = 0;
                continue;
            }
            for (int t_ntf = 0; t_ntf < min(ct_G_bin + 1, ttf_bin_cnt); t_ntf++) {
                if (t_nc > t_cmidG + dp_G[idx_2d(t_ntf, mem - b_curMem)]) {
                    t_nc        = t_cmidG + dp_G[idx_2d(t_ntf, mem - b_curMem)];
                    next_on_gpu = true;
                    t_ntf_min   = t_ntf;
                }
                if (t_nc > t_cmidG + dp_C[idx_2d(t_ntf, mem - b_curMem)] + theta) {
                    t_nc        = t_cmidG + dp_C[idx_2d(t_ntf, mem - b_curMem)] + theta;
                    next_on_gpu = false;
                    t_ntf_min   = t_ntf;
                }
            }
            dp_G[idx_2d(0, mem)] = t_cG + t_nc;
            next_on_gpu_G.set(idx_3d(wid, 0, mem), next_on_gpu);
            next_ttf_G[idx_3d(wid, 0, mem)] = t_ntf_min;
        }
        for (int t_tf = 1; t_tf < ttf_bin_cnt; t_tf++) {
            for (int mem = mem_bin_cnt - 1; mem >= 0; mem--) {
                // cur on cpu
                int    b_nttf_nG = t_tf + time_bin(t_cC + t_cmidG);
                int    b_nttf_nC = t_tf + time_bin(t_cC + t_cmidC);
                double t_next_G  = b_nttf_nG >= ttf_bin_cnt ? INFINITY : t_cmidG + dp_G[idx_2d(b_nttf_nG, mem)] + theta;
                double t_next_C  = b_nttf_nC >= ttf_bin_cnt ? INFINITY : t_cmidC + dp_C[idx_2d(b_nttf_nC, mem)];

                dp_C[idx_2d(t_tf, mem)] = t_cC + min(t_next_C, t_next_G);
                next_on_gpu_C.set(idx_3d(wid, t_tf, mem), t_next_C > t_next_G);
                next_ttf_C[idx_3d(wid, t_tf, mem)] = min(t_next_C > t_next_G ? b_nttf_nG : b_nttf_nC, ttf_bin_cnt - 1);
                // cur on gpu
                if (t_tf >= ttf_bin_cnt - ct_G_bin) {
                    dp_G[idx_2d(t_tf, mem)] = INFINITY;
                    next_on_gpu_G.set(idx_3d(wid, t_tf, mem), false);
                    next_ttf_G[idx_3d(wid, t_tf, mem)] = 0;
                    continue;
                }
                if (mem < b_curMem) {
                    dp_G[idx_2d(t_tf, mem)] = INFINITY;
                    next_on_gpu_G.set(idx_3d(wid, t_tf, mem), false);
                    next_ttf_G[idx_3d(wid, t_tf, mem)] = 0;
                    continue;
                }
                t_next_C                = dp_C[idx_2d(t_tf + ct_G_bin, mem - b_curMem)] + theta;
                t_next_G                = dp_G[idx_2d(t_tf + ct_G_bin, mem - b_curMem)];
                dp_G[idx_2d(t_tf, mem)] = t_cG + mid_node_sum_G[wid] + min(t_next_C, t_next_G);
                next_on_gpu_G.set(idx_3d(wid, t_tf, mem), t_next_C > t_next_G);
                next_ttf_G[idx_3d(wid, t_tf, mem)] = t_tf + ct_G_bin;
                // offload cur
                if (t_tf == b_t_curTf && dp_G[idx_2d(t_tf, mem)] > offload_time[mem]) {
                    dp_G[idx_2d(t_tf, mem)] = offload_time[mem];
                    next_on_gpu_G.set(idx_3d(wid, t_tf, mem), offload_next_on_g.get(mem));
                    next_ttf_G[idx_3d(wid, t_tf, mem)] = offload_next_ttf[mem];
                    offload.set(idx_3d(wid, t_tf, mem), true);
                }
            }
        }
        if ((weight_cnt - wid) % progress_interval == 0) {
            // print_dp_progress_debug(wid);
            print_progress_bar(weight_cnt - wid);
        }
    }
    // print_dp_progress_debug(0);
    print_progress_bar(weight_cnt);
    // collect result
    vector<string> override_list;
    vector<string> offload_list;

    double min_time_total = INFINITY;
    int    b_ttf_min      = -1;
    bool   on_gpu_min     = false;
    for (int t_tf = 0; t_tf < ttf_bin_cnt; t_tf++) {
        double tf_punish = t_tf * time_bin_size;
        if (min_time_total > tf_punish + dp_G[idx_2d(t_tf, mem_bin_cnt - 1)]) {
            min_time_total = tf_punish + dp_G[idx_2d(t_tf, mem_bin_cnt - 1)];
            b_ttf_min      = t_tf;
            on_gpu_min     = true;
        }
        if (min_time_total > tf_punish + dp_C[idx_2d(t_tf, mem_bin_cnt - 1)]) {
            min_time_total = tf_punish + dp_C[idx_2d(t_tf, mem_bin_cnt - 1)];
            b_ttf_min      = t_tf;
            on_gpu_min     = false;
        }
    }

    fprintf(stderr, "dp alg estimated decode batch time = %.4lf\n", min_time_total);
    bool     cur_on_gpu = on_gpu_min;
    uint16_t cur_ttf    = b_ttf_min;
    int      cur_mem    = mem_bin_cnt - 1;

    auto op_short_desc = [&gf](n_id node_id) -> string {
        return pipo_unique_op(ggml_graph_node(gf, node_id)).short_desc();
    };

    for (w_id i = 0; i < weight_cnt; i++) {
        bool     next_on_gpu_val = false;
        uint16_t next_ttf_val    = 0;
        int      next_mem_val    = cur_mem;
        size_t   idx             = idx_3d(i, cur_ttf, cur_mem);
        if (cur_ttf >= ttf_bin_cnt || cur_mem >= mem_bin_cnt || cur_mem < 0) {
            fprintf(stderr, "when tracing weight %d, cur_ttf = %d, cur_mem = %d, out of range\n", i, cur_ttf, cur_mem);
            return { {}, {} };
        }
        if (!cur_on_gpu) {
            override_list.push_back(tensors_by_name[i].first);
            next_on_gpu_val = next_on_gpu_C.get(idx);
            next_ttf_val    = next_ttf_C[idx];

            fprintf(stderr, "%s ON CPU\nNode Op: %s\nEstimated time = %.4lf\nMidNodes: [\n", tensors_by_name[i].first.c_str(), op_short_desc(w2n[i]).c_str(), cpu_compute_time_cache[i]);
            for(n_id j = w2n[i] + 1; j < (i == weight_cnt - 1 ? ggml_graph_n_nodes(gf) :w2n[i + 1]); j++) {
                fprintf(stderr, "\t%s\n", op_short_desc(j).c_str());
            }
            fprintf(stderr, "]\nEstimated time = %.4lf\n\n", next_on_gpu_val ? mid_node_sum_G[i] :mid_node_sum_C[i]);
        } else {
            next_on_gpu_val = next_on_gpu_G.get(idx);
            next_ttf_val    = next_ttf_G[idx];
            next_mem_val    = cur_mem - weight_size_bin[i];
            if (offload.get(idx)) {
                next_mem_val = cur_mem;
                override_list.push_back(tensors_by_name[i].first);
                offload_list.push_back(tensors_by_name[i].first);
                fprintf(stderr, "## OFFLOAD ##\n");
            }

            fprintf(stderr, "%s ON GPU\nNode Op: %s\nEstimated time = %.4lf\nMidNodes: [\n", tensors_by_name[i].first.c_str(),op_short_desc(w2n[i]).c_str(), gpu_compute_time_cache[i]);
            for (n_id j = w2n[i] + 1; j < (i == weight_cnt - 1 ? ggml_graph_n_nodes(gf) :w2n[i + 1]); j++) {
                fprintf(stderr, "\t%s\n", op_short_desc(j).c_str());
            }
            fprintf(stderr, "]\nEstimated time = %.4lf\n\n", mid_node_sum_G[i]);
        }
        cur_on_gpu = next_on_gpu_val;
        cur_ttf    = next_ttf_val;
        cur_mem    = next_mem_val;
    }
    return { override_list, offload_list };
}

static void print_usage(int _, char ** argv) {
    cerr << "Usage: " << argv[0] << "<model> [-r <op_perf_json>]";
    (void) _;
}

int main(int argc, char ** argv) {
    const char * op_perf_result_path = "examples/pipo-alg/perf_result.json";
    const char * model_path          = nullptr;
    double       alpha               = 1.0;
    double       theta               = 0.5;
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
            } else if (strcmp(argv[i], "-alpha") == 0) {
                if (i + 1 < argc) {
                    alpha = atof(argv[++i]);
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-theta") == 0) {
                if (i + 1 < argc) {
                    theta = atof(argv[++i]);
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

    int _dev_null  = open("/dev/null", O_WRONLY);
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

    ggml_cgraph * gf = pipo_get_graph(ctx);

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
    free_memory = free_memory * 4 / 5;

    // 需要按照 graph node 顺序排序
    auto tensor_by_name = model->tensors_by_name;
    unordered_map<string, int> tensor_by_name_idx;
    for (int i = 0; i < (int)tensor_by_name.size(); ++i) {
        tensor_by_name_idx[tensor_by_name[i].first] = i;
    }
    unordered_map<ggml_tensor*, int> tensor_by_name_node_idx;
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++){
        ggml_tensor* node = ggml_graph_node(gf, i);
        for (ggml_tensor* t : node->src){
            if (!t) break;
            if (tensor_by_name_idx.count(t->name)){
                tensor_by_name_node_idx[tensor_by_name[tensor_by_name_idx[t->name]].second] = i;
            }
        }
    }
    sort(tensor_by_name.begin(), tensor_by_name.end(), [&](const auto& a, const auto& b) {
        return tensor_by_name_node_idx[a.second] < tensor_by_name_node_idx[b.second];
    });
    auto [override_list, offload_list] = dp_strategy(gf, tensor_by_name, op_perf_results, cpu_backend_name, gpu_backend_name,
                                                         free_memory, h2d_bandwidth, alpha, theta);
    llama_free(ctx);
    llama_model_free(model);

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
    return 0;
}