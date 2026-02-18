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

/* single test result */
struct SingleTestResult {
    const pipo_unique_op & op;
    ggml_backend_t         backend;
    double                 compute_ms;
};

static vector<string> override_stratagy(llama_model * model, size_t free_mem) {
    free_mem = free_mem * 0.8;

    vector<pair<size_t, string>> arr;
    size_t                       total_size = 0;
    arr.reserve(model->tensors_by_name.size());
    for (auto & [name, t] : model->tensors_by_name) {
        size_t sz = ggml_nbytes(t);
        arr.emplace_back(make_pair(sz, name));
        total_size += sz;
    }
    size_t need_override = total_size - free_mem;

    stable_sort(arr.begin(), arr.end(),
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

static vector<string> offload_stratgy(ggml_cgraph *                                                gf,
                                      const llama_model *                                          model,
                                      const unordered_set<string> &                                override_tensors,
                                      const unordered_map<string, unordered_map<string, double>> & op_perf_result,
                                      const char *                                                 cpu_backend_name_c,
                                      const char *                                                 gpu_backend_name_c,
                                      ggml_backend_t                                               gpu_backend,
                                      double                                                       h2d_bandwidth) {
    // 参数，设与 host -> cuda 并行的 cpu 计算会慢 alpha 倍
    double alpha = 2;
    // 设由于与cpu计算并发传输慢了多少
    double belta = 1.5;

    const string cpu_backend_name(cpu_backend_name_c);
    const string gpu_backend_name(gpu_backend_name_c);

    // tensor name -> tensor size map
    unordered_map<std::string, size_t> weight_sizes;
    weight_sizes.reserve(model->tensors_by_name.size());
    for (const auto & [name, t] : model->tensors_by_name) {
        weight_sizes[name] = ggml_nbytes(t);
    }

    // compute graph nodes between override weights
    std::vector<std::tuple<std::string, std::string, std::vector<std::string>>> override_tensors_interval;
    {
        auto interval_tensors = std::vector<std::string>();
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            if (!node || pipo_is_view_op(node->op)) {
                continue;
            }
            ggml_tensor * override_src = nullptr;
            for (ggml_tensor * src : node->src) {
                if (!src) {
                    break;
                }
                if (override_tensors.count(std::string(src->name))) {
                    override_src = src;
                }
            }
            if (!override_src) {
                interval_tensors.push_back(pipo_make_op_key(node));
            } else {
                override_tensors_interval.push_back(std::make_tuple(
                    std::string(override_src->name), pipo_make_op_key(node), std::move(interval_tensors)));
            }
        }
    }

    // 用于构造 ggml_tensor
    size_t                  ctx_size    = 1024 * 1024 * 64;
    struct ggml_init_params init_params = { ctx_size, NULL, true };
    struct ggml_context *   ggml_ctx    = ggml_init(init_params);

    vector<string> offload_weights;
    double         time           = 0;
    bool           last_offloaded = false;
    for (auto & [tn, cur_node, mid_nodes] : override_tensors_interval) {
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
        double transfer_time = (double) weight_sizes.at(tn) / h2d_bandwidth * belta;

        fprintf(stderr,
                "Considering offload tensor %s [%lf MB]\nestimated transfer time = %lf ms, estimated async calculation "
                "time = %lf ms\n",
                tn.c_str(), (double) weight_sizes.at(tn) / 1024 / 1024, transfer_time, time);

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

static pair<vector<string>, vector<string>> search_strategy(
    ggml_cgraph *                                                gf,
    const llama_model *                                          model,
    const unordered_map<string, unordered_map<string, double>> & op_perf_results,
    const char *                                                 _cpu_backend_name,
    const char *                                                 _gpu_backend_name,
    ggml_backend_t                                               gpu_backend,
    size_t                                                       free_mem,
    double                                                       h2d_bandwidth) {
    const string cpu_name(_cpu_backend_name);
    const string gpu_name(_gpu_backend_name);
    // 切换后端的惩罚
    const double theta = 0.1;

    const auto & tensors_by_name = model->tensors_by_name;
    // node_index
    using n_id                   = int;
    // weight_index
    using w_id                   = int;
    // mem_bin_id
    using b_id                   = int;
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
        return op_perf_results.at(cpu_name).at(pipo_make_op_key(t));
    };
    auto weight_size = [&](w_id weight_id) -> size_t {
        return ggml_nbytes(tensors_by_name[weight_id].second);
    };
    /* 
    note, but not actualy use this
    dfs[free_mem][weight_index][last_offload_node] = 
        min(
            // override cur weight backend to gpu
            dfs[free_mem - weight_size(weight_index)][weight_index - 1][last_offload_node] + cuda_compute_time(w2n[weight_index]),
            // keep it on cpu
            dfs[free_mem][weight_index - 1][last_offload_node] + cpu_compute_time(w2n[weight_index])
            // keep its buffer on cpu but offload calculation to gpu
            dfs[free_mem][weight_index - 1][last_offload_node] + max(max(0, transimit_estimate(weight_index) - compute_estimate(last_offload_node, w2n(weight_index))) , cuda_compute_time(w2n[weight_index]))
            ) */
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
    vector<vector<double>>       dp_G(ttf_bin_cnt, vector<double>(mem_bin_cnt, 0.0));
    vector<vector<double>>       dp_C(ttf_bin_cnt, vector<double>(mem_bin_cnt, 0.0));
    vector<vector<vector<bool>>> next_on_gpu_C(weight_cnt,
                                               vector<vector<bool>>(ttf_bin_cnt, vector<bool>(mem_bin_cnt, false)));

    vector<vector<vector<bool>>>     next_on_gpu_G(weight_cnt,
                                                   vector<vector<bool>>(ttf_bin_cnt, vector<bool>(mem_bin_cnt, false)));
    vector<vector<vector<bool>>>     offload(weight_cnt,
                                             vector<vector<bool>>(ttf_bin_cnt, vector<bool>(mem_bin_cnt, false)));
    vector<vector<vector<uint16_t>>> next_ttf_C(
        weight_cnt, vector<vector<uint16_t>>(ttf_bin_cnt, vector<uint16_t>(mem_bin_cnt, false)));

    vector<vector<vector<uint16_t>>> next_ttf_G(
        weight_cnt, vector<vector<uint16_t>>(ttf_bin_cnt, vector<uint16_t>(mem_bin_cnt, false)));

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
            fprintf(stderr, "%-5.4lg ", dp_C[i][mem_bin_cnt - 1]);
        }
        fprintf(stderr, "\n# G: ");
        for (int i = 0; i < 10; i++) {
            fprintf(stderr, "%-5.4lg ", dp_G[i][mem_bin_cnt - 1]);
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
        // we only transfre one tensor parallel with compute at one time, so post transimitting tensor blocks current tensor offloading
        // 因为这玩意要用到完整的前一层的状态，所以第一个算
        // 最终用在 dp_G[t_ft] 对应数组的更新上
        vector<double>   offload_time(mem_bin_cnt, INFINITY);
        vector<bool>     offload_next_on_g(mem_bin_cnt, false);
        vector<uint16_t> offload_next_ttf(mem_bin_cnt, 0);
        // t_nft 这里表达的是下一个传输剩余的时间，没传完要罚时
        for (int t_nft = 0; t_nft < ttf_bin_cnt; t_nft++) {
            for (int mem = 0; mem < mem_bin_cnt; mem++) {
                double t_next_C      = dp_C[t_nft][mem] + theta;
                double t_next_G      = dp_G[t_nft][mem];
                bool   cur_next_on_G = t_next_C > t_next_G;
                double t_total       = t_cG + t_cmidG + t_nft * time_bin_size + min(t_next_C, t_next_G);
                if (t_total < offload_time[mem]) {
                    offload_time[mem]      = t_total;
                    offload_next_on_g[mem] = cur_next_on_G;
                    offload_next_ttf[mem]  = t_nft;
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
                if (t_nc > t_cmidG + dp_G[t_ntf][mem] + theta) {
                    t_nc        = t_cmidG + dp_G[t_ntf][mem] + theta;
                    next_on_gpu = true;
                    t_ntf_min   = t_ntf;
                }
            }
            for (int t_ntf = 0; t_ntf < min(time_bin(t_cC + t_cmidC) + 1, ttf_bin_cnt); t_ntf++) {
                if (t_nc > t_cmidC + dp_C[t_ntf][mem]) {
                    t_nc        = t_cmidC + dp_C[t_ntf][mem];
                    next_on_gpu = false;
                    t_ntf_min   = t_ntf;
                }
            }
            dp_C[0][mem]               = t_cC + t_nc;
            next_on_gpu_C[wid][0][mem] = next_on_gpu;
            next_ttf_C[wid][0][mem]    = t_ntf_min;
            // cur remain transfer time = 0 && cur on gpu
            t_nc                       = INFINITY;
            next_on_gpu                = false;
            t_ntf_min                  = 0;
            if (mem < b_curMem) {
                dp_G[0][mem]               = INFINITY;
                next_on_gpu_G[wid][0][mem] = false;
                next_ttf_G[wid][0][mem]    = 0;
                continue;
            }
            for (int t_ntf = 0; t_ntf < min(ct_G_bin + 1, ttf_bin_cnt); t_ntf++) {
                if (t_nc > t_cmidG + dp_G[t_ntf][mem - b_curMem]) {
                    t_nc        = t_cmidG + dp_G[t_ntf][mem - b_curMem];
                    next_on_gpu = true;
                    t_ntf_min   = t_ntf;
                }
                if (t_nc > t_cmidG + dp_C[t_ntf][mem - b_curMem] + theta) {
                    t_nc        = t_cmidG + dp_C[t_ntf][mem - b_curMem] + theta;
                    next_on_gpu = false;
                    t_ntf_min   = t_ntf;
                }
            }
            dp_G[0][mem]               = t_cG + t_nc;
            next_on_gpu_G[wid][0][mem] = next_on_gpu;
            next_ttf_G[wid][0][mem]    = t_ntf_min;
        }
        for (int t_tf = 1; t_tf < ttf_bin_cnt; t_tf++) {
            for (int mem = mem_bin_cnt - 1; mem >= 0; mem--) {
                // cur on cpu
                int    b_nttf_nG = t_tf + time_bin(t_cC + t_cmidG);
                int    b_nttf_nC = t_tf + time_bin(t_cC + t_cmidC);
                double t_next_G  = b_nttf_nG >= ttf_bin_cnt ? INFINITY : t_cmidG + dp_G[b_nttf_nG][mem] + theta;
                double t_next_C  = b_nttf_nC >= ttf_bin_cnt ? INFINITY : t_cmidC + dp_C[b_nttf_nC][mem];

                dp_C[t_tf][mem]               = t_cC + min(t_next_C, t_next_G);
                next_on_gpu_C[wid][t_tf][mem] = t_next_C > t_next_G;
                next_ttf_C[wid][t_tf][mem]    = min(t_next_C > t_next_G ? b_nttf_nG : b_nttf_nC, ttf_bin_cnt - 1);
                // cur on gpu
                if (t_tf >= ttf_bin_cnt - ct_G_bin) {
                    dp_G[t_tf][mem]               = INFINITY;
                    next_on_gpu_G[wid][t_tf][mem] = false;
                    next_ttf_G[wid][t_tf][mem]    = 0;
                    continue;
                }
                if (mem < b_curMem) {
                    dp_G[t_tf][mem]               = INFINITY;
                    next_on_gpu_G[wid][t_tf][mem] = false;
                    next_ttf_G[wid][t_tf][mem]    = 0;
                    continue;
                }
                t_next_C                      = dp_C[t_tf + ct_G_bin][mem - b_curMem] + theta;
                t_next_G                      = dp_G[t_tf + ct_G_bin][mem - b_curMem];
                dp_G[t_tf][mem]               = t_cG + mid_node_sum_G[wid] + min(t_next_C, t_next_G);
                next_on_gpu_G[wid][t_tf][mem] = t_next_C > t_next_G;
                next_ttf_G[wid][t_tf][mem]    = t_tf + ct_G_bin;
                // offload cur
                if (t_tf == b_t_curTf && dp_G[t_tf][mem] > offload_time[mem]) {
                    dp_G[t_tf][mem]               = offload_time[mem];
                    next_on_gpu_G[wid][t_tf][mem] = offload_next_on_g[mem];
                    next_ttf_G[wid][t_tf][mem]    = offload_next_ttf[mem];
                    offload[wid][t_tf][mem]       = true;
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
        if (min_time_total > tf_punish + dp_G[t_tf][mem_bin_cnt - 1]) {
            min_time_total = tf_punish + dp_G[t_tf][mem_bin_cnt - 1];
            b_ttf_min      = t_tf;
            on_gpu_min     = true;
        }
        if (min_time_total > tf_punish + dp_C[t_tf][mem_bin_cnt - 1]) {
            min_time_total = tf_punish + dp_C[t_tf][mem_bin_cnt - 1];
            b_ttf_min      = t_tf;
            on_gpu_min     = false;
        }
    }

    fprintf(stderr, "dp alg estimated decode batch time = %.4lf\n", min_time_total);
    bool     cur_on_gpu = on_gpu_min;
    uint16_t cur_ttf    = b_ttf_min;
    int      cur_mem    = mem_bin_cnt - 1;

    for (w_id i = 0; i < weight_cnt; i++) {
        bool     next_on_gpu_val = false;
        uint16_t next_ttf_val    = 0;
        int      next_mem_val    = cur_mem;
        if (cur_ttf >= ttf_bin_cnt || cur_mem >= mem_bin_cnt || cur_mem < 0) {
            fprintf(stderr, "when tracing weight %d, cur_ttf = %d, cur_mem = %d, out of range\n", i, cur_ttf, cur_mem);
            return { {}, {} };
        }
        if (!cur_on_gpu) {
            override_list.push_back(tensors_by_name[i].first);
            next_on_gpu_val = next_on_gpu_C[i][cur_ttf][cur_mem];
            next_ttf_val    = next_ttf_C[i][cur_ttf][cur_mem];

        } else {
            next_on_gpu_val = next_on_gpu_G[i][cur_ttf][cur_mem];
            next_ttf_val    = next_ttf_G[i][cur_ttf][cur_mem];
            next_mem_val    = cur_mem - weight_size_bin[i];

            if (offload[i][cur_ttf][cur_mem]) {
                next_mem_val = cur_mem;

                override_list.push_back(tensors_by_name[i].first);
                offload_list.push_back(tensors_by_name[i].first);
            }
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

    auto [override_list, offload_list] = search_strategy(gf, model, op_perf_results, cpu_backend_name, gpu_backend_name,
                                                         gpu_backend, free_memory, h2d_bandwidth);
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

#if 0
pipo_graph_info * llama_context::pipo_get_graph_info(std::unordered_set<std::string>* override_tensors_ptr = nullptr)
{
    uint32_t n_tokens = 1;
    uint32_t n_seqs = 1;
    uint32_t n_outputs = 1;
    auto mctx = memory->init_full(); 
        
    LLAMA_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__, n_tokens, n_seqs, n_outputs);
    GGML_ASSERT(n_outputs >= 1);

    ggml_backend_sched_reset(sched.get());

    sched_reserve();

    llama_memory_breakdown_print(this);

    // when the scheduler is reset, we cannnot reuse the old graph, so we reset the previous graph result to prevent that
    gf_res_prev->reset();

    // store the n_outputs as it is, and restore it afterwards
    // TODO: not sure if needed, might simplify in the future by removing this
    const auto save_n_outputs = this->n_outputs;

    this->n_outputs = n_outputs;

    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    llama_ubatch ubatch = balloc.ubatch_reserve(n_tokens/n_seqs, n_seqs);

    // set one output token per sequence in order to activate all backend samplers
    std::vector<llama_seq_id> seq_ids(n_seqs);
    for (uint32_t i = 0; i < n_seqs; ++i) {
        seq_ids[i] = i;
        ubatch.n_seq_id[i] = 1;
        ubatch.seq_id[i] = &seq_ids[i];
        ubatch.output[i] = true;
    }

    auto * res = gf_res_reserve.get();

    auto gtype = cparams.enable_pipo?(n_tokens > 1? LLM_GRAPH_TYPE_DEFAULT_PREFILL:LLM_GRAPH_TYPE_DEFAULT_DECODE)
                                    :LLM_GRAPH_TYPE_DEFAULT;

    const auto gparams = graph_params(res, ubatch, mctx.get(), gtype);

    res->reset();


    auto * gf = model.build_graph(gparams);

    this->n_outputs = save_n_outputs;

    pipo_graph_info* result = new pipo_graph_info();

    if (override_tensors_ptr != nullptr){
        auto& override_tensors = *override_tensors_ptr;
        auto interval_tensors = std::vector<std::string>();
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            if (!node) continue;
            if (pipo_is_view_op(node->op)) continue;
            ggml_tensor* override_src = nullptr;
            for (ggml_tensor* src : node->src){
                if (!src) break;
                if (override_tensors.count(std::string(src->name))){
                    override_src = src;
                } 
            }
            if (!override_src){
                interval_tensors.push_back(pipo_make_op_key(node));
            }
            else{
                result->override_tensors_interval.push_back(std::make_tuple(std::string(override_src->name), pipo_make_op_key(node), std::move(interval_tensors)));
            }
        }  
    }
    else{
        // tensor info
        for (const auto& [tn, t] : model.tensors_by_name){
            result->weight_sizes[tn] = ggml_nbytes(t);
        }
        // op info
        auto& seen_ops = result->unique_ops;
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            if (!node) continue;

            if (pipo_is_view_op(node->op)) continue;
            pipo_unique_op op(node);
            if (!seen_ops.insert(op).second) continue;
            // const std::string key = op.op_key();
            // fprintf(stdout, "\nop_key[%zu]: ", key.size());
            // fwrite(key.data(), 1, key.size(), stdout);
        } 
    }
    return result;
}
#endif
