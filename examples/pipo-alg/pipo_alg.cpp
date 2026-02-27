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
    const vector<pair<string, ggml_tensor *>> &                  tensors_by_name,
    const unordered_map<string, unordered_map<string, double>> & op_perf_results,
    const char *                                                 _cpu_backend_name,
    const char *                                                 _gpu_backend_name,
    size_t                                                       free_mem,
    double                                                       h2d_bandwidth,
    // cpu 计算实际比 perf 结果会慢 alpha 倍
    const double                                                 alpha = 1.0,
    // 传输实际慢 belta 倍
    const double                                                 beta  = 1.0,
    // 切换后端的惩罚，单位为毫秒
    const double                                                 theta = 0.5) {
    const string cpu_name(_cpu_backend_name);
    const string gpu_name(_gpu_backend_name);

    // node_index
    using n_id = int;
    // weight_index
    using w_id = int;
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
    for (w_id i = 0; i < weight_cnt; i++) {
        weight_tt_bin[i]   = time_bin((double) weight_size(i) / h2d_bandwidth * beta);
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

    // 由于 mem_bin 是近似的，有可能算法实际给出的结果使用了过多的内存，需要重新跑一遍整个算法来获取有效的结果。
    size_t actual_mem_usage;
    // 稍微留一点余量，尽量不触发重跑
    free_mem -= 2 * mem_bin_size;
    int    iter     = 0;
    int    max_iter = 10;
    do {
        int       mem_bin_cnt = mem_bin(free_mem);
        const int W           = weight_cnt;
        const int T           = ttf_bin_cnt;
        const int M           = mem_bin_cnt;
        auto      idx_3d      = [&](int w, int t, int m) -> size_t {
            return ((size_t) w * T + t) * M + m;
        };

        auto idx_2d = [&](int t, int m) -> size_t {
            return (size_t) t * M + m;
        };
        fprintf(stderr, "[INFO] dp arr take %.4lf MB\n", (double) (ttf_bin_cnt * mem_bin_cnt * 8) / 1024.0 / 1024.0);
        fprintf(stderr, "[INFO] dp trace arr take %.4lf MB\n",
                (double) (weight_cnt * ttf_bin_cnt * mem_bin_cnt) * 4.375 / 1024.0 / 1024.0);
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
                    double t_next_G =
                        b_nttf_nG >= ttf_bin_cnt ? INFINITY : t_cmidG + dp_G[idx_2d(b_nttf_nG, mem)] + theta;
                    double t_next_C = b_nttf_nC >= ttf_bin_cnt ? INFINITY : t_cmidC + dp_C[idx_2d(b_nttf_nC, mem)];

                    dp_C[idx_2d(t_tf, mem)] = t_cC + min(t_next_C, t_next_G);
                    next_on_gpu_C.set(idx_3d(wid, t_tf, mem), t_next_C > t_next_G);
                    next_ttf_C[idx_3d(wid, t_tf, mem)] =
                        min(t_next_C > t_next_G ? b_nttf_nG : b_nttf_nC, ttf_bin_cnt - 1);
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
        actual_mem_usage    = 0;

        auto op_short_desc = [&gf](n_id node_id) -> string {
            return pipo_unique_op(ggml_graph_node(gf, node_id)).short_desc();
        };

        for (w_id i = 0; i < weight_cnt; i++) {
            bool     next_on_gpu_val = false;
            uint16_t next_ttf_val    = 0;
            int      next_mem_val    = cur_mem;
            size_t   idx             = idx_3d(i, cur_ttf, cur_mem);
            if (cur_ttf >= ttf_bin_cnt || cur_mem >= mem_bin_cnt || cur_mem < 0) {
                fprintf(stderr, "when tracing weight %d, cur_ttf = %d, cur_mem = %d, out of range\n", i, cur_ttf,
                        cur_mem);
                return { {}, {} };
            }
            if (!cur_on_gpu) {
                override_list.push_back(tensors_by_name[i].first);
                next_on_gpu_val = next_on_gpu_C.get(idx);
                next_ttf_val    = next_ttf_C[idx];

                fprintf(stderr, "%s ON CPU\nNode Op: %s\nEstimated time = %.4lf\nMidNodes: [\n",
                        tensors_by_name[i].first.c_str(), op_short_desc(w2n[i]).c_str(), cpu_compute_time_cache[i]);
                for (n_id j = w2n[i] + 1; j < (i == weight_cnt - 1 ? ggml_graph_n_nodes(gf) : w2n[i + 1]); j++) {
                    fprintf(stderr, "\t%s\n", op_short_desc(j).c_str());
                }
                fprintf(stderr, "]\nEstimated time = %.4lf\n\n",
                        next_on_gpu_val ? mid_node_sum_G[i] : mid_node_sum_C[i]);
            } else {
                next_on_gpu_val = next_on_gpu_G.get(idx);
                next_ttf_val    = next_ttf_G[idx];
                next_mem_val    = cur_mem - weight_size_bin[i];
                actual_mem_usage += weight_size(i);
                if (offload.get(idx)) {
                    next_mem_val = cur_mem;
                    actual_mem_usage -= weight_size(i);
                    override_list.push_back(tensors_by_name[i].first);
                    offload_list.push_back(tensors_by_name[i].first);
                    fprintf(stderr, "## OFFLOAD ##\n");
                }

                fprintf(stderr, "%s ON GPU\nNode Op: %s\nEstimated time = %.4lf\nMidNodes: [\n",
                        tensors_by_name[i].first.c_str(), op_short_desc(w2n[i]).c_str(), gpu_compute_time_cache[i]);
                for (n_id j = w2n[i] + 1; j < (i == weight_cnt - 1 ? ggml_graph_n_nodes(gf) : w2n[i + 1]); j++) {
                    fprintf(stderr, "\t%s\n", op_short_desc(j).c_str());
                }
                fprintf(stderr, "]\nEstimated time = %.4lf\n\n", mid_node_sum_G[i]);
            }
            cur_on_gpu = next_on_gpu_val;
            cur_ttf    = next_ttf_val;
            cur_mem    = next_mem_val;
        }
        if (actual_mem_usage <= free_mem) {
            return { override_list, offload_list };
        }
        size_t new_target_mem = free_mem - (actual_mem_usage - free_mem);
        fprintf(stderr,
                "alg actual provide strategy with mem use of %ld bytes, but target memory usage is %ld bytes.\nRerun "
                "alg with new target mem = %ld\n",
                actual_mem_usage, free_mem, new_target_mem);
        iter += 1;
    } while (iter < max_iter);
    fprintf(stderr, "alg failed to find a strategy with max_retires = %d\n", max_iter);
    return { {}, {} };
}

static pair<vector<string>, vector<string>> greedy_strategy(
    ggml_cgraph *                                                gf,
    const vector<pair<string, ggml_tensor *>> &                  tensors_by_name,
    const unordered_map<string, unordered_map<string, double>> & op_perf_results,
    const char *                                                 _cpu_backend_name,
    const char *                                                 _gpu_backend_name,
    size_t                                                       free_mem,
    double                                                       h2d_bandwidth,
    // 与传输并发的 cpu 计算慢 alpha 倍
    const double                                                 alpha = 1.0,
    // 与 cpu 计算并发的传输慢 belta 倍
    const double                                                 beta  = 1.0,
    // 调整优先级中 权重大小/计算时间收益 的比重，越大权重大小越重要
    const double                                                 theta = 1.0) {
    const string cpu_name(_cpu_backend_name);
    const string gpu_name(_gpu_backend_name);

    // node_index
    using n_id = int;
    // weight_index
    using w_id = int;
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

    unordered_set<string> must_override;

    auto override_pri = [&](w_id index) {
        double ct = cpu_compute_time(w2n[index]);
        double gt = gpu_compute_time(w2n[index]);
        if (gt < 0.0) {
            must_override.insert(string(ggml_graph_node(gf, w2n[index])->name));
        }
        return max(0.0, (ct - gt)) / pow((weight_size(index) / 1024.0 / 1024.0 / 8.0), theta);
    };
    vector<tuple<double, size_t, string>> arr;
    arr.resize(tensors_by_name.size());
    for (w_id i = 0; i < (int) tensors_by_name.size(); i++) {
        arr[i] = { override_pri(i), weight_size(i), tensors_by_name[i].first };
    }
    stable_sort(arr.begin(), arr.end(),
                [](const tuple<double, size_t, string> & l, const tuple<double, size_t, string> & r) {
                    return get<0>(l) > get<0>(r) || (get<0>(l) == get<0>(r) && get<1>(l) < get<1>(r));
                });

    for (auto & [pri, size, name] : arr) {
        fprintf(stderr, "[%s]: pri = %lf\tsize=%lf MB\n", name.c_str(), pri, (double) size / 1024.0 / 1024.0);
    }
    unordered_set<string> gpu_set;
    int                   override_index = 0;
    for (override_index = 0; override_index < (int) arr.size(); override_index++) {
        const auto & [pri, size, tn] = arr[override_index];
        if (free_mem < size) {
            break;
        }
        if (must_override.count(tn)) {
            continue;
        }
        gpu_set.insert(tn);
        free_mem -= size;
    }
    vector<string> override_list;
    for (const auto & [tn, size] : tensors_by_name) {
        if (!gpu_set.count(tn)) {
            override_list.push_back(tn);
        }
    }
    fprintf(stderr, "override_list = [");
    for (auto & tn : override_list) {
        fprintf(stderr, "\n\t%s,", tn.c_str());
    }
    fprintf(stderr, "\n]\n");

    // offload with dp
    vector<string> offload_list;
    {
        unordered_set<string> override_set(override_list.begin(), override_list.end());
        w_id                  weight_cnt    = tensors_by_name.size();
        auto                  transfer_time = [&](w_id weight_id) -> double {
            return (double) weight_size(weight_id) / h2d_bandwidth * beta;
        };
        // 每一个带权重节点自身计算时间与它之前的带权重节点间的节点计算时间之和
        vector<double> computation_internal(weight_cnt, 0);
        n_id           prev        = 0;
        bool           prev_on_gpu = false;
        for (w_id i = 0; i < weight_cnt; i++) {
            n_id cur        = w2n[i];
            bool cur_on_gpu = override_set.count(tensors_by_name[i].first);
            if (cur_on_gpu) {
                computation_internal[i] += gpu_compute_time(cur);
            } else {
                computation_internal[i] += cpu_compute_time(cur);
            }
            for (n_id j = prev; j < cur; j++) {
                if (cur_on_gpu || prev_on_gpu) {
                    computation_internal[i] += gpu_compute_time(j);
                } else {
                    computation_internal[i] += cpu_compute_time(j);
                }
            }
            prev        = cur;
            prev_on_gpu = cur_on_gpu;
        }

        auto computation_between = [&](w_id l, w_id r) -> double {
            double result = 0;
            for (w_id i = l + 1; i < r; i++) {
                result += computation_internal[i];
            }
            return result;
        };
        // offload tensor 带来的时间差，使总计算时间时间减少时为负。越小越好
        vector<double> offload_gain(weight_cnt, INFINITY);
        vector<w_id>   offload_prev(weight_cnt, -1);
        offload_gain[0] = gpu_compute_time(w2n[0]) + transfer_time(0) - cpu_compute_time(w2n[0]);
        for (w_id cur = 1; cur < weight_cnt; cur++) {
            if (!override_set.count(tensors_by_name[cur].first)) {
                continue;
            }
            offload_gain[cur] =
                gpu_compute_time(w2n[cur]) - cpu_compute_time(w2n[cur]) +
                max((alpha - 1) * transfer_time(cur), transfer_time(cur) - computation_between(-1, cur));
            for (w_id prev = 0; prev < cur; prev++) {
                if (!override_set.count(tensors_by_name[prev].first)) {
                    continue;
                }
                double offload_cur_gain =
                    offload_gain[prev] + gpu_compute_time(w2n[cur]) - cpu_compute_time(w2n[cur]) +
                    max((alpha - 1) * transfer_time(cur), transfer_time(cur) - computation_between(prev, cur));
#if 0
                fprintf(stderr, "cur = %d, prev = %d, gain = %lf compare to %lf\n", cur, prev, offload_cur_gain, offload_gain[cur]);
                if (strstr(tensors_by_name[cur].first.c_str(), "ffn_up") || strstr(tensors_by_name[cur].first.c_str(), "ffn_gate")){
                    fprintf(stderr, "%s: pgain=%.4lf, gt=%.4lf, ct=%.4lf, tt=%.4lf, it=%.4lf\n", tensors_by_name[cur].first.c_str(), offload_gain[prev],  gpu_compute_time(w2n[cur]), cpu_compute_time(w2n[cur]), 
                    transfer_time(cur), computation_between(prev, cur));
                }
#endif
                if (offload_cur_gain < offload_gain[cur]) {
                    offload_gain[cur] = offload_cur_gain;
                    offload_prev[cur] = prev;
                }
            }
        }
        double min_gain = INFINITY;
        w_id   min_last = -1;
        for (w_id i = 0; i < weight_cnt; i++) {
            if (min_gain > offload_gain[i]) {
                min_gain = offload_gain[i];
                min_last = i;
            }
        }
        while (min_last != -1) {
            offload_list.push_back(tensors_by_name[min_last].first);
            min_last = offload_prev[min_last];
        }
        fprintf(stderr, "offload estimate gain = %lf\n", min_gain);
    }
    return { std::move(override_list), std::move(offload_list) };
}

// 帕累托状态定义
struct ParetoState {
    double time_cost;          // 从当前层到结束的累计计算时间 (不含当前层之前的等待)
    size_t mem_usage;          // 从当前层到结束占用的显存
    double required_overlap;   // 需要前一层 (i-1) 提供多少计算时间来掩盖当前层的 H2D 传输
    int    decision;           // 0: CPU, 1: GPU, 2: Hybrid(GPU Mem + CPU Compute)
    int    parent_idx;         // 回溯用：上一层帕累托列表中的索引
    bool   prev_was_gpu;       // 回溯用：上一层是否在 GPU 计算 (用于计算切换惩罚)
    size_t offload_buffer_size; // offload buffer 的大小取决于 offload 的最大的 tensor 的大小(现在的实现暂时不是)
};

// 分桶剪枝
static std::vector<ParetoState> prune_pareto_bucketed(
    std::vector<ParetoState>& states, 
    size_t max_states = 10000,
    size_t mem_bucket_count = 1000,
    size_t overlap_bucket_count = 1000) {
    
    if (states.empty()) return states;
    
    // 1. 计算内存和 overlap 的范围
    size_t min_mem = SIZE_MAX, max_mem = 0;
    double min_overlap = INFINITY, max_overlap = 0;
    for (const auto& s : states) {
        min_mem = std::min(min_mem, s.mem_usage);
        max_mem = std::max(max_mem, s.mem_usage);
        min_overlap = std::min(min_overlap, s.required_overlap);
        max_overlap = std::max(max_overlap, s.required_overlap);
    }
    
    // 2. 创建分桶网格 [mem_bucket][overlap_bucket] -> 该桶内时间最优的状态索引
    std::vector<std::vector<int>> buckets(
        mem_bucket_count, 
        std::vector<int>(overlap_bucket_count, -1));
    
    auto get_mem_bucket = [&](size_t mem) -> int {
        if (max_mem == min_mem) return 0;
        int bucket = (int)((mem - min_mem) * mem_bucket_count / (max_mem - min_mem + 1));
        return std::min(bucket, (int)mem_bucket_count - 1);
    };
    
    auto get_overlap_bucket = [&](double overlap) -> int {
        if (max_overlap == min_overlap) return 0;
        int bucket = (int)((overlap - min_overlap) * overlap_bucket_count / (max_overlap - min_overlap + 1e-6));
        return std::min(bucket, (int)overlap_bucket_count - 1);
    };
    
    // 3. 将状态分配到桶中，每个桶只保留时间最优的
    for (int i = 0; i < (int)states.size(); i++) {
        int mb = get_mem_bucket(states[i].mem_usage);
        int ob = get_overlap_bucket(states[i].required_overlap);
        
        if (buckets[mb][ob] == -1 || states[i].time_cost < states[buckets[mb][ob]].time_cost) {
            buckets[mb][ob] = i;
        }
    }
    
    // 4. 收集所有桶的代表状态
    std::vector<ParetoState> result;
    result.reserve(mem_bucket_count * overlap_bucket_count);
    for (int mb = 0; mb < (int)mem_bucket_count; mb++) {
        for (int ob = 0; ob < (int)overlap_bucket_count; ob++) {
            if (buckets[mb][ob] != -1) {
                result.push_back(states[buckets[mb][ob]]);
            }
        }
    }
    
    // 5. 如果仍然超过上限，按时间排序截取 top K
    if (result.size() > max_states) {
        std::sort(result.begin(), result.end(), [](const ParetoState& a, const ParetoState& b) {
            return a.time_cost < b.time_cost;
        });
        result.resize(max_states);
    }
    
    return result;
}

static pair<vector<string>, vector<string>> dp_strategy_pareto(
    ggml_cgraph *                                                gf,
    const vector<pair<string, ggml_tensor *>> &                  tensors_by_name,
    const unordered_map<string, unordered_map<string, double>> & op_perf_results,
    const char *                                                 _cpu_backend_name,
    const char *                                                 _gpu_backend_name,
    size_t                                                       free_mem,
    double                                                       h2d_bandwidth,
    const double                                                 alpha = 1.0,
    const double                                                 beta  = 1.0,
    const double                                                 theta = 0.5) {
    
    const string cpu_name(_cpu_backend_name);
    const string gpu_name(_gpu_backend_name);
    using n_id = int;
    using w_id = int;
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
                if (t->src[src_id] == nullptr) break;
                if (!w2i.count(string(t->src[src_id]->name))) continue;
                w_id weight_id = w2i[string(t->src[src_id]->name)];
                n2w[node_id]   = weight_id;
                w2n[weight_id] = node_id;
            }
        }
    }

    auto gpu_compute_time = [&](n_id node_id) -> double {
        const ggml_tensor * t = ggml_graph_node(gf, node_id);
        if (pipo_is_view_op(t->op)) return 0;
        if (!op_perf_results.count(gpu_name) || !op_perf_results.at(gpu_name).count(pipo_make_op_key(t)) ||
            op_perf_results.at(gpu_name).at(pipo_make_op_key(t)) == -1) return INFINITY;
        return op_perf_results.at(gpu_name).at(pipo_make_op_key(t));
    };
    auto cpu_compute_time = [&](n_id node_id) -> double {
        const ggml_tensor * t = ggml_graph_node(gf, node_id);
        if (pipo_is_view_op(t->op)) return 0;
        if (!op_perf_results.count(cpu_name) || !op_perf_results.at(cpu_name).count(pipo_make_op_key(t)) ||
            op_perf_results.at(cpu_name).at(pipo_make_op_key(t)) == -1) return INFINITY;
        return op_perf_results.at(cpu_name).at(pipo_make_op_key(t)) * alpha;
    };
    auto weight_size = [&](w_id weight_id) -> size_t {
        return ggml_nbytes(tensors_by_name[weight_id].second);
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

    vector<double> cpu_compute_time_cache(weight_cnt);
    vector<double> gpu_compute_time_cache(weight_cnt);
    vector<double> weight_transfer_time(weight_cnt);
    for (w_id i = 0; i < weight_cnt; i++) {
        cpu_compute_time_cache[i] = cpu_compute_time(w2n[i]);
        gpu_compute_time_cache[i] = gpu_compute_time(w2n[i]);
        weight_transfer_time[i]   = (double) weight_size(i) / h2d_bandwidth * beta;
    }

    // DP 状态池：pareto_states[i] 存储第 i 层之后的所有帕累托最优状态
    vector<vector<ParetoState>> pareto_states(weight_cnt + 1);
    // 初始化最后一层之后 (End State)
    pareto_states[weight_cnt].push_back({0.0, 0, 0.0, -1, -1, false, 0});

    fprintf(stderr, "[INFO] Starting Pareto DP Search...\n");

    for (w_id wid = weight_cnt - 1; wid >= 0; wid--) {
        vector<ParetoState> current_candidates;
        // 预留空间，减少 realloc
        current_candidates.reserve(pareto_states[wid + 1].size() * 3); 

        double t_cC = cpu_compute_time_cache[wid];
        double t_cG = gpu_compute_time_cache[wid];
        double t_midC = mid_node_sum_C[wid];
        double t_midG = mid_node_sum_G[wid];
        double t_h2d = weight_transfer_time[wid];
        size_t w_mem = weight_size(wid);

        // 遍历上一层 (wid+1) 的所有帕累托状态
        for (int prev_idx = 0; prev_idx < (int)pareto_states[wid + 1].size(); prev_idx++) {
            const auto& prev = pareto_states[wid + 1][prev_idx];

            // --- 决策 1: 当前层在 CPU 计算 (不占显存，无需传输) ---
            {
                size_t new_mem = prev.mem_usage;
                if (new_mem <= free_mem) {
                    double new_time = (t_cC + t_midC) + prev.time_cost;
                    if (prev.prev_was_gpu){
                        new_time = (t_cC + t_midG) + prev.time_cost;
                    } 
                    double remaining_overlap_needed = std::max(0.0, prev.required_overlap - (new_time - prev.time_cost));
                    
                    current_candidates.push_back({
                        new_time, new_mem, remaining_overlap_needed, 
                        0, prev_idx, false, prev.offload_buffer_size // decision=0 (CPU), prev_was_gpu=false (for next iter)
                    });
                }
            }

            // --- 决策 2: 当前层在 GPU 计算 (占显存，不需传输) ---
            {
                size_t new_mem = prev.mem_usage + w_mem;
                if (new_mem <= free_mem) {
                    double new_time = (t_cG + t_midG) + prev.time_cost;
                    if (!prev.prev_was_gpu) new_time += theta; // CPU -> GPU 切换
                    double remaining_overlap_needed = std::max(0.0, prev.required_overlap - (new_time - prev.time_cost));

                    current_candidates.push_back({
                        new_time, new_mem, remaining_overlap_needed, 
                        1, prev_idx, true, prev.offload_buffer_size // decision=1 (GPU)
                    });
                }
            }

            // --- 决策 3: 当前层本身在主存上，运行时异步传输到在 GPU 计算 (几乎不占显存，需传输) ---
            {
                size_t new_mem = prev.mem_usage;
                size_t offload_buf_size = prev.offload_buffer_size;
                if (w_mem > prev.offload_buffer_size) {
                    new_mem += (w_mem - prev.offload_buffer_size);
                    offload_buf_size = w_mem;
                }
                if (new_mem <= free_mem) {
                    double bubble = std::max(0.0, prev.required_overlap - (t_cC + t_midC));
                    double new_required_overlap = t_h2d;

                    double new_time = t_cG + prev.time_cost + (prev.prev_was_gpu ? t_midG : theta + t_midC) + bubble;

                    current_candidates.push_back({
                        new_time, new_mem, new_required_overlap, 
                        2, prev_idx, false, offload_buf_size // decision=2 (Hybrid)
                    });
                }
            }
        }

        // 帕累托剪枝
        pareto_states[wid] = prune_pareto_bucketed(current_candidates);
        
        // 进度打印
        if ((weight_cnt - wid) % (weight_cnt / 20 + 1) == 0) {
            fprintf(stderr, "\r[INFO] Pareto DP Progress: %d/%d layers (States: %zu)", 
                    weight_cnt - wid, weight_cnt, pareto_states[wid].size());
            fflush(stderr);
        }
    }
    fprintf(stderr, "\n");

    double min_total_time = INFINITY;
    int best_idx = -1;

    for (int i = 0; i < (int)pareto_states[0].size(); i++) {
        const auto& s = pareto_states[0][i];
        // 总时间 = 初始传输等待 + 累计计算时间
        double total = s.required_overlap + s.time_cost;
        if (total < min_total_time) {
            min_total_time = total;
            best_idx = i;
        }
    }

    if (best_idx == -1) {
        fprintf(stderr, "[ERROR] No valid strategy found (Memory constraint too tight?)\n");
        return { {}, {} };
    }

    fprintf(stderr, "[INFO] Pareto DP Estimated Time = %.4lf ms\n", min_total_time);

    vector<string> override_list; 
    vector<string> offload_list;  

    int cur_idx = best_idx;
    for (w_id wid = 0; wid < weight_cnt; wid++) {
        const auto& s = pareto_states[wid][cur_idx];
        const string& name = tensors_by_name[wid].first;

        if (s.decision == 0) {
            override_list.push_back(name);
        } else if (s.decision == 1) {
        } else if (s.decision == 2) {
            override_list.push_back(name);
            offload_list.push_back(name);
        }

        cur_idx = s.parent_idx;
    }

    return { override_list, offload_list };
}


static void print_usage(int _, char ** argv) {
    (void)_;
    cerr << "Usage: " << argv[0] << " <model_file> [options]\n\n";
    cerr << "Options:\n";
    cerr << "  -r <op_perf_json>    Path to operator performance result JSON file (default: examples/pipo-alg/perf_result.json)\n";
    cerr << "  -alpha <float>       Alpha parameter for the algorithm (default depends on algorithm)\n";
    cerr << "  -beta <float>        Beta parameter for the algorithm (default depends on algorithm)\n";
    cerr << "  -theta <float>       Theta parameter for the algorithm (default depends on algorithm)\n";
    cerr << "  -dp                  Use dynamic programming algorithm (default)\n";
    cerr << "  -greedy              Use greedy algorithm\n";
    cerr << "  -max-n <int>         Maximum context length, mainly affects KV cache size (default: 200)\n";
    cerr << "  -max-batch <int>     Maximum batch length, depends on prefill run (default: 100)\n\n";
    cerr << "Default parameter values:\n";
    cerr << "  When using DP algorithm (-dp):    alpha=1.0, beta=1.0, theta=0.5\n";
    cerr << "  When using Greedy algorithm:      alpha=1.0, beta=1.4, theta=1.25\n";
}

int main(int argc, char ** argv) {
    const char * op_perf_result_path = "examples/pipo-alg/perf_result.json";
    const char * model_path          = nullptr;
    double       alpha               = -1.0;
    double       beta                = -1.0;
    double       theta               = -1.0;
    // use dp algorithm, otherwise greedy
    bool         use_dp              = true;
    // model context len, mainly affect kv cache size
    int          max_ctx_len         = 200;
    // depend on the prefill run
    int          max_batch_len       = 100;
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
            } else if (strcmp(argv[i], "-beta") == 0) {
                if (i + 1 < argc) {
                    beta = atof(argv[++i]);
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-dp") == 0) {
                use_dp = true;
            } else if (strcmp(argv[i], "-greedy") == 0) {
                use_dp = false;
            } else if (strcmp(argv[i], "-max-n") == 0){
                if (i + 1 < argc){
                    max_ctx_len = atoi(argv[++i]);
                }
                else {
                    print_usage(argc, argv);
                    return 1;
                }
            } 
            else if (strcmp(argv[i], "-max-batch") == 0){
                if (i + 1 < argc){
                    max_batch_len = atoi(argv[++i]);
                }
                else {
                    print_usage(argc, argv);
                    return 1;
                }
            } 
            else {
                model_path = argv[i];
            }
        }
        if (model_path == nullptr || op_perf_result_path == nullptr) {
            print_usage(argc, argv);
            return 1;
        }
    }
    // default params
    if (use_dp) {
        alpha = alpha < 0 ? 1.0 : alpha;
        beta  = beta < 0 ? 1.0 : beta;
        theta = theta < 0 ? 0.5 : theta;
    } else {
        alpha = alpha < 0 ? 1.0 : alpha;
        beta  = beta < 0 ? 1.4 : beta;
        theta = theta < 0 ? 1.25 : theta;
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
    ctx_params.n_ctx                = max_ctx_len;
    ctx_params.n_batch              = max_batch_len;
    ctx_params.no_perf              = true;

    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        cout << "[Error]" << __LINE__ << ": Failed to create llama_context\n";
        return 1;
    }
    dup2(_stderr_fd, STDERR_FILENO);
    close(_dev_null);

    size_t        extra_buf_use = pipo_get_mem_usage(ctx);
    ggml_cgraph * gf            = pipo_get_graph(ctx);

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
    // reserve 250 MB overhead
    free_memory = (free_memory - extra_buf_use) - (size_t) (250 * 1024 * 1024);

    // 需要按照 graph node 顺序排序
    auto                       tensor_by_name = model->tensors_by_name;
    unordered_map<string, int> tensor_by_name_idx;
    for (int i = 0; i < (int) tensor_by_name.size(); ++i) {
        tensor_by_name_idx[tensor_by_name[i].first] = i;
    }
    unordered_map<ggml_tensor *, int> tensor_by_name_node_idx;
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        for (ggml_tensor * t : node->src) {
            if (!t) {
                break;
            }
            if (tensor_by_name_idx.count(t->name)) {
                tensor_by_name_node_idx[tensor_by_name[tensor_by_name_idx[t->name]].second] = i;
            }
        }
    }
    sort(tensor_by_name.begin(), tensor_by_name.end(), [&](const auto & a, const auto & b) {
        return tensor_by_name_node_idx[a.second] < tensor_by_name_node_idx[b.second];
    });
    auto [override_list, offload_list] =
        use_dp ? dp_strategy(gf, tensor_by_name, op_perf_results, cpu_backend_name, gpu_backend_name, free_memory,
                             h2d_bandwidth, alpha, beta, theta) :
                 greedy_strategy(gf, tensor_by_name, op_perf_results, cpu_backend_name, gpu_backend_name, free_memory,
                                 h2d_bandwidth, alpha, beta, theta);
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
