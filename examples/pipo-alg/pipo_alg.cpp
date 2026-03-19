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
    // 传输实际慢 beta 倍
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
        // Note: 这里特判了不 override output.weight，它在cpu上算时很多时候比perf结果慢，带它的算法很不稳定
        if (strstr(t->name, "output")){
            return op_perf_results.at(cpu_name).at(pipo_make_op_key(t)) * 1.5;
        }
        return op_perf_results.at(cpu_name).at(pipo_make_op_key(t)) * alpha;
    };
    auto weight_size = [&](w_id weight_id) -> size_t {
        return ggml_nbytes(tensors_by_name[weight_id].second);
    };
    // TODO: 也许可以通过对tensors分组来减少彻底搜索的计算量

    // 用于手动过滤一些强行放在 gpu 上的 weight
    auto gpu_weight_filter = [&](w_id i) -> bool{
        return strstr(tensors_by_name[i].first.c_str(), "attn");
    };
    // 单位是字节
    constexpr size_t mem_bin_size  = 1024 * 1024;
    // 单位是毫秒
    constexpr double time_bin_size = 0.2;
    auto             mem_bin       = [](size_t size) -> int {
        return size / mem_bin_size + ((size % mem_bin_size) > (mem_bin_size / 2));
    };
    auto time_bin = [](double time) -> int {
        if (time == INFINITY) return 65535;
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
        if (gpu_weight_filter(i)) cpu_compute_time_cache[i] = INFINITY;
        gpu_compute_time_cache[i] = gpu_compute_time(w2n[i]);
    }

    // 由于 mem_bin 是近似的，有可能算法实际给出的结果使用了过多的内存，需要重新跑一遍整个算法来获取有效的结果。
    size_t actual_mem_usage;
    // 稍微留一点余量，尽量不触发重跑
    size_t target_mem_usage = free_mem - 2 * mem_bin_size;
    fprintf(stderr, "alg target mem usage = %.2lf MB\n", (double) free_mem / 1024.0 / 1024.0);
    int    iter     = 0;
    int    max_iter = 10;
    do {
        int       mem_bin_cnt = mem_bin(target_mem_usage);
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
            if (!gpu_weight_filter(wid)){
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
        target_mem_usage = target_mem_usage - (actual_mem_usage - free_mem);
        fprintf(stderr,
                "alg actual provide strategy with mem use of %ld bytes, but target memory usage is %ld bytes.\nRerun "
                "alg with new target mem = %ld\n",
                actual_mem_usage, free_mem, target_mem_usage);
        iter += 1;
    } while (iter < max_iter);
    fprintf(stderr, "alg failed to find a strategy with max_retires = %d\n", max_iter);
    return { {}, {} };
}
// 尝试更改对 alpha 参数的使用
// 这里的建模并不严谨，因为当前的dp是从后向前遍历的，无法正确模拟并发传输导致的延迟
static pair<vector<string>, vector<string>> dp_strategy2(
    ggml_cgraph *                                                gf,
    const vector<pair<string, ggml_tensor *>> &                  tensors_by_name,
    const unordered_map<string, unordered_map<string, double>> & op_perf_results,
    const char *                                                 _cpu_backend_name,
    const char *                                                 _gpu_backend_name,
    size_t                                                       free_mem,
    double                                                       h2d_bandwidth,
    // cpu 计算实际比 perf 结果会慢 alpha 倍
    const double                                                 alpha = 1.2,
    // 传输实际慢 beta 倍
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
        // Note: 这里特判了不 override output.weight，它在cpu上算时很多时候比perf结果慢，带它的算法很不稳定
        if (strstr(t->name, "output")){
            return op_perf_results.at(cpu_name).at(pipo_make_op_key(t)) * 1.5;
        }
        return op_perf_results.at(cpu_name).at(pipo_make_op_key(t));
    };
    auto weight_size = [&](w_id weight_id) -> size_t {
        return ggml_nbytes(tensors_by_name[weight_id].second);
    };
    // TODO: 也许可以通过对tensors分组来减少彻底搜索的计算量

    // 用于手动过滤一些强行放在 gpu 上的 weight
    auto gpu_weight_filter = [&](w_id i) -> bool{
        return strstr(tensors_by_name[i].first.c_str(), "attn");
    };
    // 单位是字节
    constexpr size_t mem_bin_size  = 1024 * 1024;
    // 单位是毫秒
    constexpr double time_bin_size = 0.2;
    auto             mem_bin       = [](size_t size) -> int {
        return size / mem_bin_size + ((size % mem_bin_size) > (mem_bin_size / 2));
    };
    auto time_bin = [](double time) -> int {
        if (time == INFINITY) return 65535;
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
    auto parallel_transfer_overhead = [&](int time_bin){
        return (time_bin * time_bin_size) * (1 - 1 / alpha);
    };
    // cache
    vector<double> cpu_compute_time_cache(weight_cnt);
    vector<double> gpu_compute_time_cache(weight_cnt);
    for (w_id i = 0; i < weight_cnt; i++) {
        cpu_compute_time_cache[i] = cpu_compute_time(w2n[i]);
        if (gpu_weight_filter(i)) cpu_compute_time_cache[i] = INFINITY;
        gpu_compute_time_cache[i] = gpu_compute_time(w2n[i]);
    }

    // 由于 mem_bin 是近似的，有可能算法实际给出的结果使用了过多的内存，需要重新跑一遍整个算法来获取有效的结果。
    size_t actual_mem_usage;
    // 稍微留一点余量，尽量不触发重跑
    size_t target_mem_usage = free_mem - 2 * mem_bin_size;
    fprintf(stderr, "alg target mem usage = %.2lf MB\n", (double) free_mem / 1024.0 / 1024.0);
    int    iter     = 0;
    int    max_iter = 10;
    do {
        int       mem_bin_cnt = mem_bin(target_mem_usage);
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
            if (!gpu_weight_filter(wid)){
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
                    if (t_nc > t_cmidC + parallel_transfer_overhead(t_ntf) + dp_C[idx_2d(t_ntf, mem)]) {
                        t_nc        = t_cmidC+ parallel_transfer_overhead(t_ntf) + dp_C[idx_2d(t_ntf, mem)];
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
                    int    b_nttf_nG = t_tf + time_bin(t_cC * alpha + t_cmidG);
                    int    b_nttf_nC = t_tf + time_bin(t_cC * alpha + t_cmidC * alpha);
                    double t_next_G =
                        b_nttf_nG >= ttf_bin_cnt ? INFINITY : t_cmidG + dp_G[idx_2d(b_nttf_nG, mem)] + theta;
                    double t_next_C = b_nttf_nC >= ttf_bin_cnt ? INFINITY : t_cmidC * alpha + dp_C[idx_2d(b_nttf_nC, mem)];

                    dp_C[idx_2d(t_tf, mem)] = t_cC * alpha + min(t_next_C, t_next_G);
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
        target_mem_usage = target_mem_usage - (actual_mem_usage - free_mem);
        fprintf(stderr,
                "alg actual provide strategy with mem use of %ld bytes, but target memory usage is %ld bytes.\nRerun "
                "alg with new target mem = %ld\n",
                actual_mem_usage, free_mem, target_mem_usage);
        iter += 1;
    } while (iter < max_iter);
    fprintf(stderr, "alg failed to find a strategy with max_retires = %d\n", max_iter);
    return { {}, {} };
}
// 尝试提高对时间建模的精度
static pair<vector<string>, vector<string>> dp_strategy3(
    ggml_cgraph *                                                gf,
    const vector<pair<string, ggml_tensor *>> &                  tensors_by_name,
    const unordered_map<string, unordered_map<string, double>> & op_perf_results,
    const char *                                                 _cpu_backend_name,
    const char *                                                 _gpu_backend_name,
    size_t                                                       free_mem,
    double                                                       h2d_bandwidth,
    // cpu 计算实际比 perf 结果会慢 alpha 倍
    const double                                                 alpha = 1.0,
    // 传输实际慢 beta 倍
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
        // Note: 这里特判了不 override output.weight，它在cpu上算时很多时候比perf结果慢，带它的算法很不稳定
        if (strstr(t->name, "output")){
            return op_perf_results.at(cpu_name).at(pipo_make_op_key(t)) * 1.5;
        }
        return op_perf_results.at(cpu_name).at(pipo_make_op_key(t)) * alpha;
    };
    auto weight_size = [&](w_id weight_id) -> size_t {
        return ggml_nbytes(tensors_by_name[weight_id].second);
    };
    // TODO: 也许可以通过对tensors分组来减少彻底搜索的计算量

    // 用于手动过滤一些强行放在 gpu 上的 weight
    auto gpu_weight_filter = [&](w_id i) -> bool{
        return strstr(tensors_by_name[i].first.c_str(), "attn");
    };
    // 单位是字节
    constexpr size_t mem_bin_size  = 1024 * 1024;
    // 单位是毫秒
    constexpr double time_bin_size = 0.07;
    auto             mem_bin       = [](size_t size) -> int {
        return size / mem_bin_size + ((size % mem_bin_size) > (mem_bin_size / 2));
    };
    auto time_bin = [](double time) -> int {
        if (time == INFINITY) return 65535;
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
    assert(ttf_bin_cnt > 0 && ttf_bin_cnt < 65536);
    // cache
    vector<double> cpu_compute_time_cache(weight_cnt);
    vector<double> gpu_compute_time_cache(weight_cnt);
    for (w_id i = 0; i < weight_cnt; i++) {
        cpu_compute_time_cache[i] = cpu_compute_time(w2n[i]);
        if (gpu_weight_filter(i)) cpu_compute_time_cache[i] = INFINITY;
        gpu_compute_time_cache[i] = gpu_compute_time(w2n[i]);
    }

    // 由于 mem_bin 是近似的，有可能算法实际给出的结果使用了过多的内存，需要重新跑一遍整个算法来获取有效的结果。
    size_t actual_mem_usage;
    // 稍微留一点余量，尽量不触发重跑
    size_t target_mem_usage = free_mem - 2 * mem_bin_size;
    fprintf(stderr, "alg target mem usage = %.2lf MB\n", (double) free_mem / 1024.0 / 1024.0);
    int    iter     = 0;
    int    max_iter = 10;
    do {
        int       mem_bin_cnt = mem_bin(target_mem_usage);
        const int W           = weight_cnt;
        const int T           = ttf_bin_cnt;
        const size_t M           = mem_bin_cnt;
        auto      idx_3d      = [&](int w, int t, int m) -> size_t {
            return ((size_t) w * T + t) * M + m;
        };

        auto idx_2d = [&](int t, int m) -> size_t {
            return (size_t) t * M + m;
        };
        fprintf(stderr, "[INFO] dp arr take %.4lf MB\n", (double) (ttf_bin_cnt * mem_bin_cnt * 8) / 1024.0 / 1024.0);
        fprintf(stderr, "[INFO] dp trace arr take %.4lf MB\n",
                (double) (weight_cnt * ttf_bin_cnt * (size_t)mem_bin_cnt) * 4.375 / 1024.0 / 1024.0);
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
            if (!gpu_weight_filter(wid)){
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
        target_mem_usage = target_mem_usage - (actual_mem_usage - free_mem);
        fprintf(stderr,
                "alg actual provide strategy with mem use of %ld bytes, but target memory usage is %ld bytes.\nRerun "
                "alg with new target mem = %ld\n",
                actual_mem_usage, free_mem, target_mem_usage);
        iter += 1;
    } while (iter < max_iter);
    fprintf(stderr, "alg failed to find a strategy with max_retires = %d\n", max_iter);
    return { {}, {} };
}
static vector<string> offload_dp(vector<string> &                                             override_list,
                                 ggml_cgraph *                                                gf,
                                 const vector<pair<string, ggml_tensor *>> &                  tensors_by_name,
                                 const unordered_map<string, unordered_map<string, double>> & op_perf_results,
                                 const char *                                                 _cpu_backend_name,
                                 const char *                                                 _gpu_backend_name,
                                 double                                                       h2d_bandwidth) {
    const int    weight_cnt = tensors_by_name.size();
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
    double alpha            = 1.1;
    auto   gpu_compute_time = [&](n_id node_id) -> double {
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
    vector<string>        offload_list;
    unordered_set<string> override_set(override_list.begin(), override_list.end());
    // 每一个带权重节点自身计算时间与它之后的带权重节点间的节点计算时间之和
    vector<double>        computation_internal(weight_cnt, 0);
    // 前缀和
    vector<double>        computation_sum(weight_cnt + 1, 0);
    bool                  prev_on_gpu = false;
    {
        double C = 0;
        double G = 0;
        for (n_id i = 0; i < w2n[0]; i++) {
            C += cpu_compute_time(i);
            G += gpu_compute_time(i);
        }
        computation_internal[0] = override_set.count(tensors_by_name[0].first) ? C : G;
        computation_sum[0]      = computation_internal[0];
    }
    n_id prev = w2n[0];
    for (w_id i = 1; i < weight_cnt; i++) {
        n_id cur        = w2n[i];
        bool cur_on_gpu = !override_set.count(tensors_by_name[i].first);
        if (cur_on_gpu) {
            computation_internal[i] += gpu_compute_time(cur);
        } else {
            computation_internal[i] += cpu_compute_time(cur);
        }
        for (n_id j = prev + 1; j < cur; j++) {
            if (cur_on_gpu || prev_on_gpu) {
                computation_internal[i] += gpu_compute_time(j);
            } else {
                computation_internal[i] += cpu_compute_time(j);
            }
        }
        prev               = cur;
        prev_on_gpu        = cur_on_gpu;
        computation_sum[i] = computation_sum[i - 1] + computation_internal[i] +( cur_on_gpu ? gpu_compute_time(cur): cpu_compute_time(cur));
    }
    auto computation_between = [&](w_id l, w_id r) -> double {
        if (l == -1) {
            return computation_sum[r] - cpu_compute_time(w2n[r]);
        }
        return computation_sum[r] - computation_sum[l] - cpu_compute_time(w2n[r]);
    };
    vector<size_t> weight_size(weight_cnt, 0);
    vector<double> weight_trans_time(weight_cnt, INFINITY);
    double beta = 1.3;
    for (w_id wid = 0; wid < weight_cnt; wid++) {
        ggml_tensor * w  = tensors_by_name[wid].second;
        weight_size[wid] = ggml_nbytes(w);
        // if (strstr(tensors_by_name[wid].first.c_str(), "attn")){
        // weight_trans_time[wid] = 1000;
        // }else{
        weight_trans_time[wid] = beta * (double) ggml_nbytes(w) / h2d_bandwidth;
        // }
    }
    // offload tensor 带来的时间差，使总计算时间时间减少时为负。越小越好
    vector<double> offload_gain(weight_cnt, INFINITY);
    vector<w_id>   offload_prev(weight_cnt, -1);
    if (override_set.count(tensors_by_name[0].first)) {
        offload_gain[0] =
            gpu_compute_time(w2n[0]) + weight_trans_time[0] - cpu_compute_time(w2n[0]) - computation_between(-1, 0);
    }
    for (w_id cur = 1; cur < weight_cnt; cur++) {
        if (!override_set.count(tensors_by_name[cur].first)) {
            continue;
        }
        offload_gain[cur] =
            gpu_compute_time(w2n[cur]) - cpu_compute_time(w2n[cur]) +
            max((alpha - 1) * weight_trans_time[cur], weight_trans_time[cur] - computation_between(-1, cur));
        for (w_id prev = 0; prev < cur; prev++) {
            if (!override_set.count(tensors_by_name[prev].first)) {
                continue;
            }
            double offload_cur_gain =
                offload_gain[prev] + gpu_compute_time(w2n[cur]) - cpu_compute_time(w2n[cur]) +
                max((alpha - 1) * weight_trans_time[cur], weight_trans_time[cur] - computation_between(prev, cur));
            if (offload_cur_gain < offload_gain[cur]) {
                offload_gain[cur] = offload_cur_gain;
                offload_prev[cur] = prev;
            }
        }
    }
    double min_gain = 0;
    w_id   min_last = -1;
    for (w_id i = 0; i < weight_cnt; i++) {
        if (min_gain > offload_gain[i]) {
            min_gain = offload_gain[i];
            min_last = i;
        }
    }
    vector<w_id> offload_ids;
    while (min_last != -1) {
        offload_list.push_back(tensors_by_name[min_last].first);
        offload_ids.push_back(min_last);
        min_last = offload_prev[min_last];
    }
    fprintf(stderr, "offload estimate gain = %lf\n", min_gain);
    reverse(offload_list.begin(), offload_list.end());
    reverse(offload_ids.begin(), offload_ids.end());
    prev = -1;
    for (size_t i = 0; i < offload_ids.size(); i++){
        w_id wid = offload_ids[i];
        fprintf(stderr, "offload tensor[%d] %-10s\ntensor transfer time = %4lfms, tensor mid compute time = %4lfms\n", wid, tensors_by_name[wid].first.c_str(), weight_trans_time[wid], computation_between(prev, wid));
        prev = wid;
    }
    return offload_list;
}
static pair<vector<string>, vector<string>> prefill_first_strategy(
    ggml_cgraph* gf,
    const vector<pair<string, ggml_tensor *>> &                  tensors_by_name,
    int                                                          n_prompt,
    const unordered_map<string, double> &                              op_perf_results_batched,
    const unordered_map<string, unordered_map<string, double>> & op_perf_results,
    const char *                                                 _cpu_backend_name,
    const char *                                                 _gpu_backend_name,
    size_t                                                       free_mem,
    double                                                       h2d_bandwidth) {

    const int weight_cnt = tensors_by_name.size();
    const int node_cnt = ggml_graph_n_nodes(gf);
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

    vector<string> override_list;

    vector<double> p_weight_node_compute_time(weight_cnt, INFINITY);
    vector<double> p_weight_node_interval_compute_time(weight_cnt, INFINITY);
    vector<size_t> weight_size(weight_cnt, 0);
    vector<double> weight_trans_time(weight_cnt, INFINITY);
    for (w_id wid = 0; wid < weight_cnt; wid ++){
        ggml_tensor* w = tensors_by_name[wid].second;
        ggml_tensor* n = ggml_graph_node(gf, w2n[wid]);
        const string node_op_key = pipo_make_op_key(n);
        {
            double perf_time_per_batch = op_perf_results_batched.count(node_op_key) ? op_perf_results_batched.at(node_op_key) : INFINITY;
            if (perf_time_per_batch < 0 || perf_time_per_batch == INFINITY){
                perf_time_per_batch = 0;
                fprintf(stderr, "Unspported mid node [%s] for gpu between weights, which is ignored by algorithm, but should be overrided to HOST during prefill.\n", n->name);
                override_list.push_back(w->name);
                free_mem += ggml_nbytes(w);
            }
            p_weight_node_compute_time[wid] = perf_time_per_batch * n_prompt;
        }
        weight_size[wid] = ggml_nbytes(w);
        if (strstr(tensors_by_name[wid].first.c_str(), "attn")){
            // 阻止对attn的offload行为
            weight_trans_time[wid] = 1000;
        }else{
            weight_trans_time[wid] = (double) ggml_nbytes(w) / h2d_bandwidth;
        }
        {
            n_id l = w2n[wid] + 1;
            n_id r = wid == weight_cnt - 1 ? node_cnt - 1: w2n[wid + 1];
            double est_time = 0;
            for (n_id nid = l; nid < r; nid ++){
                ggml_tensor* cur_node = ggml_graph_node(gf, nid);
                if (pipo_is_view_op(cur_node->op)){
                    continue;
                }
                double cur_node_time_per_batch = op_perf_results_batched.count(pipo_make_op_key(cur_node)) ? op_perf_results_batched.at(pipo_make_op_key(cur_node)) : INFINITY;
                if (cur_node_time_per_batch < 0 || cur_node_time_per_batch == INFINITY) {
                    cur_node_time_per_batch = 0;
                    fprintf(stderr, "Unspported mid node [%s] for gpu between weights, which is ignored by algorithm, but should be overrided to HOST during prefill.\nop_key=%s\n", cur_node->name, pipo_make_op_key(cur_node).c_str());
                }
                est_time += cur_node_time_per_batch * n_prompt;
            }
            p_weight_node_interval_compute_time[wid] = est_time;
        }
    }


    {
        constexpr int MEM_BIN_SIZE = 1024 * 1024;
        auto mem_bin = [](size_t size) -> int {
            return size / MEM_BIN_SIZE + (int) (size % MEM_BIN_SIZE >= MEM_BIN_SIZE / 2);
        };
        const int mem_bin_cnt = mem_bin(free_mem);
        vector<int> weight_mem_bin(weight_cnt, 0);
        for (w_id i = 0; i < weight_cnt; i++){
            weight_mem_bin[i] = mem_bin(weight_size[i]);
        }
        vector<vector<w_id>> override_prev(weight_cnt, vector<w_id>(mem_bin_cnt, -1));

        vector<double> compute_time_from_start(weight_cnt, 0);
        vector<vector<double>> compute_time_between(weight_cnt, vector<double>(weight_cnt, 0));

        for (w_id i = 1; i < weight_cnt; i++){
            compute_time_from_start[i] = compute_time_from_start[i - 1] + p_weight_node_compute_time[i - 1]+  p_weight_node_interval_compute_time[i - 1];
            for (w_id prev = 0; prev < i; prev ++){
                compute_time_between[prev][i] = compute_time_between[prev][i - 1] + p_weight_node_compute_time[i - 1]+  p_weight_node_interval_compute_time[i - 1];
            }
        }
        #if 0
        /*
            DP logic
            实现压缩掉了第一个维度
            dp[cur_wid][prev_override_wid][mem_usage] =
                prev_override_wid < cur_wid
                // remain cur on gpu, use gpu memory, no parallel transfer during compute
                ? dp[cur_wid - 1][prev_override_wid][mem_usage - weight_size(cur_wid)] + compute_time_between(cur_wid - 1, cur_wid) + compute_time(cur_wid)
                // offload cur tensor to cpu, dynamically transfer it to gpu during computation
                : min(dp[cur_wid - 1][prev][mem_usage] + max(compute_time_betweem(prev, cur_wid), transfer_time(cur_wid)) + compute_time(cur_wid), for prev from 0 to cur_wid)
            ;
        */
        vector<vector<double>> dp(weight_cnt, vector<double>(mem_bin_cnt, INFINITY));
        for (int mem = 0; mem < mem_bin_cnt; mem++){
            dp[0][mem] = weight_trans_time[0] + p_weight_node_compute_time[0];
        }
        for (int wid = 1; wid < weight_cnt; wid ++){
            for (int mem = mem_bin_cnt - 1; mem >= 0; mem--){
                if (mem < weight_mem_bin[wid]){
                    dp[wid][mem] = max(compute_time_from_start[wid], weight_trans_time[wid]) + p_weight_node_compute_time[wid];
                    for (int prev_override_wid = 0; prev_override_wid < wid; prev_override_wid ++){
                        // override wid tensor to cpu
                        double override_time_est = dp[prev_override_wid][mem] +
                            max(compute_time_between[prev_override_wid][wid], weight_trans_time[wid])
                            +  p_weight_node_compute_time[wid];
                        if (override_time_est < dp[wid][mem]){
                            dp[wid][mem] = override_time_est;
                            override_prev[wid][mem] = prev_override_wid;
                        }
                        // keep wid tensor on gpu
                        dp[prev_override_wid][mem] = INFINITY;
                    }
                }
                else{
                    // override wid tensor to cpu
                    dp[wid][mem] = max(compute_time_from_start[wid], weight_trans_time[wid]) + p_weight_node_compute_time[wid];
                    for (int prev_override_wid = 0; prev_override_wid < wid; prev_override_wid ++){
                        double override_time_est = dp[prev_override_wid][mem] +
                            max(compute_time_between[prev_override_wid][wid], weight_trans_time[wid])
                            +  p_weight_node_compute_time[wid];
                        if (override_time_est < dp[wid][mem]){
                            dp[wid][mem] = override_time_est;
                            override_prev[wid][mem] = prev_override_wid;
                        }

                        // keep wid tensor on gpu
                        dp[prev_override_wid][mem]=
                            dp[prev_override_wid][mem - weight_mem_bin[wid]] + p_weight_node_interval_compute_time[wid - 1] +
                            p_weight_node_compute_time[wid];
                    }
                }
            }
        }
        double min_time = INFINITY;
        w_id min_override_prev = -1;
        for (w_id i = 0; i < weight_cnt; i++){
            if (min_time > dp[i][mem_bin_cnt - 1]){
                min_override_prev = i;
                min_time = dp[i][mem_bin_cnt - 1];
            }
        }
        fprintf(stderr, "alg estimate prefill time is %.2lf ms\n", min_time);

        int cur_mem = mem_bin_cnt - 1;
        w_id cur_override = min_override_prev;
        while (cur_override != -1){
            override_list.push_back(string(tensors_by_name[cur_override].first));
            w_id cur_prev = override_prev[cur_override][cur_mem];
            for (w_id i = cur_prev + 1; i < cur_override; i++){
                cur_mem -= weight_mem_bin[i];
            }
            cur_override = cur_prev;
        }
        #endif
        int N = weight_cnt;
        int M = mem_bin_cnt;

        const double INF = 1e30;

        vector<vector<double>> dp(N + 1, vector<double>(M, INF));
        vector<vector<double>> new_dp(N + 1, vector<double>(M, INF));

        struct PrevState {
            int  prev_override;
            int  prev_mem;
            bool is_override;
        };

        vector<vector<vector<PrevState>>> trace(N, vector<vector<PrevState>>(N + 1, vector<PrevState>(M)));

        dp[0][0] = 0.0;  // last_override = -1

        for (int wid = 0; wid < N; wid++) {
            for (int i = 0; i <= N; i++) {
                for (int m = 0; m < M; m++) {
                    new_dp[i][m] = INF;
                }
            }

            for (int last = -1; last < wid; last++) {
                int last_idx = last + 1;

                for (int mem = 0; mem < M; mem++) {
                    double cur_time = dp[last_idx][mem];
                    if (cur_time == INF) {
                        continue;
                    }

                    int size = weight_mem_bin[wid];

                    /* ---------------- keep on GPU ---------------- */

                    if (mem + size < M) {
                        double cost = cur_time + p_weight_node_interval_compute_time[wid == 0 ? 0 : wid - 1] +
                                      p_weight_node_compute_time[wid];

                        int new_last_idx = last_idx;
                        int new_mem      = mem + size;

                        if (cost < new_dp[new_last_idx][new_mem]) {
                            new_dp[new_last_idx][new_mem] = cost;

                            trace[wid][new_last_idx][new_mem] = { last, mem, false };
                        }
                    }

                    /* ---------------- override ---------------- */

                    double compute_window;

                    if (last == -1) {
                        compute_window = compute_time_from_start[wid];
                    } else {
                        compute_window = compute_time_between[last][wid];
                    }

                    double cost =
                        cur_time + max(compute_window, weight_trans_time[wid]) + p_weight_node_compute_time[wid];

                    int new_last     = wid;
                    int new_last_idx = new_last + 1;
                    int new_mem      = mem;

                    if (cost < new_dp[new_last_idx][new_mem]) {
                        new_dp[new_last_idx][new_mem] = cost;

                        trace[wid][new_last_idx][new_mem] = { last, mem, true };
                    }
                }
            }

            dp.swap(new_dp);
        }
        double best      = INF;
        int    best_last = -1;

        for (int last = -1; last < N; last++) {
            int idx = last + 1;
            int mem = mem_bin_cnt - 1;
            if (dp[idx][mem] < best) {
                best      = dp[idx][mem];
                best_last = last;
            }
        }

        int last = best_last;
        int mem  = mem_bin_cnt - 1;

        for (int wid = N - 1; wid >= 0; wid--) {
            int last_idx = last + 1;

            PrevState t = trace[wid][last_idx][mem];

            if (t.is_override) {
                override_list.push_back(tensors_by_name[wid].first);
                last = t.prev_override;
                mem  = t.prev_mem;
            } else {
                last = t.prev_override;
                mem  = t.prev_mem;
            }
        }

        reverse(override_list.begin(), override_list.end());
    }
    vector<string> offload_list = offload_dp(override_list, gf, tensors_by_name, op_perf_results, _cpu_backend_name, _gpu_backend_name, h2d_bandwidth);
    return {override_list, offload_list};
}



static pair<vector<string>, vector<string>> static_like_stratagy(ggml_cgraph* gf,
    const vector<pair<string, ggml_tensor *>> &                  tensors_by_name,
    const unordered_map<string, unordered_map<string, double>> & op_perf_results,
    const char *                                                 _cpu_backend_name,
    const char *                                                 _gpu_backend_name,
    size_t free_mem,
    double h2d_bandwidth,
    llama_model* model){
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
    
    size_t mem_usage = 0;
    for (auto& [n, t] : tensors_by_name){
        mem_usage += ggml_nbytes(t);
    }
    unordered_map<int, std::pair<vector<ggml_tensor*>, size_t>> tensor_map;
    for(const auto& layer : model->layers){
        int t_id = 0;
        for (; t_id < &layer.ffn_act_eps - &layer.attn_norm ; t_id ++){
            ggml_tensor* t = *(&layer.attn_norm + t_id);
            if (!t || pipo_is_view_op(t->op))continue;
            if (!tensor_map.count(t_id)){
                tensor_map.insert(make_pair(t_id, make_pair(vector<ggml_tensor*>(), 0)));
            }
            tensor_map[t_id].first.push_back(t);
            tensor_map[t_id].second += ggml_nbytes(t);
        }
    }
    vector<tuple<int ,vector<ggml_tensor*>, size_t>> tensor_groups(tensor_map.size());
    for (auto& [tid, p] : tensor_map){
        tensor_groups.push_back(make_tuple(tid, p.first, p.second));
    }
    sort(tensor_groups.begin(), tensor_groups.end(), [&](auto &l, auto& r){
        return get<2>(l) > get<2>(r);
    });
    // override
    vector<string> override_list;
    fprintf(stderr, "graph nodes cnt = %d\n", ggml_graph_n_nodes(gf));
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++){
        if (!n2w.count(i)) continue;
        if (gpu_compute_time(i) < 0 || gpu_compute_time(i) == INFINITY){
            override_list.push_back(string(tensors_by_name[n2w[i]].first));
            mem_usage -= ggml_nbytes(tensors_by_name[n2w[i]].second);
        }
    }
    int cur_override_i = 0;
    int cur_override_j = 0;
    while (mem_usage > free_mem){
        auto& override_tensor_arr = get<1>(tensor_groups[cur_override_i]);
        if ((size_t)cur_override_j >= override_tensor_arr.size()){
            cur_override_j = 0;
            cur_override_i += 1;
            continue;
        }
        override_list.push_back(override_tensor_arr[cur_override_j]->name);
        mem_usage -= ggml_nbytes(override_tensor_arr[cur_override_j]);
        cur_override_j += 1;
    }
    auto offload_list = offload_dp(override_list, gf, tensors_by_name, op_perf_results, _cpu_backend_name, _gpu_backend_name, h2d_bandwidth);
    return {override_list, offload_list};
}

 

static void print_usage(int _, char ** argv) {
    (void)_;
    cerr << "Usage: " << argv[0] << " -m <model_file> [options]\n\n";
    cerr << "Options:\n";
    cerr << "  -perf      <op_perf_json>   Path to operator performance result JSON file (default: examples/pipo-alg/perf_result.json)\n";
    cerr << "  -alpha     <float>          Alpha parameter for the algorithm\n";
    cerr << "  -beta      <float>          Beta parameter for the algorithm\n";
    cerr << "  -theta     <float>          Theta parameter for the algorithm\n";
    cerr << "  -max-batch <int>            Maximum batch length, depends on prefill run (default: 100)\n\n";
    cerr << "  -alg-no    <int>            Set algorithm (default: 0)\n";
    cerr << "  -mem       <int>            Target gpu mem usage (default: free mem count)\n";
}

int main(int argc, char ** argv) {
    const char * op_perf_result_path = "examples/pipo-alg/perf_result.json";
    const char * model_path          = nullptr;
    double       alpha               = -1.0;
    double       beta                = -1.0;
    double       theta               = -1.0;
    // depend on the prefill run
    int          max_batch_len       = 100;
    int mem_target = -1;
    int alg_no = 0;
    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-perf") == 0) {
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
            }else if (strcmp(argv[i], "-max-batch") == 0){
                if (i + 1 < argc){
                    max_batch_len = atoi(argv[++i]);
                }
                else {
                    print_usage(argc, argv);
                    return 1;
                }
            }
            else if (strcmp(argv[i], "-alg-no") == 0){
                if (i + 1 < argc){
                    alg_no = atoi(argv[++i]);
                }
                else {
                    print_usage(argc, argv);
                    return 1;
                }
            }
            else if (strcmp(argv[i], "-mem") == 0){
                if (i + 1 < argc){
                    mem_target = atoi(argv[++i]);
                }
                else {
                    print_usage(argc, argv);
                    return 1;
                }
            }
            else if (strcmp(argv[i], "-m") == 0){
                if (i + 1 < argc)
                    model_path = argv[++i];
                else {
                    print_usage(argc, argv);
                    return 1;
                }
            }
        }
        if (model_path == nullptr || op_perf_result_path == nullptr) {
            print_usage(argc, argv);
            return 1;
        }
    }
    // default params
    alpha = alpha < 0 ? 1.0 : alpha;
    beta  = beta < 0 ? 1.0 : beta;
    theta = theta < 0 ? 0.5 : theta;
   
    ifstream                                             _op_perf_result_s(op_perf_result_path, ios::in);
    nlohmann::json                                       _j              = nlohmann::json::parse(_op_perf_result_s);
    unordered_map<string, unordered_map<string, double>> op_perf_results = _j["op_perf_result"];
    unordered_map<string, double>                               op_perf_results_batched;
    if (!_j["op_perf_batched"].is_null()){
        op_perf_results_batched = _j["op_perf_batched"];
    }
    double                                               h2d_bandwidth   = _j["h2d_bandwidth"];

    int max_ctx_len = 100;
    if (!_j["context_size"].is_null()){
        max_ctx_len = _j["context_size"];
    }
    _op_perf_result_s.close();
    _j.clear();
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
    ggml_cgraph * gf            = pipo_get_graph(ctx, 1);

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
    if (mem_target == -1) {
        ggml_backend_dev_t dev = ggml_backend_get_device(gpu_backend);
        size_t             _;
        ggml_backend_dev_memory(dev, &free_memory, &_);
        fprintf(stderr, "Env free mem = %.2lf MB\n", (double) free_memory / 1024.0 / 1024.0);

        free_memory = (free_memory - extra_buf_use) - (size_t) 270 * 1024 * 1024;
    } else {
        free_memory = (size_t)mem_target * 1024 * 1024;
    }

    fprintf(stderr, "Target weight mem usage = %.2lf MB\n", (double) free_memory / 1024.0 / 1024.0);

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
    pair<vector<string>, vector<string>> alg_result_pair;
    switch (alg_no) {
        case 0:
            alg_result_pair = dp_strategy(gf, tensor_by_name, op_perf_results, cpu_backend_name, gpu_backend_name, free_memory,
                             h2d_bandwidth, alpha, beta, theta);
            break;
        case 1:
            alg_result_pair = prefill_first_strategy(gf, tensor_by_name, max_batch_len, op_perf_results_batched, op_perf_results, cpu_backend_name, gpu_backend_name, free_memory, h2d_bandwidth);
            break;
        case 2:
        default:
            alg_result_pair = static_like_stratagy(gf, tensor_by_name,  op_perf_results, cpu_backend_name, gpu_backend_name,free_memory, h2d_bandwidth, model);
            break;
        case 3:
            alg_result_pair = dp_strategy2(gf, tensor_by_name, op_perf_results, cpu_backend_name, gpu_backend_name, free_memory,
                             h2d_bandwidth, 1.2, beta, theta);
            break;
        case 4:
            alg_result_pair = dp_strategy3(gf, tensor_by_name, op_perf_results, cpu_backend_name, gpu_backend_name, free_memory,
                             h2d_bandwidth, alpha, beta, theta);
            break;
    }
    vector<string>& override_list = alg_result_pair.first;
    vector<string>& offload_list = alg_result_pair.second;

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
