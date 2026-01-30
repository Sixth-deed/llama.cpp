#pragma once
#include "llama.h"
#include "ggml.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
/* unique op struct */
struct pipo_unique_op {
    ggml_op                 op_type;
    ggml_type               node_type;
    std::vector<int64_t>         op_shape;
    std::vector<uint8_t>         op_param_bytes;
    std::vector<ggml_type>       src_types;
    std::vector<std::vector<int64_t>> src_nes;

    void debug_print(); 

    std::string op_key() const ;
    std::string short_desc() const; 

    // pipo_unique_op(const nlohmann::json & e); 
    pipo_unique_op() :
        op_type(GGML_OP_NONE),
        node_type(GGML_TYPE_F32),
        op_shape(0),
        op_param_bytes(0),
        src_types(0),
        src_nes(0) {}

    pipo_unique_op(const std::string&);
    pipo_unique_op(const ggml_tensor*);

    bool operator==(const pipo_unique_op & other) const; 
    bool operator!=(const pipo_unique_op & other) const { return !(*this == other); }

    ggml_tensor* to_tensor(ggml_context* ctx)const;
};
namespace std{
    template <>
    struct hash<pipo_unique_op> {
        size_t operator()(const pipo_unique_op & op) const{
            return std::hash<std::string>()(op.op_key());
        }
    };
}
struct pipo_graph_info{
    std::unordered_map<std::string, size_t> weight_sizes;
    std::unordered_set<pipo_unique_op> unique_ops;
    // byte per micro second
    double h2d_bandwidth;
    // 记录在计算图上被放在主存的相邻的 tensor 间的节点的计算节点。
    std::vector<std::tuple<std::string, std::string, std::vector<std::string>> > override_tensors_interval;
    pipo_graph_info() = default;
};

pipo_graph_info* pipo_get_graph_info(llama_context* ctx, std::unordered_set<std::string>* override_tensors = nullptr);

bool pipo_is_view_op(enum ggml_op op);
std::string pipo_make_op_key(const ggml_tensor * node);