#include "pipo_op_perf.h"
#include "../common/base64.hpp"
#include "llama-context.h"

#include <iostream>
#include <regex>
using namespace std;

void pipo_unique_op::debug_print() {
    std::cerr << "op_type: " << op_type << '\n';
    std::cerr << "node_type: " << node_type << '\n';
    std::cerr << "op_shape: " << op_shape.size() << '\n';
    for (size_t i = 0; i < op_shape.size(); i++) {
        std::cerr << "op_shape[" << i << "]: " << op_shape[i] << '\n';
    }
    std::cerr << '\n';
    std::cerr << "op_param_bytes: " << op_param_bytes.size() << '\n';
    for (size_t i = 0; i < op_param_bytes.size(); i++) {
        std::cerr << "op_param_bytes[" << i << "]: " << op_param_bytes[i] << '\n';
    }
    std::cerr << '\n';
    std::cerr << "src_types: " << src_types.size() << '\n';
    for (size_t i = 0; i < src_types.size(); i++) {
        std::cerr << "src_types[" << i << "]: " << src_types[i] << '\n';
    }
    std::cerr << '\n';
    std::cerr << "src_nes: " << src_nes.size() << '\n';
    for (size_t i = 0; i < src_nes.size(); i++) {
        std::cerr << "src_nes[" << i << "]: " << src_nes[i].size() << '\n';
        for (size_t j = 0; j < src_nes[i].size(); j++) {
            std::cerr << "src_nes[" << i << "][" << j << "]: " << src_nes[i][j] << '\n';
        }
        std::cerr << '\n';
    }
    std::cerr << '\n';
}

string pipo_unique_op::op_key() const {
    string key;
    key.reserve(256);
    key += std::to_string((int) op_type);
    key += "#";
    key += std::to_string((int) node_type);
    key += '[';
    for (size_t i = 0; i < op_shape.size(); i++) {
        key += ',';
        key += std::to_string(op_shape[i]);
    }
    key += "]#";
    for (size_t i = 0; i < src_types.size(); i++) {
        key += '|';
        key += std::to_string((int) src_types[i]);
        key += '[';
        for (size_t j = 0; j < src_nes[i].size(); j++) {
            key += ',';
            key += std::to_string(src_nes[i][j]);
        }
        key += ']';
    }
    key += '#';
    key += base64::encode(std::string(reinterpret_cast<const char *>(op_param_bytes.data()), op_param_bytes.size()));
    return key;
}

std::string pipo_unique_op::short_desc() const {
    string desc = string(ggml_op_name(op_type)) + ":" + string(ggml_type_name(node_type)) + "(";
    for (size_t i = 0; i < op_shape.size(); i++) {
        desc += to_string(op_shape[i]);
        if (i < op_shape.size() - 1) {
            desc += "x";
        }
    }
    desc += ")";
    return desc;
}
static std::vector<int64_t> parse_comma_list(const std::string& s) {
    std::vector<int64_t> result;
    if (s.empty() || s == ",") return result;
    
    // 去掉开头的逗号（如果存在）
    size_t start_pos = 0;
    if (s.front() == ',') start_pos = 1;
    
    std::string trimmed = s.substr(start_pos);
    if (trimmed.empty()) return result;
    
    // 用逗号分割
    size_t pos = 0;
    std::string token;
    while ((pos = trimmed.find(',')) != std::string::npos) {
        token = trimmed.substr(0, pos);
        if (!token.empty()) {
            result.push_back(std::stoll(token));
        }
        trimmed.erase(0, pos + 1);
    }
    if (!trimmed.empty()) {
        result.push_back(std::stoll(trimmed));
    }
    return result;
}

pipo_unique_op::pipo_unique_op(const std::string& op_key) {
    // 改进的正则表达式，支持任意维度
    static const std::regex op_key_regex(
        R"(^(\d+)#(\d+)\[([^\]]*)\]#((?:\|\d+\[[^\]]*\])*)#([A-Za-z0-9+/]*={0,2})$)"
    );
    
    std::smatch matches;
    if (!std::regex_match(op_key, matches, op_key_regex)) {
        throw std::runtime_error("op_key format mismatch: " + op_key.substr(0, 100));
    }
    
    // 1. 解析op_type和node_type
    op_type = static_cast<ggml_op>(std::stoi(matches[1].str()));
    node_type = static_cast<ggml_type>(std::stoi(matches[2].str()));
    
    // 2. 解析op_shape
    op_shape = parse_comma_list(matches[3].str());
    
    // 3. 解析src部分
    std::string src_part = matches[4].str();
    if (!src_part.empty()) {
        // 匹配每个src: |type[ne1,ne2,...]
        static const std::regex src_regex(R"(\|(\d+)\[([^\]]*)\])");
        auto src_begin = std::sregex_iterator(src_part.begin(), src_part.end(), src_regex);
        auto src_end = std::sregex_iterator();
        
        for (std::sregex_iterator i = src_begin; i != src_end; ++i) {
            std::smatch src_match = *i;
            // src类型
            ggml_type src_type = static_cast<ggml_type>(std::stoi(src_match[1].str()));
            src_types.push_back(src_type);
            
            // src形状
            std::vector<int64_t> src_ne = parse_comma_list(src_match[2].str());
            src_nes.push_back(src_ne);
        }
    }
    
    // 4. 解析base64参数
    std::string b64_str = matches[5].str();
    if (!b64_str.empty()) {
        try {
            std::string decoded = base64::decode(b64_str);
            op_param_bytes.assign(decoded.begin(), decoded.end());
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Base64 decode failed: ") + e.what());
        }
    }
    
}
std::string pipo_make_op_key(const ggml_tensor * node) {
    std::string key;
    key.reserve(256);

    key += std::to_string((int) node->op);
    key += "#";
    key += std::to_string((int) node->type);
    // the node size info
    key += '[';
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        key += ',';
        key += std::to_string((long long) node->ne[d]);
    }
    key += "]#";
    for (int j = 0; j < GGML_MAX_SRC; ++j) {
        const ggml_tensor * src = node->src[j];
        if (!src) break;

        key += '|';
        key += std::to_string((int) src->type);
        key += '[';
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            key += ',';
            key += std::to_string((long long) src->ne[d]);
        }
        key += ']';
    }
    key += '#';
    // key += std::string(reinterpret_cast<const char*>(node->op_params), sizeof(node->op_params));
    // base64 encode the op_params
    key += base64::encode(std::string(reinterpret_cast<const char*>(node->op_params), sizeof(node->op_params)));
    return key;
}
// static void pipo_op_recorder(ggml_cgraph * gf) {
//     static std::unordered_set<std::string> seen_ops;
//     for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
//         ggml_tensor * node = ggml_graph_node(gf, i);
//         if (!node) continue;

//         if (pipo_is_view_op(node->op)) continue;

//         const std::string key = pipo_make_op_key(node);
//         if (!seen_ops.insert(key).second) continue;

//         fprintf(stdout, "\nop_key[%zu]: ", key.size());
//         fwrite(key.data(), 1, key.size(), stdout);
//     }
// }
pipo_unique_op::pipo_unique_op(const ggml_tensor* t):
    op_type(t->op),
    node_type(t->type),
    op_shape(t->ne, t->ne + GGML_MAX_DIMS){
    src_types.reserve(GGML_MAX_SRC);
    src_nes.reserve(GGML_MAX_DIMS);
    for (int i = 0; i < GGML_MAX_SRC; i++){
        const ggml_tensor* src = t->src[i];
        if (!src) break;
        src_types.push_back(src->type);
        src_nes.emplace_back(src->ne, src->ne + GGML_MAX_DIMS);
    }
    op_param_bytes.resize(sizeof(t->op_params));
    memcpy(op_param_bytes.data(), t->op_params, sizeof(t->op_params));
}   
// pipo_unique_op::pipo_unique_op(const nlohmann::json & e) {
//     op_type                      = static_cast<ggml_op>(e.at("node").at("op").get<int>());
//     node_type                    = static_cast<ggml_type>(e.at("node").at("type").get<int>());
//     const auto & ne              = e.at("node").at("ne");
//     op_shape                     = vector<int64_t>(ne.begin(), ne.end());
//     string          op_param_raw = base64::decode(e.at("op_param").get<string>());
//     vector<uint8_t> op_param_bytes(op_param_raw.begin(), op_param_raw.end());
//     this->op_param_bytes = std::move(op_param_bytes);
//     const auto & srcs    = e.at("srcs");
//     for (const auto & src : srcs) {
//         src_types.push_back(static_cast<ggml_type>(src.at("type").get<int>()));
//         const auto & ne = src.at("ne");
//         src_nes.push_back(vector<int64_t>(ne.begin(), ne.end()));
//     }
// }

bool pipo_unique_op::operator==(const pipo_unique_op & other) const {
    return op_type == other.op_type && op_shape == other.op_shape && op_param_bytes == other.op_param_bytes &&
           src_types == other.src_types && src_nes == other.src_nes;
}

pipo_graph_info* pipo_get_graph_info(llama_context* ctx, std::unordered_set<std::string>* override_tensors){
    return ctx->get_graph_info(override_tensors);
}

bool pipo_is_view_op(enum ggml_op op) {
    switch (op) {
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            return false;
    }
}

ggml_tensor* pipo_unique_op::to_tensor(ggml_context* ctx) const{
    struct ggml_tensor * result = ggml_new_tensor(ctx, node_type, op_shape.size(), op_shape.data());
    result->op = op_type;
    vector<struct ggml_tensor *> src_tensors;
    src_tensors.resize(src_types.size());
    for (size_t i = 0; i < src_types.size(); i++) {
        src_tensors[i] = ggml_new_tensor(ctx, src_types[i], src_nes[i].size(), src_nes[i].data());
    }
    for (size_t i = 0; i < src_tensors.size(); i++) {
        result->src[i] = src_tensors[i];
    }
    for (size_t i = src_tensors.size(); i < GGML_MAX_SRC; i++) {
        result->src[i] = NULL;
    }
    memcpy(result->op_params, op_param_bytes.data(), op_param_bytes.size());
    return result;
}