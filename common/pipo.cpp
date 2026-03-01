#include "pipo.h"
#include <cstring>
#include <vector>

void pipo_tensor_layout(std::vector<llama_model_tensor_buft_override>& overrides,
                        ggml_backend_buffer_type_t cuda, ggml_backend_buffer_type_t cuda_host){

    overrides.clear();

    // qwen3-14B
    overrides.push_back({ "blk\\.([0-9]|[1-3][0-9])\\.ffn_down\\.weight", cuda_host });
    overrides.push_back({ "blk\\.([0-9]|[1-3][0-9])\\.ffn_up\\.weight", cuda_host });
    overrides.push_back({ "^token_embd\\.weight$", cuda_host });
    overrides.push_back({ ".*", cuda });

    // Terminate with nullptr
    overrides.push_back({ nullptr, nullptr });
}

void pipo_assign_offload(std::vector<const char*>& prefill_offload, std::vector<const char*>& decode_offload) {
    prefill_offload.clear();
    decode_offload.clear();
    
    // qwen3-14B
    prefill_offload.push_back("blk\\.([0-9]|[1-3][0-9])\\.ffn_down\\.weight");
    prefill_offload.push_back("blk\\.([0-9]|[1-3][0-9])\\.ffn_up\\.weight");
    decode_offload.push_back("blk\\.(2|5|8|11|14|17|20|23|26|29|32|35|38)\\.ffn_down\\.weight");
}
