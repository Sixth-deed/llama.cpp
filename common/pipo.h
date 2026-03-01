#pragma once
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <vector>

void pipo_tensor_layout(std::vector<llama_model_tensor_buft_override>& overrides,
                        ggml_backend_buffer_type_t cuda, ggml_backend_buffer_type_t cuda_host);

void pipo_assign_offload(std::vector<const char*>& prefill_offload, std::vector<const char*>& decode_offload);
