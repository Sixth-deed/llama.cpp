#pragma once

#include <ggml.h>

int init_tensor_cuda_simple(
    ggml_tensor* tensor,
    float        min,
    float        max,
    unsigned long long seed = 0);

bool init_tensor_cuda_support_type(ggml_type type);