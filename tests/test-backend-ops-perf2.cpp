#include "base64.hpp"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>
using namespace std;

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

/* json utils */
static string read_file(const string & path) {
    cerr << "# Reading: " << path << '\n' << flush;
    ifstream fs(path, ios_base::binary);
    if (!fs.is_open()) {
        fs = ifstream("../" + path, ios_base::binary);
        if (!fs.is_open()) {
            throw runtime_error("Failed to open file: " + path);
        }
    }
    fs.seekg(0, ios_base::end);
    auto size = fs.tellg();
    fs.seekg(0);
    string out;
    out.resize(static_cast<size_t>(size));
    fs.read(out.data(), static_cast<streamsize>(size));
    return out;
}

static void debug_print_whole_json(const nlohmann::json & j) {
    for (const auto & e : j) {
        int    op_id        = e.at("node").at("op").get<int>();
        string op_param_b64 = e.at("op_param").get<string>();

        string          op_param_raw = base64::decode(op_param_b64);  // binary bytes
        vector<uint8_t> op_param_bytes(op_param_raw.begin(), op_param_raw.end());

        cerr << "op_id: " << op_id << '\n';
        cerr << "op_param_bytes: " << op_param_bytes.size() << '\n';
        for (size_t i = 0; i < op_param_bytes.size(); i++) {
            cerr << "op_param_bytes[" << i << "]: " << op_param_bytes[i] << '\n';
        }
        cerr << '\n';

        // srcs
        const auto & srcs = e.at("srcs");
        for (const auto & src : srcs) {
            int          type = src.at("type").get<int>();
            const auto & ne   = src.at("ne");
            cerr << "src_type: " << type << '\n';
            cerr << "src_ne: " << ne.size() << '\n';
            for (size_t i = 0; i < ne.size(); i++) {
                cerr << "src_ne[" << i << "]: " << ne[i] << '\n';
            }
            cerr << '\n';
        }
    }
}

/* tensor random utils
    refer to test-backend-ops.cpp
*/
#ifdef __EMSCRIPTEN__
#    define N_THREADS 1
#else
#    define N_THREADS std::thread::hardware_concurrency()
#endif
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <random>
#include <thread>
#include <vector>

static void init_tensor_uniform(ggml_tensor * tensor,
                                float         min     = -1.0f,
                                float         max     = 1.0f,
                                int64_t       int_min = 0,
                                int64_t       int_max = 100) {
    size_t nels = ggml_nelements(tensor);

    // 处理整数类型
    if (tensor->type == GGML_TYPE_I8 || tensor->type == GGML_TYPE_I16 || tensor->type == GGML_TYPE_I32 ||
        tensor->type == GGML_TYPE_I64) {
        // 为整数类型创建对应大小的缓冲区
        size_t element_size = 0;
        switch (tensor->type) {
            case GGML_TYPE_I8:
                element_size = sizeof(int8_t);
                break;
            case GGML_TYPE_I16:
                element_size = sizeof(int16_t);
                break;
            case GGML_TYPE_I32:
                element_size = sizeof(int32_t);
                break;
            case GGML_TYPE_I64:
                element_size = sizeof(int64_t);
                break;
            default:
                break;
        }

        std::vector<uint8_t> data(nels * element_size);

        {
            // 并行初始化整数
            static const size_t                            n_threads  = N_THREADS;
            static std::vector<std::default_random_engine> generators = []() {
                std::random_device                      rd;
                std::vector<std::default_random_engine> vec;
                vec.reserve(n_threads);
                for (size_t i = 0; i < n_threads; i++) {
                    vec.emplace_back(rd());
                }
                return vec;
            }();

            auto init_thread = [&](size_t ith, size_t start, size_t end) {
                // 使用int64_t生成随机整数，然后转换为目标类型
                std::uniform_int_distribution<int64_t> distribution(int_min, int_max);
                auto &                                 gen = generators[ith];

                switch (tensor->type) {
                    case GGML_TYPE_I8:
                        {
                            int8_t * ptr = reinterpret_cast<int8_t *>(data.data());
                            for (size_t i = start; i < end; i++) {
                                ptr[i] = static_cast<int8_t>(distribution(gen));
                            }
                            break;
                        }
                    case GGML_TYPE_I16:
                        {
                            int16_t * ptr = reinterpret_cast<int16_t *>(data.data());
                            for (size_t i = start; i < end; i++) {
                                ptr[i] = static_cast<int16_t>(distribution(gen));
                            }
                            break;
                        }
                    case GGML_TYPE_I32:
                        {
                            int32_t * ptr = reinterpret_cast<int32_t *>(data.data());
                            for (size_t i = start; i < end; i++) {
                                ptr[i] = static_cast<int32_t>(distribution(gen));
                            }
                            break;
                        }
                    case GGML_TYPE_I64:
                        {
                            int64_t * ptr = reinterpret_cast<int64_t *>(data.data());
                            for (size_t i = start; i < end; i++) {
                                ptr[i] = distribution(gen);
                            }
                            break;
                        }
                    default:
                        break;
                }
            };

            if (n_threads == 1) {
                init_thread(0, 0, nels);
            } else {
                std::vector<std::future<void>> tasks;
                tasks.reserve(n_threads);
                for (size_t i = 0; i < n_threads; i++) {
                    size_t start = i * nels / n_threads;
                    size_t end   = (i + 1) * nels / n_threads;
                    tasks.push_back(std::async(std::launch::async, init_thread, i, start, end));
                }
                for (auto & t : tasks) {
                    t.get();
                }
            }
        }

        // 设置张量数据
        ggml_backend_tensor_set(tensor, data.data(), 0, data.size());
    }
    // 处理浮点数类型
    else if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> data(nels);
        {
            // parallel initialization
            static const size_t                            n_threads  = N_THREADS;
            static std::vector<std::default_random_engine> generators = []() {
                std::random_device                      rd;
                std::vector<std::default_random_engine> vec;
                vec.reserve(n_threads);
                for (size_t i = 0; i < n_threads; i++) {
                    vec.emplace_back(rd());
                }
                return vec;
            }();

            auto init_thread = [&](size_t ith, size_t start, size_t end) {
                std::uniform_real_distribution<float> distribution(min, max);
                auto &                                gen = generators[ith];
                for (size_t i = start; i < end; i++) {
                    data[i] = distribution(gen);
                }
            };

            if (n_threads == 1) {
                init_thread(0, 0, nels);
            } else {
                std::vector<std::future<void>> tasks;
                tasks.reserve(n_threads);
                for (size_t i = 0; i < n_threads; i++) {
                    size_t start = i * nels / n_threads;
                    size_t end   = (i + 1) * nels / n_threads;
                    tasks.push_back(std::async(std::launch::async, init_thread, i, start, end));
                }
                for (auto & t : tasks) {
                    t.get();
                }
            }
        }
        ggml_backend_tensor_set(tensor, data.data(), 0, nels * sizeof(float));
    }
    // 处理量化类型
    else if (ggml_is_quantized(tensor->type) || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) {
        GGML_ASSERT(nels % ggml_blck_size(tensor->type) == 0);

        std::vector<float> data(nels);
        {
            // parallel initialization
            static const size_t                            n_threads  = N_THREADS;
            static std::vector<std::default_random_engine> generators = []() {
                std::random_device                      rd;
                std::vector<std::default_random_engine> vec;
                vec.reserve(n_threads);
                for (size_t i = 0; i < n_threads; i++) {
                    vec.emplace_back(rd());
                }
                return vec;
            }();

            auto init_thread = [&](size_t ith, size_t start, size_t end) {
                std::uniform_real_distribution<float> distribution(min, max);
                auto &                                gen = generators[ith];
                for (size_t i = start; i < end; i++) {
                    data[i] = distribution(gen);
                }
            };

            if (n_threads == 1) {
                init_thread(0, 0, nels);
            } else {
                std::vector<std::future<void>> tasks;
                tasks.reserve(n_threads);
                for (size_t i = 0; i < n_threads; i++) {
                    size_t start = i * nels / n_threads;
                    size_t end   = (i + 1) * nels / n_threads;
                    tasks.push_back(std::async(std::launch::async, init_thread, i, start, end));
                }
                for (auto & t : tasks) {
                    t.get();
                }
            }
        }

        // 量化处理
        std::vector<float> imatrix(tensor->ne[0], 1.0f);
        const float *      im = imatrix.data();
        if (!ggml_quantize_requires_imatrix(tensor->type)) {
            if (data[0] > 0.5f * (min + max)) {
                im = nullptr;
            }
        }

        std::vector<uint8_t> dataq(ggml_row_size(tensor->type, nels));
        {
            // parallel quantization by block
            size_t blck_size = ggml_blck_size(tensor->type);
            size_t n_blocks  = nels / blck_size;

            auto quantize_thread = [&](size_t start, size_t end) {
                ggml_quantize_chunk(tensor->type, data.data(), dataq.data(), start * blck_size, end - start, blck_size,
                                    im);
            };

            const size_t min_blocks_per_thread = 1;
            const size_t n_quant_threads       = std::min<size_t>(std::max<size_t>(N_THREADS / 2, 1),
                                                                  std::max<size_t>(1, n_blocks / min_blocks_per_thread));

            if (n_quant_threads == 1) {
                quantize_thread(0, n_blocks);
            } else {
                std::vector<std::future<void>> tasks;
                tasks.reserve(n_quant_threads);
                for (size_t i = 0; i < n_quant_threads; i++) {
                    size_t start = i * n_blocks / n_quant_threads;
                    size_t end   = (i + 1) * n_blocks / n_quant_threads;
                    tasks.push_back(std::async(std::launch::async, quantize_thread, start, end));
                }
                for (auto & t : tasks) {
                    t.get();
                }
            }
        }
        ggml_backend_tensor_set(tensor, dataq.data(), 0, dataq.size());
    } else {
        GGML_ABORT("Unsupported tensor type in init_tensor_uniform");
    }
}

/* unique op struct */
struct UniqueOp {
    ggml_op                 op_type;
    ggml_type               node_type;
    vector<int64_t>         op_shape;
    vector<uint8_t>         op_param_bytes;
    vector<ggml_type>       src_types;
    vector<vector<int64_t>> src_nes;

    void debug_print() {
        cerr << "op_type: " << op_type << '\n';
        cerr << "node_type: " << node_type << '\n';
        cerr << "op_shape: " << op_shape.size() << '\n';
        for (size_t i = 0; i < op_shape.size(); i++) {
            cerr << "op_shape[" << i << "]: " << op_shape[i] << '\n';
        }
        cerr << '\n';
        cerr << "op_param_bytes: " << op_param_bytes.size() << '\n';
        for (size_t i = 0; i < op_param_bytes.size(); i++) {
            cerr << "op_param_bytes[" << i << "]: " << op_param_bytes[i] << '\n';
        }
        cerr << '\n';
        cerr << "src_types: " << src_types.size() << '\n';
        for (size_t i = 0; i < src_types.size(); i++) {
            cerr << "src_types[" << i << "]: " << src_types[i] << '\n';
        }
        cerr << '\n';
        cerr << "src_nes: " << src_nes.size() << '\n';
        for (size_t i = 0; i < src_nes.size(); i++) {
            cerr << "src_nes[" << i << "]: " << src_nes[i].size() << '\n';
            for (size_t j = 0; j < src_nes[i].size(); j++) {
                cerr << "src_nes[" << i << "][" << j << "]: " << src_nes[i][j] << '\n';
            }
            cerr << '\n';
        }
        cerr << '\n';
    }

    string op_key() const {
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
        key +=
            base64::encode(std::string(reinterpret_cast<const char *>(op_param_bytes.data()), op_param_bytes.size()));
        return key;
    }

    string short_desc() const {
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

    UniqueOp(const nlohmann::json & e) {
        op_type                      = static_cast<ggml_op>(e.at("node").at("op").get<int>());
        node_type                    = static_cast<ggml_type>(e.at("node").at("type").get<int>());
        const auto & ne              = e.at("node").at("ne");
        op_shape                     = vector<int64_t>(ne.begin(), ne.end());
        string          op_param_raw = base64::decode(e.at("op_param").get<string>());
        vector<uint8_t> op_param_bytes(op_param_raw.begin(), op_param_raw.end());
        this->op_param_bytes = std::move(op_param_bytes);
        const auto & srcs    = e.at("srcs");
        for (const auto & src : srcs) {
            src_types.push_back(static_cast<ggml_type>(src.at("type").get<int>()));
            const auto & ne = src.at("ne");
            src_nes.push_back(vector<int64_t>(ne.begin(), ne.end()));
        }
    }

    UniqueOp() :
        op_type(GGML_OP_NONE),
        node_type(GGML_TYPE_F32),
        op_shape(0),
        op_param_bytes(0),
        src_types(0),
        src_nes(0) {}

    bool operator==(const UniqueOp & other) const {
        return op_type == other.op_type && op_shape == other.op_shape && op_param_bytes == other.op_param_bytes &&
               src_types == other.src_types && src_nes == other.src_nes;
    }

    bool operator!=(const UniqueOp & other) const { return !(*this == other); }
};

/* single test result */
struct SingleTestResult {
    const UniqueOp & op;
    ggml_backend_t   backend;
    int              batch_size;
    size_t           transfer_bytes;
    double           transfer_ms;
    double           compute_ms;
};

static SingleTestResult run_single_test(const UniqueOp & op, ggml_backend_t backend, int batch_size, int n_iter) {
    size_t                       ctx_size    = 1024 * 1024 * 64;  // 足以容纳图节点
    struct ggml_init_params      init_params = { ctx_size, NULL, true };
    struct ggml_context *        ctx         = ggml_init(init_params);
    // 2. 创建 src tensors
    vector<struct ggml_tensor *> src_tensors;
    src_tensors.resize(op.src_types.size());
    for (size_t i = 0; i < op.src_types.size(); i++) {
        src_tensors[i] = ggml_new_tensor(ctx, op.src_types[i], op.src_nes[i].size(), op.src_nes[i].data());
    }

    // 用于测量传输时间
    struct ggml_init_params      cpu_init_params = { ctx_size, NULL, true };
    struct ggml_context *        cpu_ctx         = ggml_init(cpu_init_params);
    vector<struct ggml_tensor *> cpu_src_tensors;
    cpu_src_tensors.resize(op.src_types.size());
    for (size_t i = 0; i < op.src_types.size(); i++) {
        cpu_src_tensors[i] = ggml_new_tensor(cpu_ctx, op.src_types[i], op.src_nes[i].size(), op.src_nes[i].data());
    }

    // 3. 创建 result tensor
    struct ggml_tensor * result = ggml_new_tensor(ctx, op.node_type, op.op_shape.size(), op.op_shape.data());

    // 4. 构建计算图
    result->op = op.op_type;
    for (size_t i = 0; i < src_tensors.size(); i++) {
        result->src[i] = src_tensors[i];
    }
    for (size_t i = src_tensors.size(); i < GGML_MAX_SRC; i++) {
        result->src[i] = NULL;
    }
    memcpy(result->op_params, op.op_param_bytes.data(), op.op_param_bytes.size());
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);

    if (!ggml_backend_supports_op(backend, result)) {
        cerr << "op " << op.short_desc() << " not supported by backend " << ggml_backend_name(backend) << '\n';
        ggml_free(ctx);
        return SingleTestResult{ op, backend, batch_size, 0, -1.0, -1.0 };
    }

    // 5. 后端分配
    ggml_backend_buffer_t buffer      = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_t        cpu_backend = ggml_backend_init_by_name("cpu", NULL);
    ggml_backend_buffer_t cpu_buffer  = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu_backend);
    if (!buffer) {
        cerr << "Failed to allocate buffer for " << op.op_type << '\n';
        ggml_free(ctx);
        return SingleTestResult{ op, backend, batch_size, 0, -1.0, -1.0 };
    }
    if (!cpu_buffer) {
        cerr << "Failed to allocate buffer for " << op.op_type << '\n';
        ggml_free(ctx);
        ggml_free(cpu_ctx);
        return SingleTestResult{ op, backend, batch_size, 0, -1.0, -1.0 };
    }

    // 5. 准备随机数据
    vector<ggml_tensor *> * tensors_view;
    if (strcmp(ggml_backend_name(backend), "CPU") == 0) {
        tensors_view = &src_tensors;
    } else {
        tensors_view = &cpu_src_tensors;
    }
    // GGML_OP_GET_ROWS 需要特殊处理, 要防止下标越界
    if (op.op_type == GGML_OP_GET_ROWS) {
        init_tensor_uniform(tensors_view->at(0));
        init_tensor_uniform(tensors_view->at(1), 0, 0, 0, tensors_view->at(0)->ne[1] - 1);
    } else {
        for (size_t i = 0; i < tensors_view->size(); i++) {
            init_tensor_uniform(tensors_view->at(i));
        }
    }

    // 6. 传输数据
    size_t  transfer_bytes   = 0;
    int64_t t_transfer_start = ggml_time_us();
    if (strcmp(ggml_backend_name(backend), "CPU") != 0) {
        for (size_t i = 0; i < cpu_src_tensors.size(); i++) {
            ggml_backend_tensor_copy(cpu_src_tensors[i], src_tensors[i]);
            transfer_bytes += ggml_nbytes(src_tensors[i]);
        }
    }
    ggml_backend_synchronize(backend);
    int64_t t_transfer_end = ggml_time_us();
    double  transfer_ms    = (t_transfer_end - t_transfer_start) / 1000.0;

    // warmup
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);

    // 7. 执行计算图
    int64_t t_compute_start = ggml_time_us();
    for (int i = 0; i < n_iter; i++) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);

    int64_t t_compute_end = ggml_time_us();
    double  compute_ms    = (t_compute_end - t_compute_start) / 1000.0 / n_iter;

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return SingleTestResult{ op, backend, batch_size, transfer_bytes, transfer_ms, compute_ms };
}

/* main */
int main(int argc, char ** argv) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <json_file>\n";
        return 1;
    }
    string           json_file = argv[1];
    string           json_text = read_file(json_file);
    nlohmann::json   j         = nlohmann::json::parse(json_text);  // j is an array
    // debug_print_whole_json(j);
    vector<UniqueOp> ops;
    ops.resize(j.size());
    for (size_t i = 0; i < j.size(); i++) {
        ops[i] = UniqueOp(j[i]);
    }

    ggml_backend_t cpu_backend = ggml_backend_init_by_name("cpu", NULL);
    ggml_backend_t gpu_backend = NULL;
    size_t         dev_count   = ggml_backend_dev_count();
    for (size_t i = 0; i < dev_count; ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu_backend = ggml_backend_dev_init(d, NULL);
            break;
        }
    }

    vector<SingleTestResult> results;
    for (size_t i = 0; i < ops.size(); i++) {
        cerr << "perf op: " << ops[i].short_desc() << '\n';
        results.push_back(run_single_test(ops[i], cpu_backend, 1, 10));
        if (gpu_backend) {
            results.push_back(run_single_test(ops[i], gpu_backend, 1, 10));
        }
    }

    // for (size_t i = 0; i < results.size(); i++) {
    //     cerr << "perf result[" << i << "]: " << ggml_backend_name(results[i].backend) << '\n';
    //     cerr << "op: " << results[i].op.short_desc() << '\n';
    //     cerr << "transfer_ms: " << results[i].transfer_ms << '\n';
    //     cerr << "compute_ms: " << results[i].compute_ms << '\n';
    //     cerr << '\n';
    // }

    // save as json
    nlohmann::json j_results;
    for (size_t i = 0; i < results.size(); i++) {
        j_results[i]["op"]     = results[i].op.op_key();
        j_results[i]["info"]   = results[i].op.short_desc();
        j_results[i]["result"] = {
            { "backend",        ggml_backend_name(results[i].backend) },
            { "transfer_bytes", results[i].transfer_bytes             },
            { "transfer_ms",    results[i].transfer_ms                },
            { "compute_ms",     results[i].compute_ms                 },
        };
    }
    cout << j_results.dump(4) << '\n';

    ggml_backend_free(cpu_backend);
    if (gpu_backend) {
        ggml_backend_free(gpu_backend);
    }

    return 0;
}
