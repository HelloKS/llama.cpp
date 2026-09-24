#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-rpc.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static ggml_backend_meta_split_state ssm_split(const ggml_tensor * tensor, void *) {
    if (strcmp(tensor->name, "ids") == 0) {
        return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    }
    if (strcmp(tensor->name, "xbc") == 0) {
        // Split whole head groups in each of the x, B, and C segments.
        return {GGML_BACKEND_SPLIT_AXIS_0, {512, 512, 256, 256}, {1, 2}, 2};
    }
    const auto axis = strcmp(tensor->name, "A") == 0 ? GGML_BACKEND_SPLIT_AXIS_1 : GGML_BACKEND_SPLIT_AXIS_0;
    return {axis, {tensor->ne[axis] / 2, tensor->ne[axis] / 2}, {1}, 1};
}

static void compare_ssm(ggml_tensor * actual, ggml_tensor * expected, int parts) {
    std::vector<float> a(ggml_nelements(actual)), b(a.size());
    ggml_backend_tensor_get(actual, a.data(), 0, ggml_nbytes(actual));
    ggml_backend_tensor_get(expected, b.data(), 0, ggml_nbytes(expected));
    const size_t n = a.size() / parts;
    for (int part = 0; part < parts; ++part) {
        double error = 0.0, norm = 0.0;
        for (size_t i = part * n; i < (part + 1) * n; ++i) {
            GGML_ASSERT(std::isfinite(a[i]) && std::isfinite(b[i]));
            const double delta = double(a[i]) - b[i];
            error += delta * delta;
            norm += double(b[i]) * b[i];
        }
        GGML_ASSERT(norm > 0.0);
        fprintf(stderr, "SSM tensor/RPC %s part %d NMSE %.3g\n", actual->name, part, error / norm);
        GGML_ASSERT(error / norm < 2e-7);
    }
}

static void test_ssm_tensor_rpc(ggml_backend_t backend_a, ggml_backend_t backend_b) {
    ggml_backend_dev_t devices[] = {ggml_backend_get_device(backend_a), ggml_backend_get_device(backend_b)};
    ggml_backend_t meta = ggml_backend_dev_init(ggml_backend_meta_device(devices, 2, ssm_split, nullptr), nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    GGML_ASSERT(meta != nullptr && cpu != nullptr);

    const int64_t d_state = 128, head_dim = 64, n_head = 16, n_group = 4, n_seq = 3;
    const int64_t d_inner = head_dim * n_head, state_size = d_state * d_inner;
    for (int64_t n_tok : {4, 512, 1027, 2048}) {
        const int K = n_tok == 512 ? 1 : 4;
        ggml_init_params params = {32 * ggml_tensor_overhead() + ggml_graph_overhead_custom(32, false), nullptr, true};
        ggml_context * inputs = ggml_init(params);
        ggml_context * ctx = ggml_init(params);
        ggml_tensor * state = ggml_new_tensor_2d(inputs, GGML_TYPE_F32, state_size, 5);
        ggml_tensor * xbc = ggml_new_tensor_2d(inputs, GGML_TYPE_F32, d_inner + 2 * d_state * n_group, n_tok * n_seq);
        ggml_tensor * dt = ggml_new_tensor_3d(inputs, GGML_TYPE_F32, n_head, n_tok, n_seq);
        ggml_tensor * A = ggml_new_tensor_2d(inputs, GGML_TYPE_F32, 1, n_head);
        ggml_tensor * ids = ggml_new_tensor_1d(inputs, GGML_TYPE_I32, n_seq);
        ggml_tensor * cache = ggml_new_tensor_3d(inputs, GGML_TYPE_F32, state_size, n_seq, K);
        ggml_tensor * tensors[] = {state, xbc, dt, A, ids, cache};
        const char * names[] = {"state", "xbc", "dt", "A", "ids", "cache"};
        for (size_t i = 0; i < 6; ++i) {
            ggml_set_name(tensors[i], names[i]);
        }
        ggml_backend_buffer_t input_buffer = ggml_backend_alloc_ctx_tensors(inputs, meta);
        GGML_ASSERT(input_buffer != nullptr);
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        for (ggml_tensor * tensor : tensors) {
            if (tensor == ids) {
                const int32_t values[] = {4, 0, 2};
                ggml_backend_tensor_set(tensor, values, 0, sizeof(values));
            } else {
                std::vector<float> values(ggml_nelements(tensor));
                for (float & value : values) {
                    value = tensor == A ? -0.01f + 0.005f * dist(rng) : dist(rng);
                }
                ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
            }
        }

        ggml_tensor * s = ggml_reshape_4d(ctx, state, d_state, head_dim, n_head, 5);
        ggml_tensor * x = ggml_view_4d(ctx, xbc, head_dim, n_head, n_tok, n_seq,
                                     head_dim * sizeof(float), xbc->nb[1], n_tok * xbc->nb[1], 0);
        ggml_tensor * B = ggml_view_4d(ctx, xbc, d_state, n_group, n_tok, n_seq,
                                     d_state * sizeof(float), xbc->nb[1], n_tok * xbc->nb[1], d_inner * sizeof(float));
        ggml_tensor * C = ggml_view_4d(ctx, xbc, d_state, n_group, n_tok, n_seq,
                                     d_state * sizeof(float), xbc->nb[1], n_tok * xbc->nb[1], (d_inner + d_state * n_group) * sizeof(float));
        ggml_tensor * scan = ggml_ssm_scan(ctx, s, x, dt, A, B, C, ids, K);
        ggml_set_name(scan, "ssm_scan");
        ggml_tensor * snapshots = ggml_view_3d(ctx, scan, state_size, n_seq, K,
                                              state_size * sizeof(float), state_size * n_seq * sizeof(float), ggml_nelements(x) * sizeof(float));
        snapshots = ggml_cpy(ctx, snapshots, cache);
        ggml_set_name(snapshots, "snapshots");
        ggml_tensor * y = ggml_cont(ctx, ggml_view_4d(ctx, scan, head_dim, n_head, n_tok, n_seq,
                                                   head_dim * sizeof(float), d_inner * sizeof(float), n_tok * d_inner * sizeof(float), 0));
        ggml_set_name(y, "y");
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
        ggml_build_forward_expand(graph, snapshots);
        ggml_build_forward_expand(graph, y);
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(meta));
        GGML_ASSERT(ggml_gallocr_alloc_graph(alloc, graph));
        auto reference = ggml_backend_graph_copy(cpu, graph);
        GGML_ASSERT(reference.buffer != nullptr);
        for (int repeat = 0; repeat < 2; ++repeat) {
            GGML_ASSERT(ggml_backend_graph_compute(meta, graph) == GGML_STATUS_SUCCESS);
            GGML_ASSERT(ggml_backend_graph_compute(cpu, reference.graph) == GGML_STATUS_SUCCESS);
            for (int i = 0; i < graph->n_nodes; ++i) {
                if (graph->nodes[i] == y || graph->nodes[i] == snapshots) {
                    compare_ssm(graph->nodes[i], reference.graph->nodes[i], graph->nodes[i] == y ? 1 : K);
                }
            }
        }
        fprintf(stderr, "SSM tensor/RPC n_tok=%lld K=%d passed\n", (long long)n_tok, K);
        ggml_backend_graph_copy_free(reference);
        ggml_gallocr_free(alloc);
        ggml_backend_buffer_free(input_buffer);
        ggml_free(ctx);
        ggml_free(inputs);
    }
    ggml_backend_free(cpu);
    ggml_backend_free(meta);
}

static void test_graph_cache(ggml_backend_t backend, const char * endpoint) {
    ggml_backend_t peer = ggml_backend_rpc_init(endpoint, 0);
    GGML_ASSERT(peer != nullptr);

    ggml_init_params params = {
        /* .mem_size   = */ 2*ggml_tensor_overhead() + ggml_graph_overhead_custom(1, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);
    ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor * output = ggml_scale(ctx, input, 2.0f);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 1, false);
    ggml_build_forward_expand(graph, output);
    graph->uid = ggml_graph_next_uid();
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer != nullptr);

    for (int i = 0; i < 4; ++i) {
        if (i == 2) {
            // Freeing any buffer invalidates all graphs on this connection.
            ggml_backend_buffer_t temporary = ggml_backend_alloc_buffer(peer, 64);
            GGML_ASSERT(temporary != nullptr);
            ggml_backend_buffer_free(temporary);
        }
        const float value = float(i + 1);
        ggml_backend_tensor_set(input, &value, 0, sizeof(value));
        GGML_ASSERT(ggml_backend_graph_compute(i % 2 ? peer : backend, graph) == GGML_STATUS_SUCCESS);
        float result = 0.0f;
        ggml_backend_tensor_get(output, &result, 0, sizeof(result));
        GGML_ASSERT(result == 2.0f*value);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(peer);
}

int main(int argc, char ** argv) {
    GGML_ASSERT(argc == 3);
    ggml_backend_load_all();

    const char * endpoint_a = argv[1];
    const char * endpoint_b = argv[2];

    ggml_backend_t backend_a = ggml_backend_rpc_init(endpoint_a, 0);
    ggml_backend_t backend_b = ggml_backend_rpc_init(endpoint_b, 0);
    GGML_ASSERT(backend_a != nullptr);
    GGML_ASSERT(backend_b != nullptr);

    test_graph_cache(backend_a, endpoint_a);
    test_ssm_tensor_rpc(backend_a, backend_b);

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() + ggml_graph_overhead_custom(1, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend_a);
    GGML_ASSERT(buffer != nullptr);

    // A remote pointer allocated by server A is not meaningful to server B.
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 1, false);
    graph->nodes[0] = tensor;
    graph->n_nodes = 1;

    GGML_ASSERT(ggml_backend_graph_compute(backend_b, graph) == GGML_STATUS_SUCCESS);
    // Wait for server B to finish the graph before the script checks its log.
    size_t free_mem;
    size_t total_mem;
    ggml_backend_rpc_get_device_memory(endpoint_b, 0, &free_mem, &total_mem);
    GGML_ASSERT(total_mem > 0);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend_b);
    ggml_backend_free(backend_a);
    return 0;
}
