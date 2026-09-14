#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-rpc.h"
#include "ggml.h"

#include <vector>

static void test_deferred_inputs(ggml_backend_t backend_a, ggml_backend_t backend_b, bool parallel) {
    auto cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    GGML_ASSERT(cpu);
    ggml_backend_t backends[] = {backend_a, backend_b, cpu};
    auto sched = ggml_backend_sched_new(backends, nullptr, 3, 64, parallel, false);
    ggml_init_params params = {16*ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false), nullptr, true};
    auto ctx = ggml_init(params);
    auto x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
    auto late_a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
    auto late_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
    auto first = ggml_scale(ctx, x, 2.0f);
    auto sum_a = ggml_add(ctx, first, late_a);
    auto sum_b = ggml_add(ctx, sum_a, late_b);
    for (auto t : {x, late_a, late_b}) {
        ggml_set_input(t);
        ggml_backend_sched_set_tensor_backend(sched, t, cpu);
    }
    for (auto t : {first, sum_a}) {
        ggml_set_output(t);
        ggml_backend_sched_set_tensor_backend(sched, t, backend_a);
    }
    ggml_set_output(sum_b);
    ggml_backend_sched_set_tensor_backend(sched, sum_b, backend_b);
    auto graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, sum_b);
    struct state {
        ggml_tensor * consumers[2];
        ggml_tensor * producers[2];
        ggml_tensor * inputs[2];
        int iteration = 0;
        int calls = 0;
        bool fail = false;
    } data = {{sum_a, sum_b}, {first, sum_a}, {late_a, late_b}};
    ggml_backend_sched_set_prepare_callback(sched, [](ggml_tensor * t, bool ask, void * opaque) {
        auto & s = *static_cast<state *>(opaque);
        for (int i = 0; i < 2; ++i) {
            if (t != s.consumers[i]) {
                continue;
            }
            if (ask) {
                return true;
            }
            if (s.fail) {
                return false;
            }
            GGML_ASSERT(s.calls++ == i);
            float values[16];
            ggml_backend_tensor_get(s.producers[i], values, 0, sizeof(values));
            for (float value : values) {
                GGML_ASSERT(value == 2.0f*(s.iteration + 1) + (i ? 10 + s.iteration : 0));
            }
            for (float & value : values) {
                value = 10*(i + 1) + s.iteration;
            }
            ggml_backend_tensor_set(s.inputs[i], values, 0, sizeof(values));
            return true;
        }
        return !ask;
    }, &data);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
    GGML_ASSERT(ggml_backend_sched_get_n_splits(sched) >= 3);
    for (int iteration = 0; iteration < 3; ++iteration) {
        data.iteration = iteration;
        data.calls = 0;
        std::vector<float> values(16, iteration + 1.0f);
        ggml_backend_tensor_set(x, values.data(), 0, values.size()*sizeof(float));
        GGML_ASSERT(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(data.calls == 2);
        ggml_backend_tensor_get(sum_b, values.data(), 0, values.size()*sizeof(float));
        for (float value : values) {
            GGML_ASSERT(value == 32 + 4*iteration);
        }
    }
    data.fail = true;
    GGML_ASSERT(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_FAILED);
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_free(cpu);
}

static void test_allreduce(ggml_backend_t backend_a, ggml_backend_t backend_b) {
    auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_a));
    auto comm_init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init");
    auto comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_free");
    auto allreduce = (ggml_backend_comm_allreduce_tensor_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_tensor");
    GGML_ASSERT(comm_init && comm_free && allreduce);
    ggml_backend_t backends[] = {backend_a, backend_b};
    void * comm = comm_init(backends, 2);
    GGML_ASSERT(comm);

    for (int64_t n : {16, 65536}) {
        ggml_context * contexts[2];
        ggml_backend_buffer_t buffers[2];
        ggml_tensor * inputs[2];
        ggml_tensor * outputs[2];
        ggml_cgraph * graphs[2];
        for (int rank = 0; rank < 2; ++rank) {
            ggml_init_params params = {3*ggml_tensor_overhead() + ggml_graph_overhead_custom(4, false), nullptr, true};
            contexts[rank] = ggml_init(params);
            inputs[rank] = ggml_new_tensor_1d(contexts[rank], GGML_TYPE_F32, n);
            outputs[rank] = ggml_scale(contexts[rank], inputs[rank], rank + 2.0f);
            graphs[rank] = ggml_new_graph_custom(contexts[rank], 4, false);
            ggml_build_forward_expand(graphs[rank], outputs[rank]);
            graphs[rank]->uid = ggml_graph_next_uid();
            buffers[rank] = ggml_backend_alloc_ctx_tensors(contexts[rank], backends[rank]);
            GGML_ASSERT(buffers[rank]);
        }

        // Changed inputs must reach reused graphs before their reductions.
        std::vector<float> values[2] = {std::vector<float>(n), std::vector<float>(n)};
        std::vector<float> result(n);
        for (int iteration = 0; iteration < 3; ++iteration) {
            for (int rank = 0; rank < 2; ++rank) {
                for (int64_t i = 0; i < n; ++i) {
                    values[rank][i] = i % 8 + iteration + rank;
                }
                ggml_backend_tensor_set_async(backends[rank], inputs[rank], values[rank].data(), 0, n*sizeof(float));
                GGML_ASSERT(ggml_backend_graph_compute_async(backends[rank], graphs[rank]) == GGML_STATUS_SUCCESS);
            }
            GGML_ASSERT(allreduce(comm, outputs));
            for (int rank = 0; rank < 2; ++rank) {
                ggml_backend_tensor_get_async(backends[rank], outputs[rank], result.data(), 0, n*sizeof(float));
                ggml_backend_synchronize(backends[rank]);
                for (int64_t i = 0; i < n; ++i) {
                    GGML_ASSERT(result[i] == 5.0f*(i % 8 + iteration) + 3.0f);
                }
            }
        }
        for (int rank = 0; rank < 2; ++rank) {
            ggml_backend_buffer_free(buffers[rank]);
            ggml_free(contexts[rank]);
        }
    }
    comm_free(comm);
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
    test_allreduce(backend_a, backend_b);
    test_deferred_inputs(backend_a, backend_b, false);
    test_deferred_inputs(backend_a, backend_b, true);
    ggml_backend_free(backend_b);
    ggml_backend_free(backend_a);
    return 0;
}
