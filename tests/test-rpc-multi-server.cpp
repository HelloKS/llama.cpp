#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-rpc.h"
#include "ggml.h"

#include <vector>

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
    ggml_backend_free(backend_b);
    ggml_backend_free(backend_a);
    return 0;
}
