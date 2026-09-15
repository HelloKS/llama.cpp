# DeepSeek-V4.1-Flash vision

Convert the vision encoder and projector from the original checkpoint:

```sh
python3 convert_hf_to_gguf.py ds41 --mmproj --outtype bf16 --outfile mmproj-ds41-bf16.gguf
```

The converter writes the `deepseek41v` projector type. V4.1 uses a row-major image grid with a newline after each row and learned start/end embeddings. It does not use V4's interleaved rows or image padding. Use the V4.1 projector with the V4.1 language model.

Existing V4.1 target GGUFs produced by this branch already contain the image expert-routing biases. They do not need conversion or quantization again. Rebuild the server and RPC workers from the updated source; only the separate mmproj needs conversion.

For a target split across the local device and an RPC worker:

```sh
llama-server -m model-quant.gguf --mmproj mmproj-ds41-bf16.gguf \
    --rpc 192.168.100.11:50052 --lazy-mode on -c 8192 \
    --host 0.0.0.0 --port 8000
```

Send images through the server's usual image input or `image_url` chat content. To try a local image with the CLI:

```sh
llama-mtmd-cli -m model-quant.gguf --mmproj mmproj-ds41-bf16.gguf \
    --rpc 192.168.100.11:50052 --lazy-mode on -c 8192 \
    --image image.png -p "Describe this image."
```

The default image budget is 1024 tokens, including newline and start/end embeddings. Images use the vision routing bias and skip Engram lookups; subsequent text resumes Engram with its history cut at the image boundary. Language-model attention stays causal for image embeddings.

Implementation references: the official [vision encoder](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/vision.py), [image processor](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/image_processor.py), and [language model](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py).

## Bounded decoder prefill

Bounded decoder replay is enabled by default for DeepSeek-V4.1. Set `LLAMA_DSV41_CED=0` to use full decoder computation. This follows the approximate decoder replay described in sections 2.2 and 3.2.2 of the [technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf).

```sh
llama-server -m quant.gguf -md dspark.gguf --mmproj mmproj.gguf \
    --rpc 192.168.100.11:50052 --fit off --lazy-mode on \
    -c 262144 -b 2048 -ub 512 -fa on \
    --reasoning-preserve --spec-type draft-dspark --spec-draft-n-max 5 \
    --host 0.0.0.0 --port 8000
```

The encoder and decoder global KV projection process every token. The decoder attention and MoE layers process only the trailing attention window, extended for speculative rollback and the DSpark attention window. Decoder SWA is restricted to that retained segment. Requested logits are always retained; requests for every token's logits use the full computation. DSpark injects only the contiguous suffix of valid target features into its window cache.

Replay runs per physical microbatch, so `-ub` must be substantially larger than the replay window to save work. Increasing `-ub` also increases compute-buffer memory; reduce it if allocation fails. Mixed-sequence microbatches and non-contiguous or multi-dimensional positions use the full path. Unrecognized decoder source layouts and consumers that require every intermediate feature also use the full path. Existing prefix checkpoints still include SWA state; this does not implement global-KV-only prefix caching or encoder replay after discarding SWA.

The replay result is approximate and can depend on microbatch and cache boundaries. Compare quality on representative prompts before using it for workloads that require exact full-forward results. Log verbosity 4 shows encoder and decoder token counts when a replay graph is built. Rebuild both the server and RPC workers to include the CUDA indexer optimization.

The exact path also avoids index scoring when all keys and candidate blocks fit within their respective selection limits. CUDA indexer blocks whose mask is entirely negative infinity return before loading keys or computing scores. Dense mask storage, mask scanning, and top-k selection for larger histories remain.

Validation commands:

```sh
build/bin/test-llama-archs -a deepseek41 -s 17
build/bin/test-backend-ops test -b CUDA0 -o LIGHTNING_INDEXER -p 'sparse_k=3'
build/bin/llama-bench -m quant.gguf --rpc 192.168.100.11:50052 \
    -sm layer -ngl 999 --lazy-mode on -fa on -p 2048,8192 -n 0 -b 2048 -ub 512 -r 3
```

Run the benchmark once with `LLAMA_DSV41_CED=0` and once with the default, with the server stopped to release its allocations. These measurements isolate the target model; also check real-text server prompts with DSpark enabled. Synthetic token distributions can differ in Engram page locality and expert routing.

Large CPU row gathers request multiple graph threads, with one task per 256 requested rows up to the configured CPU thread count. Small gathers alone keep the graph single-threaded; other operations in the same graph can require more threads. This allows lazy Engram page reads to run concurrently during prefill without increasing the physical microbatch or allocating a second table cache. It still uses the OS page cache; more concurrent reads can increase the resident working set. Keep `-b` and `-ub` at values that leave room for DSpark loading.

### Explicit Engram reads

Use `--lazy-mode on-direct` to read Engram rows with the shared reader from [PR #28136](https://github.com/ggml-org/llama.cpp/pull/28136). This fork extends its Qwen and Gemma PLE support to DeepSeek-V4.1. Keep the rest of the server command, including `-b` and `-ub`, unchanged when comparing it with `--lazy-mode on`.

The reader sorts row requests, reuses duplicate rows within each worker's chunk, and uses concurrent buffered `pread()` calls. It dequantizes the rows to F32 before uploading each Engram input. Hashing, token history, CED, and DSpark feature handling are unchanged. This is an exact replacement for the Engram gather, not an additional approximation.

Startup must report `direct reads enabled for blk.N.engram_embd.weight` for each Engram table. If the file cannot be reopened or the platform lacks support, that table uses the existing mapped reads. Windows currently uses this fallback.

Despite its name, `on-direct` does not use `O_DIRECT` or bypass the OS page cache. It avoids mapped-page faults in the graph by reading the requested rows explicitly. Each Engram input also keeps an F32 host staging buffer of `n_tokens * hash_heads * head_dim * sizeof(float)` bytes; it does not cache the full table. Measure cold and warm real-text requests with DSpark enabled to check both throughput and memory headroom.

For prefill microbatches of at least 32 tokens, direct Engram reads start during input preparation and finish before each table's first consumer. Earlier layers can execute while reusable reader workers gather the rows. This is enabled by default, including with RPC tensor splitting, and reuses the existing F32 staging buffers. Set `LLAMA_DSV41_ENGRAM_ASYNC=0` on `llama-server` to use synchronous gathering for comparison. Small decode batches and unsupported graph paths keep synchronous gathering. No batch or context increase is required.

### Tensor splitting over two RPC workers

This fork includes [PR #26610](https://github.com/ggml-org/llama.cpp/pull/26610). Its pairwise reduction requires exactly two RPC devices on separate endpoints. Run an RPC worker on each Spark, including the Spark that runs `llama-server`, and select only the two RPC devices for the target model. A local CUDA device plus one RPC device uses the generic reduction path.

Build the client and both workers from the same source with `GGML_RPC=ON`; this change uses RPC protocol 8. Start each worker with `ggml-rpc-server --host 0.0.0.0 --port 50052`. Use the two mutually reachable RDMA addresses in the client's `--rpc` list, including the local worker's RDMA address rather than a loopback address. Rank 1 connects to rank 0 on the first endpoint's RPC port plus 1000, or port 51052 with this example. An unreachable peer can stall communicator initialization.

Replace the target server's split and device options with the following, substituting the addresses and device names reported by `--list-devices`:

```sh
--rpc SPARK_A_RDMA_IP:50052,SPARK_B_RDMA_IP:50052 --device RPC0,RPC1 -sm tensor -ts 1,1 -fa on
```

Keep context and batch sizes unchanged for the comparison. CED and `--lazy-mode on-direct` can stay enabled. Check for `pairwise communicator initialized` and RDMA negotiation on the worker-to-worker connection. `GGML_RPC_NO_COMM=1` disables the custom reduction for a separate fallback comparison.

Large F32 partials are sent as BF16 and restored before addition; this introduces rounding relative to F32 reductions. The implementation still uses host staging and synchronization, not NCCL or GPU-direct transfers. Test target-only output first, then DSpark, including retained features, cache reuse, and memory headroom. Local TCP checks do not establish GB10/RDMA performance or full DSpark compatibility.

### NCCL between RPC workers

The default automatically selects NCCL for large reductions when both workers support it, with pairwise RPC exchange as the initialization fallback. No environment variable is needed to enable this. To override the selection, set `GGML_RPC_ALLREDUCE` on the `llama-server` client; it negotiates the route with both workers. Both workers must use the same NCCL runtime version, at least 2.18, and be built with `GGML_CUDA=ON`, `GGML_CUDA_NCCL=ON`, and `GGML_RPC=ON`. Keep RDMA enabled. Rebuild and restart the client and both workers because protocol 8 is incompatible with protocol 7.

Supported modes:

- `pairwise`: existing direct worker-to-worker transport; disables NCCL selection.
- `nccl-exchange`: grouped NCCL send/receive for F32 reductions with at least 32,768 elements, preserving the existing BF16 wire conversion and local F32 addition.
- `nccl-f32`: native F32 NCCL all-reduce for the same large reductions. This changes wire traffic and numerical behavior relative to the BF16 exchange.
- `nccl-decode`: BF16 exchange for large reductions and native F32 NCCL all-reduce below 32,768 elements. Requires NCCL interface v2 on both workers; rebuild the client and both workers.
- `auto` (default): try `nccl-exchange`, then use pairwise only after coordinated initialization cleanup if NCCL is unavailable. Forced NCCL modes fail instead of silently falling back.

All modes except `nccl-decode` retain the current pairwise route for smaller reductions. `GGML_RPC_NO_COMM` continues to disable the custom communicator. Keep DSpark, Engram overlap, tile selection, batch, and context settings fixed when comparing PP and TG. The exchange mode aims to preserve the current per-rank result, which can differ between ranks because only the received partial is rounded to BF16. It does not introduce BF16 rounding of the local partial or the final sum.

For the experimental decode route, prefix the existing client command with `GGML_RPC_ALLREDUCE=nccl-decode`. Startup must report `small reductions use NCCL fp32 allreduce`. This choice is negotiated by the client, so no new environment setting is needed on the workers. The mode fails if either worker lacks support; it does not silently run the old route. Return to `GGML_RPC_ALLREDUCE=auto` for the previous behavior.

Small NCCL reductions are enqueued between producer and consumer work on the worker's compute stream. They bypass the pairwise path's explicit GPU synchronization, host tensor readback, host exchange, tensor upload, and separate addition. They require no conversion scratch buffer. NCCL can still use internal network staging on GB10 and wait for enqueue progress; this is not a claim of GPUDirect RDMA or zero host overhead. Small reductions use F32 without BF16 rounding, while large reductions keep the existing exchange behavior. The existing watchdog and target/draft communicator sharing also apply.

Startup reports the selected NCCL mode and library version. Set `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS=INIT,NET` on both workers to verify that NCCL selects the ConnectX-7 RDMA path. Existing RPC RDMA negotiation alone does not prove the NCCL transport selection. Spark does not support conventional GPUDirect RDMA to CUDA allocations; use NCCL's supported network staging path rather than forcing GPUDirect settings.

Target and draft contexts on the same ordered RPC devices share the communicator. Loading DSpark reports `reusing communicator` instead of initializing NCCL again. The client releases the worker communicator only after its last context is freed.

For tensor splitting across RPC workers, the meta backend preserves worker subgraph IDs for its eight most recently rebuilt nonzero graph IDs. Repeated segments can use RPC graph recompute even after another segment ran. This caches only IDs, uses the workers' existing graph storage, and does not allocate another set of model weights or compute buffers. Local meta graph preparation still runs when segments alternate. Set `GGML_META_RPC_GRAPH_CACHE=0` on the client to disable ID reuse across rebuilds.

RPC also recognizes identical graph definitions after upstream rebuilds assign new IDs. Matching ignores graph IDs and tensor object addresses, but compares serialized operations, parameters, shapes, strides, buffer/data addresses, flags, use counts, and graph connectivity. A hash selects candidates; the full definition must match before reuse. Tensor contents still come from the current buffers. The client retains at most 256 definitions and 32 MiB per remote device and invalidates them when a buffer is freed or the worker graph cache rolls over. Zero-ID graphs remain uncached. This is enabled by default; set `GGML_RPC_GRAPH_CACHE=0` on the client to disable definition matching. `GGML_RPC_DEBUG=1` reports `graph definition reused` on a match. Only the client needs rebuilding for definition matching; the worker UID and warmup diagnostics described below require rebuilt workers.

RPC workers assign cached graphs a stable worker-local UID. CUDA warmup messages include that UID, node count, and first node's operation and name. Repeated resets on a stable model segment warrant investigation; resets with UID zero may come from temporary pairwise reduction graphs. Different prompt sizes, cache lengths, and draft batches can still require different graphs. Rebuild the client and both workers for these changes.

Communication uses the worker's CUDA compute stream and reusable scratch buffers. A progress watchdog terminates the worker on an asynchronous failure or 60 seconds without completion progress. Restart both workers and the client after such a failure; the implementation does not replay an in-flight reduction through a fallback. The timeout also applies to NCCL initialization, but does not change the existing pairwise socket bootstrap timeout behavior.

For a direct numerical check after rebuilding, run `GGML_RPC_ALLREDUCE=nccl-decode build/bin/test-rpc-multi-server SPARK_A_RDMA_IP:50052 SPARK_B_RDMA_IP:50052` against idle workers, then repeat with `nccl-exchange` and `nccl-f32` if comparing modes. This checks fractional inputs, decode-sized tensors, both sides of the reduction threshold, graph reuse, and deferred input copies. It also queues 80 dependent small reductions without a client readback, alternating target and draft communicator handles, to check stream ordering and event-ring reuse. Do not run this test concurrently with the model server. CPU/Metal builds and local CPU RPC tests do not validate CUDA/NCCL execution or prove a TG gain on GB10.

### Experimental SM12x Q2_K MoE tile selection

By default, Q2_K MoE tile width uses the average routed tokens per expert on SM120/121, equivalent to `GGML_CUDA_Q2_K_MOE_NCOLS=-1`. Values `32`, `64`, and `128` instead supply a fixed token-column target to the existing tile selector. Set `0` to restore the original selection. This affects only MoE batches of at least 32 tokens and retains the existing weight and activation layouts.

For RPC execution, rebuild and restart **both RPC workers** from this source. No environment variable is needed for the default; set overrides on both workers. Setting it only on the client does not change the workers' CUDA kernels. Keep DSpark, Engram overlap, batch, and context settings unchanged. CUDA correctness and GB10 performance must be checked on the workers; a CPU or Metal build does not validate this path.

### Compact Q2_K expert scheduling

On SM120/121, Q2_K MoE MMQ batches of at least 32 tokens use compact scheduling by default. A GPU kernel builds a list of occupied expert/token tiles from the current routing bounds. Persistent blocks process these tiles through the existing Q2_K tile math, using the full reduction dimension per tile. This removes empty expert tiles and stream-K partial-result fixups from this path. Smaller batches and unsupported shapes retain the original scheduler.

The list is rebuilt on the same stream on every execution, including CUDA graph replay. No routing data is read back to the CPU. Scratch storage is bounded by `8 * (total_expert_assignments / tile_width + expert_count) + 4` bytes before allocator rounding; at 2048 tokens, top-6 routing, 384 experts, and tile width 32, this is about 6 KiB. No extra weight copy is stored.

Set `GGML_CUDA_Q2_K_MOE_COMPACT=0` on both RPC workers to use the original scheduler while retaining the current tile-width setting. This switch is independent of `GGML_CUDA_Q2_K_MOE_NCOLS`; set both to `0` to restore both original choices. A GPU trace shows `mmq_make_expert_tiles` followed by `mul_mat_q2_k_expert_tiles` when compact scheduling runs. End-to-end PP improvement requires measurement; this change does not reduce Q2_K multiplication work.

The existing backend test covers balanced and skewed routing, empty experts, output-row and token tails, broadcast and per-expert activations, and the small-batch boundary:

```sh
./build/bin/test-backend-ops test -b CUDA0 -o MUL_MAT_ID -p 'type_a=q2_K,.*routing=(balanced|skewed)'
```

### DSpark backend top-k with tensor splitting

Tensor mode permits a top-k-only backend sampler for DeepSeek V4/V4.1 and their DFlash/DSpark drafts when the actual output projection has mirrored meta storage. A draft that shares its target's output projection checks that target tensor. Sharded output heads, other sampler chains, and unsupported backend operations retain the CPU fallback.

DSpark already requests a top-10 backend chain. The workers compute the candidate set and return ten I32 token IDs plus ten F32 logits per draft output row (80 bytes of payload) instead of a full vocabulary row. The CPU still sorts the small candidate set and makes the final draft selection. This targets drafting/TG overhead; it does not accelerate the prompt's transformer computation.

Rebuild the client and RPC workers. No new option is required when draft backend sampling is enabled. Look for `tensor backend top-k enabled for seq_id=... (replicated output head)` in the client log. Use `--no-spec-draft-backend-sampling` on the client to compare with the CPU sampler, keeping other settings unchanged.

The model-free test executes the real sampler graph on a two-rank CPU meta backend, checks candidate IDs and scores for multiple output rows, and reuses graphs with changing logits:

```sh
./build/bin/test-backend-sampler --test tensor_top_k
```

It can also use device 0 from each of two RPC workers:

```sh
./build/bin/test-backend-sampler --test tensor_top_k --rpc 192.168.100.10:50052,192.168.100.11:50052
```

CPU meta and two local CPU RPC workers pass this test. CUDA/RDMA execution and the full DSpark workload still require validation on the Sparks.

## Investigating slow prompt processing

Throughput alone does not distinguish GPU kernels, CPU work, page faults, or RPC waits. The CPU compute-buffer size also does not show how much time runs on the CPU. Collect the following with no other inference requests running.

### Uncached real-text requests

Start with DSpark disabled: remove `-md`, `--spec-type`, and the draft-specific options from the server command. Keep the target model, layer placement, context size, batch sizes, and lazy mode the same. Use `--perf -lv 4` and leave CED at its enabled default. Save the full startup and request logs. For a separate placement capture, also set `GGML_SCHED_DEBUG=2`; it prints graph splits and node/backend assignments, not execution times. Disable this verbose graph dump for timing and Nsight captures.

Use a representative UTF-8 text file with at least 2048 tokens:

```sh
python3 scripts/debug-prefill.py prompt.txt --tokens 2048 | tee pp-target.jsonl
```

This sends one warmup and two measured requests to `http://127.0.0.1:8000`. Every request disables prompt-cache reuse and generates only one token. The output includes server prompt timings, request timestamps, and a hash of the token sequence. It rejects reused or truncated prompts. Warmup can populate OS page caches even though KV reuse is disabled. The script uses Python's standard library and does not print prompt or generated text.

Repeat with DSpark enabled and the same other settings, saving `pp-dspark.jsonl`. Compare records with the same token hash and `valid: true`. Start with 2048 tokens; use `--tokens 8192` later to check scaling. Do not repeat one short sentence to fill the prompt: that changes Engram page locality and expert routing.

Check startup for model/quantization metadata, layer placement, `n_ubatch`, flash attention, and `RDMA activated`. Check for `bounded decoder replay enabled` and `CED encoder tokens = ..., decoder tokens = ...`. The latter is printed when a replay graph is built, not on every reused graph. No CED graph message can mean the replay path was ineligible; the startup enable message alone does not prove that tokens were skipped. Compare the decoder count with and without DSpark, since its retained window can reduce the saving.

### CPU and storage activity

On each Linux node, substitute the PID of its server or RPC worker and collect during the requests in separate terminals. `pidstat` and `iostat` are supplied by `sysstat`.

```sh
pidstat -u -r -d -p PID 1 > pp-process.txt
iostat -xz 1 > pp-disk.txt
vmstat 1 > pp-memory.txt
```

Stop these monitors with Ctrl+C after the requests. Keep the files from each node separately. Ignore the first since-boot interval from `iostat` and `vmstat`. Major faults, process reads, and disk latency that coincide with prompt stalls support a storage bottleneck. Minor faults alone do not prove disk I/O. Swap activity indicates memory pressure. Busy CPU time with little I/O needs a CPU profile or graph placement inspection.

With lazy mode, Engram tables use a CPU buffer and mapped rows. Their reads can fault on demand. Do not turn lazy mode off until memory headroom is known: loading the complete tables changes the memory requirement substantially.

### CUDA timeline on both nodes

Use an installed Nsight Systems CLI compatible with the GB10 driver. Launch both processes under Nsight so that each GPU is captured. In the following commands, replace `YOUR_RPC_WORKER_COMMAND` and `YOUR_SERVER_COMMAND` with the complete commands normally used on those nodes. Stop the existing instances before relaunching.

On the RPC node:

```sh
nsys launch --session-new=pp-rpc --trace=cuda,nvtx,osrt --cuda-graph-trace=node \
    YOUR_RPC_WORKER_COMMAND
```

On the server node, first use the target-only configuration from above:

```sh
nsys launch --session-new=pp-main --trace=cuda,nvtx,osrt --cuda-graph-trace=node \
    YOUR_SERVER_COMMAND
```

Wait for model loading to finish and run one unrecorded request:

```sh
python3 scripts/debug-prefill.py prompt.txt --tokens 2048 --warmup 0 --runs 1
```

Start capture on the RPC node with `nsys start --session=pp-rpc --sample=none --cpuctxsw=none -o pp-rpc`, then on the server node with `nsys start --session=pp-main --sample=none --cpuctxsw=none -o pp-main`. These sampling controls belong to `start`; Nsight Systems 2025.3 rejects them on `launch`. Run one more request on the server node:

```sh
python3 scripts/debug-prefill.py prompt.txt --tokens 2048 --warmup 0 --runs 1 | tee pp-profile.jsonl
```

Stop capture on each respective node and export its summary:

```sh
# Server node
nsys stop --session=pp-main
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,osrt_sum pp-main.nsys-rep > pp-main-stats.txt
nsys shutdown --session=pp-main --kill=none

# RPC node
nsys stop --session=pp-rpc
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,osrt_sum pp-rpc.nsys-rep > pp-rpc-stats.txt
nsys shutdown --session=pp-rpc --kill=none
```

These commands follow the [Nsight Systems interactive CLI workflow](https://docs.nvidia.com/nsight-systems/UserGuide/index.html). Node-level CUDA graph tracing exposes individual kernels but adds overhead; use the unprofiled measurements for throughput. Keep both `.nsys-rep` files as well as the summaries. CPU sampling is disabled in this first capture; OS runtime tracing alone cannot identify all CPU work or RDMA polling.

| Observation during the request | Next investigation |
| --- | --- |
| GPU kernel activity fills most of the request interval | Rank MoE matmul, attention, indexer, and conversion kernels by total duration. |
| Both GPUs have long idle gaps and disk reads/major faults rise | Investigate lazy Engram reads and memory pressure. |
| GPU gaps coincide with CPU compute but little storage I/O | Inspect CPU-assigned graph nodes and capture a CPU profile. |
| Copies and waits dominate between graph splits | Inspect RPC staging, transfer sizes, round trips, and dependency scheduling. |
| DSpark increases decoder replay count or feature transfers | Attribute its extra prompt time to retained target work, feature extraction/copies, and draft processing separately. |

Layer splitting can make the two GPUs take turns; this alone is not evidence of a transfer problem. A long CUDA synchronization call can also be waiting for useful kernels. Compare the GPU timeline and CPU waits together; do not add overlapping API and kernel durations. This RPC backend has no asynchronous tensor-copy hook, and its Linux RDMA transport uses host buffers and 256 KiB chunks. Link bandwidth alone therefore does not rule out communication overhead.

Return the full server startup/request log, both process/storage captures, the JSONL results, and both Nsight summaries first. Include the exact commands and binary versions from both nodes. If the summaries cannot explain the elapsed time, inspect the request interval in the two timeline reports.
