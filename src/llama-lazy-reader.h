#pragma once

// Serves rows of a lazy tensor with explicit pread()s instead of demand paging.
// This works because the row indices of a whole ubatch are known host-side
// before the graph runs. Hands out F32 rows, like ggml_get_rows does.

#include "ggml.h"
#include "llama-impl.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

class llama_lazy_reader_pool {
public:
    explicit llama_lazy_reader_pool(int n_threads) {
        try {
            for (int i = 0; i < n_threads; ++i) {
                workers.emplace_back([this] {
                    for (;;) {
                        std::function<void()> task;
                        {
                            std::unique_lock<std::mutex> lock(mutex);
                            ready.wait(lock, [this] { return stopping || !tasks.empty(); });
                            if (tasks.empty()) {
                                return;
                            }
                            task = std::move(tasks.back());
                            tasks.pop_back();
                        }
                        task();
                    }
                });
            }
        } catch (...) {
            stop();
            throw;
        }
    }

    ~llama_lazy_reader_pool() { stop(); }

    void submit(std::vector<std::function<void()>> batch) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            GGML_ASSERT(tasks.empty());
            tasks = std::move(batch);
        }
        ready.notify_all();
    }

private:
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        ready.notify_all();
        for (auto & worker : workers) {
            worker.join();
        }
    }

    std::mutex mutex;
    std::condition_variable ready;
    std::vector<std::thread> workers;
    std::vector<std::function<void()>> tasks;
    bool stopping = false;
};

struct llama_lazy_read_stats {
    uint64_t calls = 0;
    uint64_t bytes = 0;
    int64_t read_us = 0;
    int64_t total_us = 0;
};

struct llama_lazy_reader {
#ifdef _WIN32
    // pread()/open() are unavailable on Windows; --lazy-mode on-direct falls
    // back to the lazy mmap reads there (see llama_model_base::load_lazy_reader)
    const int64_t head_dim = 0;
    const size_t page_size = 0;

    void gather(const int32_t *, int64_t, float *, llama_lazy_read_stats * = nullptr) const {
        GGML_ABORT("lazy direct reads are not supported on this platform");
    }
    std::shared_future<void> gather_async(const int32_t *, int64_t, float *, llama_lazy_read_stats * = nullptr) const {
        GGML_ABORT("lazy direct reads are not supported on this platform");
    }
#else
    llama_lazy_reader(int fd, size_t base, size_t row_size, int64_t n_rows, int n_threads,
                      enum ggml_type type, int64_t head_dim, size_t page_size = 0)
        : fd(fd), base(base), row_size(row_size), n_rows(n_rows), n_threads(n_threads),
          head_dim(head_dim), to_float(type == GGML_TYPE_F32 ? nullptr : ggml_get_type_traits(type)->to_float), page_size(page_size) {
        // F32 rows have no dequantizer; they are staged as-is, like ggml_get_rows
        GGML_ASSERT((type == GGML_TYPE_F32 || to_float != nullptr) && head_dim > 0);
    }

    llama_lazy_reader(const llama_lazy_reader &) = delete;
    llama_lazy_reader & operator=(const llama_lazy_reader &) = delete;

    ~llama_lazy_reader() {
        pool.reset();
        if (fd >= 0) {
            ::close(fd);
        }
    }

    const int      fd;
    const size_t   base;       // file offset of row 0
    const size_t   row_size;   // bytes per quantized row
    const int64_t  n_rows;
    const int      n_threads;  // in-flight read workers
    const int64_t  head_dim;   // F32 elements per staged row
    ggml_to_float_t to_float;  // same dequantizer the ggml_get_rows CPU kernel uses
    const size_t page_size;    // zero keeps one read per unique row

    // fill dst with the n gathered rows, dequantized to F32:
    // dst[slot * head_dim, ...) = to_float(table[rows[slot]])
    // thread-safe; never lets an exception escape a worker thread
    void gather(const int32_t * rows, int64_t n, float * dst, llama_lazy_read_stats * stats = nullptr) const {
        gather_async(rows, n, dst, stats).get();
    }

    // One batch per reader bounds queued work. The caller keeps dst and stats alive until completion.
    std::shared_future<void> gather_async(const int32_t * rows, int64_t n, float * dst, llama_lazy_read_stats * stats = nullptr) const {
        const int64_t start = stats ? ggml_time_us() : 0;
        std::unique_lock<std::mutex> lock(submit_mutex);
        if (pending.valid()) {
            pending.wait();
        }
        if (stats) {
            *stats = {};
        }
        std::vector<std::pair<int32_t, int32_t>> pairs; // (row, dst slot)
        pairs.reserve(n);
        for (int64_t i = 0; i < n; ++i) {
            GGML_ASSERT(rows[i] >= 0 && (int64_t) rows[i] < n_rows);
            pairs.emplace_back(rows[i], (int32_t) i);
        }

        std::sort(pairs.begin(), pairs.end()); // equal rows adjacent, file order

        std::vector<int64_t> boundaries;
        if (page_size) {
            boundaries.push_back(0);
            for (int64_t i = 0; i < n; ) {
                i = group_end(pairs, i, n);
                boundaries.push_back(i);
            }
        }
        const int64_t n_ranges = page_size ? (int64_t) boundaries.size() - 1 : n;
        const int n_workers = (int) std::min<int64_t>(n_threads, std::max<int64_t>(1, std::min(n / 32, n_ranges)));
        if (n_workers == 1) {
            run_range(pairs, 0, n, dst, stats);
            if (stats) {
                stats->total_us = ggml_time_us() - start;
            }
            std::promise<void> done;
            done.set_value();
            return done.get_future().share();
        }

        struct request {
            std::vector<std::pair<int32_t, int32_t>> pairs;
            std::vector<std::exception_ptr> errors;
            std::vector<int64_t> boundaries;
            std::vector<llama_lazy_read_stats> stats;
            std::atomic<int> remaining;
            std::promise<void> done;
            explicit request(int count) : errors(count), remaining(count) {}
        };
        auto work = std::make_shared<request>(n_workers);
        work->pairs = std::move(pairs);
        work->boundaries = std::move(boundaries);
        if (stats) {
            work->stats.resize(n_workers);
        }
        std::vector<std::function<void()>> tasks;
        tasks.reserve(n_workers);
        for (int w = 0; w < n_workers; ++w) {
            tasks.emplace_back([this, work, n_ranges, n_workers, w, dst, stats, start] {
                try {
                    const int64_t begin = n_ranges * w / n_workers;
                    const int64_t end = n_ranges * (w + 1) / n_workers;
                    run_range(work->pairs, page_size ? work->boundaries[begin] : begin,
                            page_size ? work->boundaries[end] : end, dst, stats ? &work->stats[w] : nullptr);
                } catch (...) {
                    work->errors[w] = std::current_exception();
                }
                if (work->remaining.fetch_sub(1) == 1) {
                    for (const auto & error : work->errors) {
                        if (error) {
                            work->done.set_exception(error);
                            return;
                        }
                    }
                    if (stats) {
                        for (const auto & part : work->stats) {
                            stats->calls += part.calls;
                            stats->bytes += part.bytes;
                            stats->read_us += part.read_us;
                        }
                        stats->total_us = ggml_time_us() - start;
                    }
                    work->done.set_value();
                }
            });
        }
        if (!pool) {
            pool = std::make_unique<llama_lazy_reader_pool>(n_threads);
        }
        pending = work->done.get_future().share();
        pool->submit(std::move(tasks));
        return pending;
    }

private:
    mutable std::mutex submit_mutex;
    mutable std::unique_ptr<llama_lazy_reader_pool> pool;
    mutable std::shared_future<void> pending;

    int64_t group_end(const std::vector<std::pair<int32_t, int32_t>> & pairs, int64_t begin, int64_t end) const {
        const size_t off = base + (size_t) pairs[begin].first * row_size;
        size_t limit = (off + row_size - 1) / page_size;
        int64_t i = begin + 1;
        for (; i < end; ++i) {
            const size_t next = base + (size_t) pairs[i].first * row_size;
            if (next / page_size > limit || next - off + row_size > std::max<size_t>(65536, row_size)) {
                break;
            }
            limit = (next + row_size - 1) / page_size;
        }
        return i;
    }

    void run_range(const std::vector<std::pair<int32_t, int32_t>> & pairs,
                   int64_t begin, int64_t end, float * dst, llama_lazy_read_stats * stats) const {
        std::vector<uint8_t> bounce(page_size ? std::max<size_t>(65536, row_size) : row_size);
        for (int64_t i = begin; i < end; ) {
            int64_t stop = page_size ? group_end(pairs, i, end) : i + 1;
            while (stop < end && pairs[stop].first == pairs[i].first) {
                ++stop;
            }
            const size_t off = base + (size_t) pairs[i].first * row_size;
            const size_t size = ((size_t) pairs[stop - 1].first - pairs[i].first + 1) * row_size;
            for (size_t done = 0; done < size; ) {
                const int64_t start = stats ? ggml_time_us() : 0;
                const ssize_t n_read = ::pread(fd, bounce.data() + done, size - done, off + done);
                if (stats) {
                    stats->calls++;
                    stats->read_us += ggml_time_us() - start;
                }
                if (n_read < 0 && errno == EINTR) {
                    continue;
                }
                if (n_read <= 0) {
                    throw std::runtime_error(format("lazy direct read of %zu bytes at file offset %zu failed: %s",
                            size, off, n_read == 0 ? "unexpected EOF" : strerror(errno)));
                }
                done += n_read;
                if (stats) {
                    stats->bytes += n_read;
                }
            }
            for (int64_t row = i; row < stop; ) {
                int64_t j = row;
                while (j + 1 < stop && pairs[j + 1].first == pairs[row].first) {
                    ++j;
                }
                const uint8_t * src = bounce.data() + ((size_t) pairs[row].first - pairs[i].first) * row_size;
                float * first = dst + (size_t) pairs[row].second * head_dim;
                if (to_float) {
                    to_float(src, first, head_dim);
                } else {
                    memcpy(first, src, (size_t) head_dim * sizeof(float));
                }
                for (int64_t k = row + 1; k <= j; ++k) {
                    memcpy(dst + (size_t) pairs[k].second * head_dim, first, (size_t) head_dim * sizeof(float));
                }
                row = j + 1;
            }
            i = stop;
        }
    }
#endif
};
