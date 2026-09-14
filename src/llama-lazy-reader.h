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

struct llama_lazy_reader {
#ifdef _WIN32
    // pread()/open() are unavailable on Windows; --lazy-mode on-direct falls
    // back to the lazy mmap reads there (see llama_model_base::load_lazy_reader)
    const int64_t head_dim = 0;

    void gather(const int32_t *, int64_t, float *) const {
        GGML_ABORT("lazy direct reads are not supported on this platform");
    }
    std::shared_future<void> gather_async(const int32_t *, int64_t, float *) const {
        GGML_ABORT("lazy direct reads are not supported on this platform");
    }
#else
    llama_lazy_reader(int fd, size_t base, size_t row_size, int64_t n_rows, int n_threads,
                      enum ggml_type type, int64_t head_dim)
        : fd(fd), base(base), row_size(row_size), n_rows(n_rows), n_threads(n_threads),
          head_dim(head_dim), to_float(type == GGML_TYPE_F32 ? nullptr : ggml_get_type_traits(type)->to_float) {
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

    // fill dst with the n gathered rows, dequantized to F32:
    // dst[slot * head_dim, ...) = to_float(table[rows[slot]])
    // thread-safe; never lets an exception escape a worker thread
    void gather(const int32_t * rows, int64_t n, float * dst) const {
        gather_async(rows, n, dst).get();
    }

    // One batch per reader bounds queued work. The caller keeps dst alive until completion.
    std::shared_future<void> gather_async(const int32_t * rows, int64_t n, float * dst) const {
        std::unique_lock<std::mutex> lock(submit_mutex);
        if (pending.valid()) {
            pending.wait();
        }
        std::vector<std::pair<int32_t, int32_t>> pairs; // (row, dst slot)
        pairs.reserve(n);
        for (int64_t i = 0; i < n; ++i) {
            GGML_ASSERT(rows[i] >= 0 && (int64_t) rows[i] < n_rows);
            pairs.emplace_back(rows[i], (int32_t) i);
        }

        std::sort(pairs.begin(), pairs.end()); // equal rows adjacent, file order

        const int n_workers = (int) std::min<int64_t>(n_threads, std::max<int64_t>(1, n / 32));
        if (n_workers == 1) {
            run_range(pairs, 0, n, dst);
            std::promise<void> done;
            done.set_value();
            return done.get_future().share();
        }

        struct request {
            std::vector<std::pair<int32_t, int32_t>> pairs;
            std::vector<std::exception_ptr> errors;
            std::atomic<int> remaining;
            std::promise<void> done;
            explicit request(int count) : errors(count), remaining(count) {}
        };
        auto work = std::make_shared<request>(n_workers);
        work->pairs = std::move(pairs);
        std::vector<std::function<void()>> tasks;
        tasks.reserve(n_workers);
        for (int w = 0; w < n_workers; ++w) {
            tasks.emplace_back([this, work, n, n_workers, w, dst] {
                try {
                    run_range(work->pairs, n * w / n_workers, n * (w + 1) / n_workers, dst);
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

    void run_range(const std::vector<std::pair<int32_t, int32_t>> & pairs,
                   int64_t begin, int64_t end, float * dst) const {
        std::vector<uint8_t> bounce(row_size);
        for (int64_t i = begin; i < end; ) {
            int64_t j = i;
            while (j + 1 < end && pairs[j + 1].first == pairs[i].first) {
                ++j; // dedup: one read serves the whole run
            }
            const size_t off = base + (size_t) pairs[i].first * row_size;
            for (size_t done = 0; done < row_size; ) {
                const ssize_t n_read = ::pread(fd, bounce.data() + done, row_size - done, off + done);
                if (n_read < 0 && errno == EINTR) {
                    continue; // interrupted by a signal without SA_RESTART
                }
                if (n_read <= 0) {
                    throw std::runtime_error(format("lazy direct read of %zu bytes at file offset %zu failed: %s",
                            row_size, off, n_read == 0 ? "unexpected EOF" : strerror(errno)));
                }
                done += n_read;
            }
            float * first = dst + (size_t) pairs[i].second * head_dim;
            if (to_float) {
                to_float(bounce.data(), first, head_dim);
            } else {
                memcpy(first, bounce.data(), (size_t) head_dim * sizeof(float));
            }
            for (int64_t k = i + 1; k <= j; ++k) {
                memcpy(dst + (size_t) pairs[k].second * head_dim, first, (size_t) head_dim * sizeof(float));
            }
            i = j + 1;
        }
    }
#endif
};
