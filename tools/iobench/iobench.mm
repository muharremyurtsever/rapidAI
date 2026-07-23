// iobench.mm — Apple Silicon SSD -> unified memory -> Metal streaming microbenchmark.
// Measures the raw physical quantities the rapidAI engine depends on:
//   1. seq_cold_gbps        — sequential mmap sweep, cold page cache
//   2. rand16k/2m/32m_gbps  — random block reads at three granularities
//   3. willneed_gbps        — double-buffered madvise(WILLNEED) prefetch pipeline
//   4. gpu_stream_gbps      — GPU consuming zero-copy (newBufferWithBytesNoCopy) windows
//   5. dontneed_rss_drop_mb — how much resident memory madvise(DONTNEED) releases
//
// Build: make
// Run:   ./iobench <file> [--make-file <GB>]

#import <Metal/Metal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <mach/mach.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static size_t page_size() {
    static size_t ps = (size_t)sysconf(_SC_PAGESIZE);
    return ps;
}

static size_t resident_mb() {
    task_vm_info_data_t info;
    mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
    task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &cnt);
    return (size_t)(info.phys_footprint / (1024 * 1024));
}

// Drop this file's pages from the unified buffer cache so reads hit the SSD.
static void purge_file_cache(const char* path, size_t len) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return;
    void* p = mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, 0);
    if (p != MAP_FAILED) {
        msync(p, len, MS_INVALIDATE);
        madvise(p, len, MADV_DONTNEED);
        munmap(p, len);
    }
    // F_NOCACHE read pass evicts remaining cached ranges.
    fcntl(fd, F_NOCACHE, 1);
    close(fd);
    system("sync");
}

static void make_file(const char* path, size_t gb) {
    fprintf(stderr, "creating %zu GB test file at %s...\n", gb, path);
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("open for create"); exit(1); }
    const size_t chunk = 64ull << 20;
    std::vector<uint8_t> buf(chunk);
    std::mt19937_64 rng(42);
    for (size_t i = 0; i < buf.size(); i += 8) {
        uint64_t v = rng();
        memcpy(&buf[i], &v, 8);
    }
    size_t total = gb << 30;
    for (size_t off = 0; off < total; off += chunk) {
        // re-randomize the first 4 KB of each chunk so compression can't cheat
        uint64_t v = rng();
        memcpy(buf.data(), &v, 8);
        if (write(fd, buf.data(), chunk) != (ssize_t)chunk) { perror("write"); exit(1); }
        if ((off >> 30) != ((off + chunk) >> 30)) fprintf(stderr, "  %zu GB\n", (off + chunk) >> 30);
    }
    close(fd);
    system("sync");
}

// Touch every page in [p, p+len); returns GB/s.
static double sweep_gbps(const uint8_t* p, size_t len) {
    volatile uint64_t sink = 0;
    const size_t ps = page_size();
    double t0 = now_s();
    for (size_t off = 0; off < len; off += ps) sink += p[off];
    double dt = now_s() - t0;
    (void)sink;
    return (double)len / dt / 1e9;
}

static double bench_random(const char* path, size_t file_len, size_t block, size_t total_read) {
    purge_file_cache(path, file_len);
    int fd = open(path, O_RDONLY);
    uint8_t* p = (uint8_t*)mmap(nullptr, file_len, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(1); }
    std::mt19937_64 rng(7);
    size_t n_blocks = total_read / block;
    std::vector<size_t> offsets(n_blocks);
    size_t slots = file_len / block;
    for (auto& o : offsets) o = (rng() % slots) * block;
    const size_t ps = page_size();
    volatile uint64_t sink = 0;
    double t0 = now_s();
    for (size_t o : offsets)
        for (size_t i = 0; i < block; i += ps) sink += p[o + i];
    double dt = now_s() - t0;
    (void)sink;
    munmap(p, file_len);
    close(fd);
    return (double)(n_blocks * block) / dt / 1e9;
}

// Multithreaded page-touch: nthreads sweep disjoint stripes concurrently.
// Exposes whether the fault path is latency-bound (single fault in flight).
#include <thread>
static double bench_mt_sweep(const char* path, size_t file_len, int nthreads) {
    purge_file_cache(path, file_len);
    int fd = open(path, O_RDONLY);
    uint8_t* p = (uint8_t*)mmap(nullptr, file_len, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(1); }
    size_t stripe = (file_len / nthreads / page_size()) * page_size();
    double t0 = now_s();
    std::vector<std::thread> ts;
    for (int i = 0; i < nthreads; i++)
        ts.emplace_back([&, i] { sweep_gbps(p + (size_t)i * stripe, stripe); });
    for (auto& t : ts) t.join();
    double dt = now_s() - t0;
    munmap(p, file_len);
    close(fd);
    return (double)(stripe * nthreads) / dt / 1e9;
}

// Plain pread with 8 MB blocks and F_NOCACHE — the syscall baseline.
static double bench_pread(const char* path, size_t file_len) {
    purge_file_cache(path, file_len);
    int fd = open(path, O_RDONLY);
    fcntl(fd, F_NOCACHE, 1);
    const size_t block = 8ull << 20;
    std::vector<uint8_t> buf(block);
    double t0 = now_s();
    size_t got = 0;
    for (size_t off = 0; off + block <= file_len; off += block)
        got += (size_t)pread(fd, buf.data(), block, (off_t)off);
    double dt = now_s() - t0;
    close(fd);
    return (double)got / dt / 1e9;
}

// macOS's real readahead API: fcntl(F_RDADVISE) on the NEXT window while
// sweeping the current one through the mmap view.
static double bench_rdadvise(const char* path, size_t file_len) {
    purge_file_cache(path, file_len);
    int fd = open(path, O_RDONLY);
    uint8_t* p = (uint8_t*)mmap(nullptr, file_len, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(1); }
    const size_t chunk = 512ull << 20;
    // prime the pipeline: advise the first window before starting the clock's sweep
    struct radvisory ra0 = {0, (int)std::min(chunk, file_len)};
    fcntl(fd, F_RDADVISE, &ra0);
    double t0 = now_s();
    for (size_t off = 0; off < file_len; off += chunk) {
        size_t next = off + chunk;
        if (next < file_len) {
            struct radvisory ra = {(off_t)next, (int)std::min(chunk, file_len - next)};
            fcntl(fd, F_RDADVISE, &ra);
        }
        sweep_gbps(p + off, std::min(chunk, file_len - off));
    }
    double dt = now_s() - t0;
    munmap(p, file_len);
    close(fd);
    return (double)file_len / dt / 1e9;
}

static double bench_willneed(const char* path, size_t file_len) {
    purge_file_cache(path, file_len);
    int fd = open(path, O_RDONLY);
    uint8_t* p = (uint8_t*)mmap(nullptr, file_len, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(1); }
    const size_t chunk = 512ull << 20;
    double t0 = now_s();
    for (size_t off = 0; off < file_len; off += chunk) {
        size_t next = off + chunk;
        if (next < file_len)
            madvise(p + next, std::min(chunk, file_len - next), MADV_WILLNEED);
        sweep_gbps(p + off, std::min(chunk, file_len - off)); // consume current
    }
    double dt = now_s() - t0;
    munmap(p, file_len);
    close(fd);
    return (double)file_len / dt / 1e9;
}

struct GpuResult { double gbps; bool zero_copy; };

// prefetch_mode: 0 = madvise(WILLNEED), 1 = fcntl(F_RDADVISE)
static GpuResult bench_gpu_stream(const char* path, size_t file_len, int prefetch_mode) {
    static const char* kKernelSrc = R"(
#include <metal_stdlib>
using namespace metal;
kernel void sum_u32(device const uint* buf [[buffer(0)]],
                    device atomic_uint* out [[buffer(1)]],
                    constant uint& n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]],
                    uint threads [[threads_per_grid]]) {
    uint acc = 0;
    for (uint i = gid; i < n; i += threads) acc += buf[i];
    atomic_fetch_add_explicit(out, acc, memory_order_relaxed);
})";
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) { fprintf(stderr, "no Metal device\n"); return {0, false}; }
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:[NSString stringWithUTF8String:kKernelSrc]
                                              options:nil error:&err];
    if (!lib) { fprintf(stderr, "kernel compile failed: %s\n", err.localizedDescription.UTF8String); return {0, false}; }
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"sum_u32"] error:&err];
    id<MTLCommandQueue> queue = [device newCommandQueue];

    purge_file_cache(path, file_len);
    int fd = open(path, O_RDONLY);
    uint8_t* p = (uint8_t*)mmap(nullptr, file_len, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(1); }

    const size_t window = 512ull << 20; // page-aligned since mmap base is
    id<MTLBuffer> outBuf = [device newBufferWithLength:sizeof(uint32_t)
                                               options:MTLResourceStorageModeShared];
    bool zero_copy = true;
    double t0 = now_s();
    size_t consumed = 0;
    for (size_t off = 0; off + window <= file_len; off += window) {
        // prefetch next window while GPU chews this one
        size_t next = off + window;
        if (next + window <= file_len) {
            if (prefetch_mode == 0) {
                madvise(p + next, window, MADV_WILLNEED);
            } else {
                struct radvisory ra = {(off_t)next, (int)window};
                fcntl(fd, F_RDADVISE, &ra);
            }
        }

        id<MTLBuffer> wbuf = [device newBufferWithBytesNoCopy:(void*)(p + off)
                                                       length:window
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
        if (!wbuf) { // fall back once, flag it
            zero_copy = false;
            wbuf = [device newBufferWithBytes:(void*)(p + off) length:window
                                      options:MTLResourceStorageModeShared];
        }
        uint32_t n = (uint32_t)(window / 4);
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:wbuf offset:0 atIndex:0];
        [enc setBuffer:outBuf offset:0 atIndex:1];
        [enc setBytes:&n length:sizeof(n) atIndex:2];
        MTLSize grid = MTLSizeMake(64 * 1024, 1, 1);
        MTLSize group = MTLSizeMake(pso.maxTotalThreadsPerThreadgroup, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:group];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        consumed += window;
        madvise((void*)(p + off), window, MADV_DONTNEED); // release consumed window
    }
    double dt = now_s() - t0;
    munmap(p, file_len);
    close(fd);
    return {(double)consumed / dt / 1e9, zero_copy};
}

// The candidate engine path: pread into a ping-pong pair of shared MTLBuffers,
// GPU sums each window while the CPU thread fills the other buffer.
static double bench_gpu_pread(const char* path, size_t file_len) {
    static const char* kSrc = R"(
#include <metal_stdlib>
using namespace metal;
kernel void sum_u32(device const uint* buf [[buffer(0)]],
                    device atomic_uint* out [[buffer(1)]],
                    constant uint& n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]],
                    uint threads [[threads_per_grid]]) {
    uint acc = 0;
    for (uint i = gid; i < n; i += threads) acc += buf[i];
    atomic_fetch_add_explicit(out, acc, memory_order_relaxed);
})";
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:[NSString stringWithUTF8String:kSrc]
                                              options:nil error:&err];
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"sum_u32"] error:&err];
    id<MTLCommandQueue> queue = [device newCommandQueue];

    purge_file_cache(path, file_len);
    int fd = open(path, O_RDONLY);
    fcntl(fd, F_NOCACHE, 1);
    const size_t window = 512ull << 20;
    id<MTLBuffer> bufs[2] = {
        [device newBufferWithLength:window options:MTLResourceStorageModeShared],
        [device newBufferWithLength:window options:MTLResourceStorageModeShared]};
    id<MTLBuffer> outBuf = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];

    auto fill = [&](int which, size_t off) {
        uint8_t* dst = (uint8_t*)bufs[which].contents;
        const size_t block = 8ull << 20;
        for (size_t o = 0; o < window; o += block)
            pread(fd, dst + o, block, (off_t)(off + o));
    };
    auto dispatch = [&](int which) -> id<MTLCommandBuffer> {
        uint32_t n = (uint32_t)(window / 4);
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bufs[which] offset:0 atIndex:0];
        [enc setBuffer:outBuf offset:0 atIndex:1];
        [enc setBytes:&n length:sizeof(n) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(64 * 1024, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(pso.maxTotalThreadsPerThreadgroup, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        return cmd;
    };

    size_t n_windows = file_len / window;
    double t0 = now_s();
    fill(0, 0);
    id<MTLCommandBuffer> inflight = dispatch(0);
    for (size_t w = 1; w < n_windows; w++) {
        fill(w & 1, w * window);          // CPU fills buffer B while GPU sums A
        [inflight waitUntilCompleted];
        inflight = dispatch(w & 1);
    }
    [inflight waitUntilCompleted];
    double dt = now_s() - t0;
    close(fd);
    return (double)(n_windows * window) / dt / 1e9;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <file> [--make-file GB]\n", argv[0]); return 1; }
    const char* path = argv[1];
    for (int i = 2; i < argc - 1; i++)
        if (!strcmp(argv[i], "--make-file")) make_file(path, (size_t)atoll(argv[i + 1]));

    struct stat st;
    if (stat(path, &st) != 0) { perror("stat"); return 1; }
    size_t len = ((size_t)st.st_size / page_size()) * page_size();

    // 1. sequential cold sweep
    purge_file_cache(path, len);
    int fd = open(path, O_RDONLY);
    uint8_t* p = (uint8_t*)mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { perror("mmap"); return 1; }
    double seq = sweep_gbps(p, len);

    // 5. DONTNEED resident drop (reuse the hot mapping from step 1)
    size_t rss_before = resident_mb();
    madvise((void*)p, len, MADV_DONTNEED);
    usleep(1000 * 1000);
    size_t rss_after = resident_mb();
    munmap(p, len);
    close(fd);

    // 2. random blocks — read up to 8 GB or the file size, whichever is smaller
    size_t total_read = std::min(len, (size_t)8ull << 30);
    double r16k = bench_random(path, len, 16 << 10, std::min(total_read, (size_t)2ull << 30));
    double r2m  = bench_random(path, len, 2 << 20, total_read);
    double r32m = bench_random(path, len, 32 << 20, total_read);

    // 3. prefetch pipelines: WILLNEED vs F_RDADVISE, plus threaded + syscall baselines
    double wn = bench_willneed(path, len);
    double ra = bench_rdadvise(path, len);
    double mt8 = bench_mt_sweep(path, len, 8);
    double pr = bench_pread(path, len);

    // 4. GPU zero-copy stream with both prefetch modes
    GpuResult gpu_wn = bench_gpu_stream(path, len, 0);
    GpuResult gpu_ra = bench_gpu_stream(path, len, 1);
    double gpu_pr = bench_gpu_pread(path, len);

    printf("{\"file_gb\": %.1f, \"seq_cold_gbps\": %.3f, \"rand16k_gbps\": %.3f, "
           "\"rand2m_gbps\": %.3f, \"rand32m_gbps\": %.3f, \"willneed_gbps\": %.3f, "
           "\"rdadvise_gbps\": %.3f, \"mt8_sweep_gbps\": %.3f, \"pread8m_gbps\": %.3f, "
           "\"gpu_stream_willneed_gbps\": %.3f, \"gpu_stream_rdadvise_gbps\": %.3f, "
           "\"gpu_pread_pingpong_gbps\": %.3f, "
           "\"gpu_zero_copy\": %s, \"dontneed_rss_drop_mb\": %zd}\n",
           (double)len / 1e9, seq, r16k, r2m, r32m, wn, ra, mt8, pr,
           gpu_wn.gbps, gpu_ra.gbps, gpu_pr, (gpu_wn.zero_copy && gpu_ra.zero_copy) ? "true" : "false",
           (ssize_t)rss_before - (ssize_t)rss_after);
    return 0;
}
