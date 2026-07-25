// rapidAI Phase 1b native port: lazy expert-bank fetch as an MLX primitive.
//
// Motivation (docs/experiments/phase1b-native-port-report.md): the Python
// streamed path must materialize the routing indices at graph-BUILD time
// (np.array(indices)) to fetch experts from disk, splitting every token into
// per-layer partial evals. Measured breakdown: 0.03 ms Python work vs 0.63 ms
// eval per call; a persistent bank + precomputed slot indices runs the same
// 144 gather_qmm calls at 0.044 ms/call in ONE graph.
//
// Design (single-stream, mid-buffer signal/wait): BankFetch is a primitive
// that runs on the SAME stream as the model. At eval_gpu (encode) time it:
//   1. registers an MTLSharedEventListener notification at value v_gpu whose
//      handler (on the listener's dispatch queue) reads the now materialized
//      routing indices, runs an O(1) slot LRU, preads missing expert rows
//      straight into a persistent unified-memory bank buffer (zero staging
//      copies), writes the slot indices, and sets the event to v_cpu;
//   2. encodes, in the CURRENT command buffer, encodeSignalEvent(v_gpu) —
//      which fires mid-buffer right after the producer of `indices` — and
//      encodeWait(v_cpu), which gates every later op (the gather_qmm that
//      consumes the bank).
// The whole token stays one lazy graph with no Python-side synchronization,
// no command-buffer commits of our own, and NO MLX cross-stream machinery.
// The dependency chain telescopes inside a single queue, so it cannot cycle
// regardless of how MLX batches command buffers.
//
// Why not a CPU-stream primitive: a first implementation used the CPU stream
// plus MLX's cross-stream fences. It worked under per-token blocking evals
// but deadlocked (GPU watchdog "Command buffer execution failed: GPU
// Timeout") on the abandoned in-flight token that mlx_lm's async pipelining
// leaves behind: the CPU<->GPU fence ping-pong of a fully async graph stalls
// inside large command buffers. A second attempt (commit-split with a
// completion handler) hit MLX's gpu::eval, which adds a completion handler
// to the command buffer it captured BEFORE eval_gpu ran — committing inside
// eval_gpu trips "Completed handler provided after commit call".

#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/metal/device.h"
#include "mlx/mlx.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;
namespace nb = nanobind;

namespace {

mx::Dtype dtype_from_string(const std::string& s) {
  if (s == "U32") return mx::uint32;
  if (s == "BF16") return mx::bfloat16;
  if (s == "F16") return mx::float16;
  if (s == "F32") return mx::float32;
  if (s == "U8") return mx::uint8;
  if (s == "I8") return mx::int8;
  throw std::invalid_argument("unsupported safetensors dtype: " + s);
}

ssize_t pread_full(int fd, void* dst, size_t nbytes, int64_t offset) {
  char* p = static_cast<char*>(dst);
  size_t done = 0;
  while (done < nbytes) {
    ssize_t got = ::pread(fd, p + done, nbytes - done, offset + done);
    if (got <= 0) return got;
    done += static_cast<size_t>(got);
  }
  return static_cast<ssize_t>(done);
}

// (per-expert file paths, per-expert absolute offsets, row_nbytes,
//  row_shape, dtype) — paths may repeat (stacked layout: all identical);
// per-expert exports may shard experts across files.
using PartSpec = std::tuple<
    std::vector<std::string>,
    std::vector<int64_t>,
    int64_t,
    std::vector<int>,
    std::string>;

struct BankPart {
  std::vector<int> fds;  // per expert
  std::vector<int64_t> offsets;
  int64_t row_nbytes{0};
  mx::Shape row_shape;
  mx::Dtype dtype{mx::uint32};
  mx::allocator::Buffer buffer{nullptr};
  size_t bank_nbytes{0};
};

class NativeExpertStore {
 public:
  NativeExpertStore(int n_experts, int capacity_slots,
                    const std::vector<PartSpec>& specs)
      : n_experts_(n_experts), capacity_(capacity_slots) {
    if (capacity_slots < 1) {
      throw std::invalid_argument("capacity_slots must be >= 1");
    }
    for (const auto& spec : specs) {
      BankPart p;
      const auto& paths = std::get<0>(spec);
      p.offsets = std::get<1>(spec);
      p.row_nbytes = std::get<2>(spec);
      for (int d : std::get<3>(spec)) p.row_shape.push_back(d);
      p.dtype = dtype_from_string(std::get<4>(spec));
      if (static_cast<int>(p.offsets.size()) != n_experts ||
          static_cast<int>(paths.size()) != n_experts) {
        throw std::invalid_argument("paths/offsets size != n_experts");
      }
      p.fds.reserve(n_experts);
      for (const auto& path : paths) {
        auto it = fds_.find(path);
        if (it == fds_.end()) {
          int fd = ::open(path.c_str(), O_RDONLY);
          if (fd < 0) throw std::runtime_error("cannot open " + path);
          it = fds_.emplace(path, fd).first;
        }
        p.fds.push_back(it->second);
      }
      p.bank_nbytes = static_cast<size_t>(capacity_slots) * p.row_nbytes;
      p.buffer = mx::allocator::malloc(p.bank_nbytes);
      parts_.push_back(std::move(p));
    }
    free_.reserve(capacity_slots);
    for (int s = capacity_slots - 1; s >= 0; --s) free_.push_back(s);
    if (mx::metal::is_available()) {
      auto* dev = mx::metal::device(mx::Device::gpu).mtl_device();
      shared_event_ = NS::TransferPtr(dev->newSharedEvent());
    }
  }

  ~NativeExpertStore() {
    for (auto& p : parts_) mx::allocator::free(p.buffer);
    for (auto& [_, fd] : fds_) ::close(fd);
  }

  NativeExpertStore(const NativeExpertStore&) = delete;
  NativeExpertStore& operator=(const NativeExpertStore&) = delete;

  // Runs on the Metal completion thread (GPU path) or the CPU stream thread
  // (CPU path). Calls on one store are serialized by the event chain, but a
  // mutex keeps this safe under any scheduling.
  void fetch_into_slots(const uint32_t* idx, size_t n, uint32_t* slot_out) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < n; ++i) {
      uint32_t e = idx[i];
      auto seen = call_cache_.find(e);
      if (seen != call_cache_.end()) {
        slot_out[i] = seen->second;
        continue;
      }
      uint32_t slot = lookup_or_load(e);
      call_cache_.emplace(e, slot);
      slot_out[i] = slot;
    }
    call_cache_.clear();
  }

  MTL::SharedEvent* shared_event() { return shared_event_.get(); }
  uint64_t next_event_value() {
    return event_counter_.fetch_add(1, std::memory_order_relaxed) + 1;
  }

  static MTL::SharedEventListener* listener() {
    // One serial listener queue for all stores: fetches are naturally
    // serialized by graph order during decode.
    static MTL::SharedEventListener* l =
        MTL::SharedEventListener::alloc()->init();
    return l;
  }

  int n_experts() const { return n_experts_; }
  int capacity() const { return capacity_; }
  size_t n_parts() const { return parts_.size(); }
  int64_t hits() const { return hits_.load(std::memory_order_relaxed); }
  int64_t misses() const { return misses_.load(std::memory_order_relaxed); }
  int64_t evictions() const {
    return evictions_.load(std::memory_order_relaxed);
  }
  int64_t bytes_read() const {
    return bytes_read_.load(std::memory_order_relaxed);
  }

  std::vector<BankPart> parts_;

 private:
  uint32_t lookup_or_load(uint32_t e) {
    auto it = pos_.find(e);
    if (it != pos_.end()) {
      lru_.splice(lru_.begin(), lru_, it->second.second);
      hits_.fetch_add(1, std::memory_order_relaxed);
      return it->second.first;
    }
    misses_.fetch_add(1, std::memory_order_relaxed);
    uint32_t slot;
    if (!free_.empty()) {
      slot = free_.back();
      free_.pop_back();
    } else {
      uint32_t victim = lru_.back();
      lru_.pop_back();
      slot = pos_[victim].first;
      pos_.erase(victim);
      evictions_.fetch_add(1, std::memory_order_relaxed);
    }
    lru_.push_front(e);
    pos_[e] = {slot, lru_.begin()};
    for (auto& p : parts_) {
      char* dst = static_cast<char*>(p.buffer.raw_ptr()) +
          static_cast<int64_t>(slot) * p.row_nbytes;
      if (pread_full(p.fds[e], dst, p.row_nbytes, p.offsets[e]) !=
          p.row_nbytes) {
        throw std::runtime_error("short read fetching expert " +
                                 std::to_string(e));
      }
      bytes_read_.fetch_add(p.row_nbytes, std::memory_order_relaxed);
    }
    return slot;
  }

  int n_experts_;
  int capacity_;
  std::unordered_map<std::string, int> fds_;
  // expert -> (slot, position in lru_); lru_ holds expert ids, MRU first.
  std::unordered_map<uint32_t, std::pair<uint32_t, std::list<uint32_t>::iterator>>
      pos_;
  std::list<uint32_t> lru_;
  std::vector<uint32_t> free_;
  std::unordered_map<uint32_t, uint32_t> call_cache_;
  std::mutex mutex_;
  NS::SharedPtr<MTL::SharedEvent> shared_event_;
  std::atomic<uint64_t> event_counter_{0};
  std::atomic<int64_t> hits_{0}, misses_{0}, evictions_{0}, bytes_read_{0};
};

// Python-facing handle. The store itself is owned by a plain C++ shared_ptr
// so that copies captured in Metal listener handlers can be destroyed WITHOUT
// the GIL: mx.synchronize holds the GIL while blocking, so any GIL-acquiring
// destructor on the listener queue would deadlock the whole pipeline (this
// exact deadlock was observed as a GPU watchdog timeout).
struct StoreHandle {
  std::shared_ptr<NativeExpertStore> ptr;
};

class BankFetch : public mx::Primitive {
 public:
  BankFetch(mx::Stream stream, std::shared_ptr<NativeExpertStore> store)
      : mx::Primitive(stream), store_(std::move(store)) {}

  void check_indices(const mx::array& idx) {
    if (idx.dtype() != mx::uint32) {
      throw std::invalid_argument("BankFetch indices must be uint32");
    }
    if (!idx.flags().row_contiguous) {
      throw std::invalid_argument("BankFetch indices must be contiguous");
    }
  }

  void set_bank_outputs(std::vector<mx::array>& outputs) {
    for (size_t i = 0; i < store_->parts_.size(); ++i) {
      // Zero-copy view of the persistent bank buffer; no-op deleter keeps
      // the buffer alive across calls (freed by the store's destructor).
      outputs[i + 1].set_data(store_->parts_[i].buffer,
                              [](mx::allocator::Buffer) {});
    }
  }

  void eval_gpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override {
    const auto& idx = inputs[0];
    check_indices(idx);
    outputs[0].set_data(mx::allocator::malloc(outputs[0].nbytes()));
    set_bank_outputs(outputs);
    auto& enc = mx::metal::get_command_encoder(stream());
    const uint32_t* in_ptr = idx.data<uint32_t>();
    uint32_t* out_ptr = outputs[0].data<uint32_t>();
    size_t n = idx.size();
    auto* ev = store_->shared_event();
    uint64_t v_gpu = store_->next_event_value();
    uint64_t v_cpu = store_->next_event_value();
    static const bool dbg = ::getenv("RAPIDAI_BANK_DEBUG") != nullptr;
    if (dbg) {
      struct timeval tv; gettimeofday(&tv, nullptr);
      fprintf(stderr, "[%ld.%06d] [bankfetch %p] encode v_gpu=%llu v_cpu=%llu\n",
              tv.tv_sec % 1000, tv.tv_usec, (void*)store_.get(),
              (unsigned long long)v_gpu, (unsigned long long)v_cpu);
      fflush(stderr);
    }
    // Register the fetch BEFORE the GPU can reach the signal. The handler
    // owns array copies so the buffers cannot be recycled before it runs.
    ev->notifyListener(
        NativeExpertStore::listener(), v_gpu,
        MTL::SharedEventNotificationFunction(
            [store = store_, idx_keep = idx, out_keep = outputs[0], in_ptr,
             out_ptr, n, v_cpu](MTL::SharedEvent* e, uint64_t) {
              static const bool dbg2 =
                  ::getenv("RAPIDAI_BANK_DEBUG") != nullptr;
              if (dbg2) {
                struct timeval tv; gettimeofday(&tv, nullptr);
                fprintf(stderr, "[%ld.%06d] [bankfetch %p] fire -> v_cpu=%llu\n",
                        tv.tv_sec % 1000, tv.tv_usec, (void*)store.get(),
                        (unsigned long long)v_cpu);
                fflush(stderr);
              }
              store->fetch_into_slots(in_ptr, n, out_ptr);
              e->setSignaledValue(v_cpu);
            }));
    // Mid-buffer: signal fires right after the producer of `idx`; the wait
    // gates everything encoded afterwards (the gather_qmm on the bank).
    enc.end_encoding();
    auto* buf = enc.get_command_buffer();
    buf->encodeSignalEvent(ev, v_gpu);
    buf->encodeWait(ev, v_cpu);
  }

  // CPU fallback (no Metal): same math via the CPU stream encoder. Safe here
  // because with no GPU there is a single stream and no cross-stream waits.
  void eval_cpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override {
    const auto& idx = inputs[0];
    check_indices(idx);
    auto& enc = mx::cpu::get_command_encoder(stream());
    enc.set_input_array(idx);
    outputs[0].set_data(mx::allocator::malloc(outputs[0].nbytes()));
    set_bank_outputs(outputs);
    for (auto& o : outputs) enc.set_output_array(o);
    const uint32_t* in_ptr = idx.data<uint32_t>();
    uint32_t* out_ptr = outputs[0].data<uint32_t>();
    size_t n = idx.size();
    enc.dispatch([store = store_, in_ptr, out_ptr, n]() {
      store->fetch_into_slots(in_ptr, n, out_ptr);
    });
    enc.add_temporary(idx);  // keep indices alive until the task runs
  }

  const char* name() const override { return "BankFetch"; }

 private:
  std::shared_ptr<NativeExpertStore> store_;
};

std::vector<mx::array> bank_fetch(const mx::array& indices,
                                  const StoreHandle& handle) {
  const auto& store = handle.ptr;
  // Same stream as the surrounding model ops (mlx_lm sets it as default).
  auto s = mx::default_stream(mx::default_device());
  std::vector<mx::Shape> shapes;
  std::vector<mx::Dtype> dtypes;
  shapes.push_back(indices.shape());
  dtypes.push_back(mx::uint32);
  for (const auto& p : store->parts_) {
    mx::Shape shape;
    shape.push_back(store->capacity());
    for (auto d : p.row_shape) shape.push_back(d);
    shapes.push_back(std::move(shape));
    dtypes.push_back(p.dtype);
  }
  return mx::array::make_arrays(
      std::move(shapes), dtypes, std::make_shared<BankFetch>(s, store),
      {indices});
}

}  // namespace

NB_MODULE(_rapidai_bank, m) {
  m.doc() = "rapidAI native expert-bank fetch (lazy, eval-time disk I/O)";
  nb::class_<StoreHandle>(m, "NativeExpertStore")
      .def(
          "__init__",
          [](StoreHandle* self, int n_experts, int capacity_slots,
             const std::vector<PartSpec>& parts) {
            new (self) StoreHandle{std::make_shared<NativeExpertStore>(
                n_experts, capacity_slots, parts)};
          },
          nb::arg("n_experts"), nb::arg("capacity_slots"), nb::arg("parts"))
      .def_prop_ro("n_experts",
                   [](const StoreHandle& h) { return h.ptr->n_experts(); })
      .def_prop_ro("capacity",
                   [](const StoreHandle& h) { return h.ptr->capacity(); })
      .def_prop_ro("n_parts",
                   [](const StoreHandle& h) { return h.ptr->n_parts(); })
      .def_prop_ro("hits", [](const StoreHandle& h) { return h.ptr->hits(); })
      .def_prop_ro("misses",
                   [](const StoreHandle& h) { return h.ptr->misses(); })
      .def_prop_ro("evictions",
                   [](const StoreHandle& h) { return h.ptr->evictions(); })
      .def_prop_ro("bytes_read",
                   [](const StoreHandle& h) { return h.ptr->bytes_read(); });
  m.def("bank_fetch", &bank_fetch, nb::arg("indices"), nb::arg("store"),
        "Lazy fetch: returns [slot_indices, *bank_parts]; disk I/O happens "
        "in a Metal listener handler at graph evaluation time.");
}
