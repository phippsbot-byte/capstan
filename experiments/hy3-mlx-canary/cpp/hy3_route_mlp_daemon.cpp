#include "hy3_expert_bank.h"
#include "hy3_q4_affine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <unistd.h>

namespace fs = std::filesystem;

constexpr double kMaxDenseCacheGiB = 16.0;
constexpr double kMaxPackedCacheGiB = 16.0;
constexpr double kMaxCombinedCacheGiB = 16.0;
constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;
constexpr size_t kDenseExpertBytes = static_cast<size_t>(3) * static_cast<size_t>(1536) * static_cast<size_t>(4096) * sizeof(float);
constexpr int kHybridDenseRouteThreshold = 8;

enum class Q4ExecutionMode {
  Dense,
  Direct,
  Hybrid,
};

static Q4ExecutionMode parse_q4_mode(const std::string &mode) {
  if (mode == "dense") return Q4ExecutionMode::Dense;
  if (mode == "direct") return Q4ExecutionMode::Direct;
  if (mode == "hybrid") return Q4ExecutionMode::Hybrid;
  throw std::runtime_error("--q4-mode must be dense, direct, or hybrid");
}

static const char *q4_mode_name(Q4ExecutionMode mode) {
  switch (mode) {
    case Q4ExecutionMode::Dense: return "dense";
    case Q4ExecutionMode::Direct: return "direct";
    case Q4ExecutionMode::Hybrid: return "hybrid";
  }
  return "unknown";
}

static bool use_dense_for_expert(Q4ExecutionMode mode, int route_count) {
  if (mode == Q4ExecutionMode::Dense) return true;
  if (mode == Q4ExecutionMode::Direct) return false;
  return route_count >= kHybridDenseRouteThreshold;
}

struct Args {
  fs::path index;
  fs::path root;
  double dense_cache_gib = 0.0;
  double packed_cache_gib = 0.0;
  Q4ExecutionMode q4_mode = Q4ExecutionMode::Dense;
};

struct DenseExpertBank {
  DenseQ4Affine up;
  DenseQ4Affine gate;
  DenseQ4Affine down;
};

#pragma pack(push, 1)
struct RequestHeader {
  char magic[4];
  uint32_t layer;
  uint32_t seq_len;
  uint32_t topk;
  uint32_t flags;
};

struct ResponseHeader {
  char magic[4];
  uint32_t status;
  uint32_t payload_floats;
  uint32_t read_calls;
  uint32_t reserved;
  double compute_s;
  uint64_t bytes_read;
};
#pragma pack(pop)
static_assert(sizeof(RequestHeader) == 20);
static_assert(sizeof(ResponseHeader) == 36);

static Args parse_args(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    auto need = [&](const char *name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
      return argv[++i];
    };
    if (key == "--index") args.index = need("--index");
    else if (key == "--root") args.root = need("--root");
    else if (key == "--dense-cache-gib") args.dense_cache_gib = std::stod(need("--dense-cache-gib"));
    else if (key == "--packed-cache-gib") args.packed_cache_gib = std::stod(need("--packed-cache-gib"));
    else if (key == "--q4-mode") args.q4_mode = parse_q4_mode(need("--q4-mode"));
    else if (key == "--help" || key == "-h") {
      std::fprintf(stderr, "usage: hy3_route_mlp_daemon --index compact-index.tsv --root sidecar-root [--dense-cache-gib N<=16] [--packed-cache-gib N<=16] [--q4-mode dense|direct|hybrid]\n");
      std::exit(0);
    } else {
      throw std::runtime_error("unknown arg: " + key);
    }
  }
  if (args.index.empty()) throw std::runtime_error("--index is required");
  if (args.root.empty()) throw std::runtime_error("--root is required");
  if (!std::isfinite(args.dense_cache_gib) || args.dense_cache_gib < 0.0 || args.dense_cache_gib > kMaxDenseCacheGiB) {
    throw std::runtime_error("--dense-cache-gib must be finite and in [0, 16]");
  }
  if (!std::isfinite(args.packed_cache_gib) || args.packed_cache_gib < 0.0 || args.packed_cache_gib > kMaxPackedCacheGiB) {
    throw std::runtime_error("--packed-cache-gib must be finite and in [0, 16]");
  }
  if (args.dense_cache_gib + args.packed_cache_gib > kMaxCombinedCacheGiB) {
    throw std::runtime_error("combined dense and packed cache budget must be <= 16 GiB");
  }
  return args;
}

static bool read_full(int fd, void *dst, size_t n) {
  auto *p = static_cast<uint8_t *>(dst);
  size_t done = 0;
  while (done < n) {
    ssize_t got = ::read(fd, p + done, n - done);
    if (got == 0) return false;
    if (got < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error(std::string("read failed: ") + std::strerror(errno));
    }
    done += static_cast<size_t>(got);
  }
  return true;
}

static void write_full(int fd, const void *src, size_t n) {
  const auto *p = static_cast<const uint8_t *>(src);
  size_t done = 0;
  while (done < n) {
    ssize_t put = ::write(fd, p + done, n - done);
    if (put < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error(std::string("write failed: ") + std::strerror(errno));
    }
    done += static_cast<size_t>(put);
  }
}

static DenseExpertBank dequantize_expert_bank(const ExpertBank &bank) {
  DenseExpertBank dense;
  qlinear_dequantize(bank.up_w, bank.up_s, bank.up_b, 1536, 4096, 512, 64, dense.up);
  qlinear_dequantize(bank.gate_w, bank.gate_s, bank.gate_b, 1536, 4096, 512, 64, dense.gate);
  qlinear_dequantize(bank.down_w, bank.down_s, bank.down_b, 4096, 1536, 192, 24, dense.down);
  return dense;
}

static size_t dense_bank_bytes(const DenseExpertBank &bank) {
  return (bank.up.weights.size() + bank.gate.weights.size() + bank.down.weights.size()) * sizeof(float);
}

struct DenseCacheEntry {
  std::unique_ptr<DenseExpertBank> bank;
  size_t bytes = 0;
  uint64_t last_use = 0;
};

class DenseExpertCache {
 public:
  explicit DenseExpertCache(size_t max_bytes) : max_bytes_(max_bytes) {}

  const DenseExpertBank &get(
      int layer,
      int expert,
      const std::vector<Entry> &entries,
      const std::unordered_map<uint64_t, Span> &spans,
      FdCache &fds,
      uint32_t &read_calls,
      uint64_t &bytes_read) {
    ++tick_;
    uint64_t key = span_key(layer, expert);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      it->second.last_use = tick_;
      ++hits_;
      return *it->second.bank;
    }
    ++misses_;
    ExpertBank raw = load_expert_bank(entries, spans, fds, layer, expert);
    bytes_read += raw.raw.size();
    ++read_calls;
    auto dense = std::make_unique<DenseExpertBank>(dequantize_expert_bank(raw));
    size_t bytes = dense_bank_bytes(*dense);
    if (max_bytes_ == 0 || bytes > max_bytes_) {
      scratch_ = std::move(dense);
      return *scratch_;
    }
    while (bytes_ + bytes > max_bytes_ && !cache_.empty()) evict_one();
    DenseCacheEntry entry;
    entry.bytes = bytes;
    entry.last_use = tick_;
    entry.bank = std::move(dense);
    auto [inserted, _] = cache_.emplace(key, std::move(entry));
    bytes_ += bytes;
    return *inserted->second.bank;
  }

  bool contains(int layer, int expert) const {
    return cache_.find(span_key(layer, expert)) != cache_.end();
  }

  size_t entries() const { return cache_.size(); }
  uint64_t hits() const { return hits_; }
  uint64_t misses() const { return misses_; }
  uint64_t evictions() const { return evictions_; }
  size_t bytes() const { return bytes_; }
  size_t max_bytes() const { return max_bytes_; }

 private:
  void evict_one() {
    auto victim = cache_.begin();
    for (auto it = cache_.begin(); it != cache_.end(); ++it) {
      if (it->second.last_use < victim->second.last_use) victim = it;
    }
    bytes_ -= victim->second.bytes;
    cache_.erase(victim);
    ++evictions_;
  }

  size_t max_bytes_ = 0;
  size_t bytes_ = 0;
  uint64_t tick_ = 0;
  uint64_t hits_ = 0;
  uint64_t misses_ = 0;
  uint64_t evictions_ = 0;
  std::unique_ptr<DenseExpertBank> scratch_;
  std::unordered_map<uint64_t, DenseCacheEntry> cache_;
};

static void compute_expert_route_dense(
    const DenseExpertBank &bank,
    const float *x,
    size_t route,
    std::vector<float> &route_outputs,
    std::vector<float> &xvec,
    std::vector<float> &up,
    std::vector<float> &gate,
    std::vector<float> &hidden,
    std::vector<float> &down) {
  xvec.assign(x, x + 4096);
  qlinear_dense(xvec, bank.up, up);
  qlinear_dense(xvec, bank.gate, gate);
  hidden.resize(1536);
  for (int i = 0; i < 1536; ++i) hidden[static_cast<size_t>(i)] = silu(gate[static_cast<size_t>(i)]) * up[static_cast<size_t>(i)];
  qlinear_dense(hidden, bank.down, down);
  std::copy(down.begin(), down.end(), route_outputs.begin() + route * 4096);
}

static void compute_expert_route_direct(
    const ExpertBank &bank,
    const float *x,
    size_t route,
    std::vector<float> &route_outputs,
    std::vector<float> &xvec,
    std::vector<float> &hidden,
    std::vector<float> &down,
    std::vector<float> &group_sums) {
  xvec.assign(x, x + 4096);
  qlinear_pair_swiglu_direct(
      xvec,
      bank.up_w,
      bank.up_s,
      bank.up_b,
      bank.gate_w,
      bank.gate_s,
      bank.gate_b,
      1536,
      4096,
      512,
      64,
      hidden,
      group_sums);
  qlinear_direct(hidden, bank.down_w, bank.down_s, bank.down_b, 4096, 1536, 192, 24, down, group_sums);
  std::copy(down.begin(), down.end(), route_outputs.begin() + route * 4096);
}

static uint64_t span_nbytes_for_route(const std::unordered_map<uint64_t, Span> &spans, int layer, int expert) {
  auto it = spans.find(span_key(layer, expert));
  if (it == spans.end()) throw std::runtime_error("missing expert span for daemon request");
  return it->second.nbytes;
}

struct ComputeResult {
  std::vector<float> actual;
  uint32_t read_calls = 0;
  uint32_t packed_cache_hits = 0;
  uint64_t bytes_read = 0;
  double compute_s = 0.0;
};

static ComputeResult compute_request(
    int layer,
    int seq_len,
    int topk,
    const std::vector<float> &hidden_flat,
    const std::vector<int32_t> &experts_flat,
    const std::vector<float> &route_weights_flat,
    const std::vector<Entry> &entries,
    const std::unordered_map<uint64_t, Span> &spans,
    FdCache &fds,
    DenseExpertCache &dense_cache,
    PackedExpertCache &packed_cache,
    Q4ExecutionMode q4_mode) {
  if (layer < 0) throw std::runtime_error("layer must be non-negative");
  if (seq_len < 1 || seq_len > 4096) throw std::runtime_error("seq_len out of supported range");
  if (topk < 1 || topk > 64) throw std::runtime_error("topk out of supported range");
  if (hidden_flat.size() != static_cast<size_t>(seq_len) * 4096) throw std::runtime_error("hidden payload size mismatch");
  if (experts_flat.size() != static_cast<size_t>(seq_len) * static_cast<size_t>(topk)) throw std::runtime_error("experts payload size mismatch");
  if (route_weights_flat.size() != experts_flat.size()) throw std::runtime_error("route_weights payload size mismatch");

  ComputeResult result;
  result.actual.assign(static_cast<size_t>(seq_len) * 4096, 0.0f);
  auto t0 = std::chrono::steady_clock::now();
  const uint64_t packed_hits_before = packed_cache.hits();

  std::vector<int> unique_experts;
  std::unordered_set<int> seen;
  unique_experts.reserve(experts_flat.size());
  for (int32_t expert : experts_flat) {
    if (expert < 0 || expert >= 192) throw std::runtime_error("expert id out of range");
    if (seen.insert(static_cast<int>(expert)).second) unique_experts.push_back(static_cast<int>(expert));
  }

  std::unordered_map<int, int> route_counts;
  route_counts.reserve(unique_experts.size());
  for (int32_t expert : experts_flat) ++route_counts[static_cast<int>(expert)];

  std::vector<float> route_outputs(static_cast<size_t>(seq_len) * static_cast<size_t>(topk) * 4096, 0.0f);
  std::vector<float> xvec, up, gate, swiglu_hidden, down, group_sums;
  for (int expert : unique_experts) {
    const bool use_dense =
        q4_mode == Q4ExecutionMode::Hybrid && dense_cache.contains(layer, expert)
            ? true
            : use_dense_for_expert(q4_mode, route_counts[expert]);
    ExpertBank loaded_bank;
    const ExpertBank *raw_bank_ptr = nullptr;
    const DenseExpertBank *dense_bank = nullptr;
    if (use_dense) {
      dense_bank = &dense_cache.get(layer, expert, entries, spans, fds, result.read_calls, result.bytes_read);
    } else {
      raw_bank_ptr = packed_cache.find(layer, expert);
      if (raw_bank_ptr == nullptr) {
        loaded_bank = load_expert_bank(entries, spans, fds, layer, expert);
        result.bytes_read += loaded_bank.raw.size();
        ++result.read_calls;
        raw_bank_ptr = packed_cache.insert(std::move(loaded_bank));
        if (raw_bank_ptr == nullptr) raw_bank_ptr = &loaded_bank;
      }
    }
    for (int token = 0; token < seq_len; ++token) {
      const float *x = hidden_flat.data() + static_cast<size_t>(token) * 4096;
      for (int k = 0; k < topk; ++k) {
        size_t route = static_cast<size_t>(token) * static_cast<size_t>(topk) + static_cast<size_t>(k);
        if (experts_flat[route] != expert) continue;
        (void)span_nbytes_for_route(spans, layer, expert);
        if (use_dense) {
          compute_expert_route_dense(*dense_bank, x, route, route_outputs, xvec, up, gate, swiglu_hidden, down);
        } else {
          compute_expert_route_direct(*raw_bank_ptr, x, route, route_outputs, xvec, swiglu_hidden, down, group_sums);
        }
      }
    }
  }

  for (int token = 0; token < seq_len; ++token) {
    size_t token_base = static_cast<size_t>(token) * 4096;
    for (int k = 0; k < topk; ++k) {
      size_t route = static_cast<size_t>(token) * static_cast<size_t>(topk) + static_cast<size_t>(k);
      float weight = route_weights_flat[route];
      size_t route_base = route * 4096;
      for (int i = 0; i < 4096; ++i) result.actual[token_base + static_cast<size_t>(i)] += weight * route_outputs[route_base + static_cast<size_t>(i)];
    }
  }

  auto t1 = std::chrono::steady_clock::now();
  result.compute_s = std::chrono::duration<double>(t1 - t0).count();
  result.packed_cache_hits = static_cast<uint32_t>(packed_cache.hits() - packed_hits_before);
  return result;
}

static void send_error(const std::string &message) {
  ResponseHeader hdr{};
  std::memcpy(hdr.magic, "HY3E", 4);
  hdr.status = 1;
  hdr.payload_floats = static_cast<uint32_t>(message.size());
  write_full(STDOUT_FILENO, &hdr, sizeof(hdr));
  if (!message.empty()) write_full(STDOUT_FILENO, message.data(), message.size());
}

int main(int argc, char **argv) {
  try {
    Args args = parse_args(argc, argv);
    std::vector<Entry> entries = load_entries(args.index);
    auto spans = build_spans(entries);
    FdCache fds(args.root);
    size_t dense_cache_bytes = static_cast<size_t>(args.dense_cache_gib * kBytesPerGiB);
    size_t packed_cache_bytes = static_cast<size_t>(args.packed_cache_gib * kBytesPerGiB);
    DenseExpertCache dense_cache(dense_cache_bytes);
    PackedExpertCache packed_cache(packed_cache_bytes);
    std::fprintf(
        stderr,
        "hy3_route_mlp_daemon ready entries=%zu spans=%zu dense_cache_gib=%.3f packed_cache_gib=%.3f dense_expert_gib=%.6f q4_mode=%s\n",
        entries.size(),
        spans.size(),
        args.dense_cache_gib,
        args.packed_cache_gib,
        static_cast<double>(kDenseExpertBytes) / kBytesPerGiB,
        q4_mode_name(args.q4_mode));
    std::fflush(stderr);

    while (true) {
      RequestHeader req{};
      if (!read_full(STDIN_FILENO, &req, sizeof(req))) break;
      try {
        if (std::memcmp(req.magic, "HY3R", 4) != 0) throw std::runtime_error("bad request magic");
        if (req.layer > 1000 || req.seq_len == 0 || req.seq_len > 4096 || req.topk == 0 || req.topk > 64) throw std::runtime_error("bad request dimensions");
        size_t hidden_count = static_cast<size_t>(req.seq_len) * 4096;
        size_t route_count = static_cast<size_t>(req.seq_len) * static_cast<size_t>(req.topk);
        std::vector<float> hidden(hidden_count);
        std::vector<int32_t> experts(route_count);
        std::vector<float> weights(route_count);
        if (!read_full(STDIN_FILENO, hidden.data(), hidden.size() * sizeof(float))) throw std::runtime_error("short hidden payload");
        if (!read_full(STDIN_FILENO, experts.data(), experts.size() * sizeof(int32_t))) throw std::runtime_error("short expert payload");
        if (!read_full(STDIN_FILENO, weights.data(), weights.size() * sizeof(float))) throw std::runtime_error("short route weight payload");
        ComputeResult result = compute_request(static_cast<int>(req.layer), static_cast<int>(req.seq_len), static_cast<int>(req.topk), hidden, experts, weights, entries, spans, fds, dense_cache, packed_cache, args.q4_mode);
        ResponseHeader hdr{};
        std::memcpy(hdr.magic, "HY3O", 4);
        hdr.status = 0;
        hdr.payload_floats = static_cast<uint32_t>(result.actual.size());
        hdr.read_calls = result.read_calls;
        hdr.reserved = result.packed_cache_hits;
        hdr.compute_s = result.compute_s;
        hdr.bytes_read = result.bytes_read;
        write_full(STDOUT_FILENO, &hdr, sizeof(hdr));
        write_full(STDOUT_FILENO, result.actual.data(), result.actual.size() * sizeof(float));
        std::fflush(stdout);
      } catch (const std::exception &e) {
        send_error(e.what());
        std::fflush(stdout);
        return 1;
      }
    }
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "hy3_route_mlp_daemon: %s\n", e.what());
    return 2;
  }
}
