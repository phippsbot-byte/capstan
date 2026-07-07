#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <unistd.h>

namespace fs = std::filesystem;

struct Entry {
  int layer = 0;
  int expert = 0;
  std::string family;
  std::string kind;
  uint64_t offset = 0;
  uint64_t nbytes = 0;
  std::string file;
};

struct Span {
  int layer = 0;
  int expert = 0;
  std::string file;
  uint64_t offset = 0;
  uint64_t nbytes = 0;
  int tensors = 0;
};

struct Args {
  fs::path index;
  fs::path root;
  fs::path trace;
  fs::path fixture;
  fs::path fixture_list;
  std::optional<int> layer;
  std::vector<int> explicit_layers;
  std::vector<int> experts;
  int topk = 0;
  int iterations = 1;
  int simulate_tokens = 0;
  int slot_bank = 0;
  bool no_read = false;
  std::string checksum = "sample";
  std::string policy = "freq";
  std::string route_pattern = "rolling";
};

static std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> out;
  std::string item;
  std::stringstream ss(s);
  while (std::getline(ss, item, delim)) out.push_back(item);
  return out;
}

static std::vector<int> parse_csv_ints(const std::string &s) {
  std::vector<int> out;
  for (const auto &part : split(s, ',')) {
    if (!part.empty()) out.push_back(std::stoi(part));
  }
  return out;
}

static std::vector<int> parse_layers(const std::string &s) {
  auto dash = s.find('-');
  if (dash == std::string::npos) return parse_csv_ints(s);
  int start = std::stoi(s.substr(0, dash));
  int end = std::stoi(s.substr(dash + 1));
  if (end < start) throw std::runtime_error("invalid --layers range");
  std::vector<int> out;
  for (int v = start; v <= end; ++v) out.push_back(v);
  return out;
}

static void usage() {
  std::cerr << "usage: hy3_sidecar_io --index compact-index.tsv --root sidecar-root "
               "(--fixture parity.json | --fixture-list fixtures.txt | --trace route.tsv | --layer N --experts a,b,c | --layers A-B --topk K) "
               "[--simulate-tokens N --slot-bank N --policy lru|freq --route-pattern fixed|hot|rolling] "
               "[--iterations N] [--checksum sample|full|none] [--no-read]\n";
}

static Args parse_args(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    auto need_value = [&](const char *name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
      return argv[++i];
    };
    if (key == "--index") args.index = need_value("--index");
    else if (key == "--root") args.root = need_value("--root");
    else if (key == "--trace") args.trace = need_value("--trace");
    else if (key == "--fixture") args.fixture = need_value("--fixture");
    else if (key == "--fixture-list") args.fixture_list = need_value("--fixture-list");
    else if (key == "--layer") args.layer = std::stoi(need_value("--layer"));
    else if (key == "--layers") args.explicit_layers = parse_layers(need_value("--layers"));
    else if (key == "--experts") args.experts = parse_csv_ints(need_value("--experts"));
    else if (key == "--topk") args.topk = std::stoi(need_value("--topk"));
    else if (key == "--iterations") args.iterations = std::stoi(need_value("--iterations"));
    else if (key == "--simulate-tokens") args.simulate_tokens = std::stoi(need_value("--simulate-tokens"));
    else if (key == "--slot-bank") args.slot_bank = std::stoi(need_value("--slot-bank"));
    else if (key == "--policy") args.policy = need_value("--policy");
    else if (key == "--route-pattern") args.route_pattern = need_value("--route-pattern");
    else if (key == "--checksum") args.checksum = need_value("--checksum");
    else if (key == "--no-read") args.no_read = true;
    else if (key == "--help" || key == "-h") { usage(); std::exit(0); }
    else throw std::runtime_error("unknown arg: " + key);
  }
  if (args.index.empty()) throw std::runtime_error("--index is required");
  if (args.root.empty()) throw std::runtime_error("--root is required");
  if (args.iterations < 1) throw std::runtime_error("--iterations must be >= 1");
  if (!(args.checksum == "sample" || args.checksum == "full" || args.checksum == "none")) {
    throw std::runtime_error("--checksum must be sample, full, or none");
  }
  if (!(args.policy == "lru" || args.policy == "freq")) {
    throw std::runtime_error("--policy must be lru or freq");
  }
  if (!(args.route_pattern == "fixed" || args.route_pattern == "hot" || args.route_pattern == "rolling")) {
    throw std::runtime_error("--route-pattern must be fixed, hot, or rolling");
  }
  if (args.simulate_tokens < 0 || args.slot_bank < 0) {
    throw std::runtime_error("--simulate-tokens and --slot-bank must be non-negative");
  }
  if (args.simulate_tokens > 0 && args.slot_bank < 1) {
    throw std::runtime_error("--simulate-tokens requires --slot-bank >= 1");
  }
  if (!args.fixture.empty() || !args.fixture_list.empty()) {
    return args;
  }
  if (!args.trace.empty() && args.slot_bank < 1) {
    throw std::runtime_error("--trace requires --slot-bank >= 1");
  }
  if (!args.trace.empty()) {
    return args;
  }
  if (args.layer.has_value()) {
    if (args.experts.empty()) throw std::runtime_error("--layer requires --experts");
  } else {
    if (args.explicit_layers.empty() || args.topk < 1) throw std::runtime_error("use --layers and --topk, or --layer and --experts");
    args.experts.clear();
    for (int e = 0; e < args.topk; ++e) args.experts.push_back(e);
  }
  return args;
}

static std::vector<Entry> load_entries(const fs::path &index_path) {
  FILE *f = std::fopen(index_path.c_str(), "r");
  if (!f) throw std::runtime_error("failed to open index: " + index_path.string() + ": " + std::strerror(errno));
  char *line = nullptr;
  size_t cap = 0;
  std::vector<Entry> entries;
  while (getline(&line, &cap, f) != -1) {
    std::string s(line);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    if (s.empty() || s[0] == '#') continue;
    if (s.rfind("layer\t", 0) == 0) continue;
    auto cols = split(s, '\t');
    if (cols.size() != 7) {
      std::free(line);
      std::fclose(f);
      throw std::runtime_error("bad compact-index row with " + std::to_string(cols.size()) + " cols");
    }
    Entry e;
    e.layer = std::stoi(cols[0]);
    e.expert = std::stoi(cols[1]);
    e.family = cols[2];
    e.kind = cols[3];
    e.offset = std::stoull(cols[4]);
    e.nbytes = std::stoull(cols[5]);
    e.file = cols[6];
    entries.push_back(std::move(e));
  }
  std::free(line);
  std::fclose(f);
  if (entries.empty()) throw std::runtime_error("compact index has no entries");
  return entries;
}

static uint64_t span_key(int layer, int expert) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32) |
         static_cast<uint32_t>(expert);
}

static std::unordered_map<uint64_t, Span> build_spans(const std::vector<Entry> &entries) {
  std::unordered_map<uint64_t, Span> spans;
  for (const auto &e : entries) {
    uint64_t key = span_key(e.layer, e.expert);
    auto it = spans.find(key);
    if (it == spans.end()) {
      Span s;
      s.layer = e.layer;
      s.expert = e.expert;
      s.file = e.file;
      s.offset = e.offset;
      s.nbytes = e.nbytes;
      s.tensors = 1;
      spans.emplace(key, std::move(s));
    } else {
      auto &s = it->second;
      if (s.file != e.file) throw std::runtime_error("expert span crosses files");
      uint64_t start = std::min(s.offset, e.offset);
      uint64_t end = std::max(s.offset + s.nbytes, e.offset + e.nbytes);
      s.offset = start;
      s.nbytes = end - start;
      s.tensors += 1;
    }
  }
  for (const auto &[_, s] : spans) {
    if (s.tensors != 9) {
      throw std::runtime_error("expected 9 tensors per expert span, got " + std::to_string(s.tensors));
    }
  }
  return spans;
}

static uint64_t fnv1a_update(uint64_t h, const std::vector<uint8_t> &buf) {
  constexpr uint64_t prime = 1099511628211ull;
  for (uint8_t b : buf) {
    h ^= b;
    h *= prime;
  }
  return h;
}

static uint64_t fnv1a_update_sample(uint64_t h, const std::vector<uint8_t> &buf) {
  constexpr uint64_t prime = 1099511628211ull;
  auto add = [&](uint8_t b) {
    h ^= b;
    h *= prime;
  };
  const size_t n = buf.size();
  const size_t sample = std::min<size_t>(64, n);
  for (size_t i = 0; i < sample; ++i) add(buf[i]);
  if (n > sample) {
    size_t start = n > 64 ? n - 64 : sample;
    for (size_t i = start; i < n; ++i) add(buf[i]);
  }
  for (int shift = 0; shift < 64; shift += 8) add(static_cast<uint8_t>((n >> shift) & 0xff));
  return h;
}

class FdCache {
 public:
  explicit FdCache(fs::path root) : root_(std::move(root)) {}
  ~FdCache() {
    for (auto &[_, fd] : fds_) {
      if (fd >= 0) ::close(fd);
    }
  }
  int get(const std::string &rel) {
    auto it = fds_.find(rel);
    if (it != fds_.end()) return it->second;
    fs::path p = root_ / rel;
    int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("open failed for " + p.string() + ": " + std::strerror(errno));
    fds_[rel] = fd;
    return fd;
  }
 private:
  fs::path root_;
  std::unordered_map<std::string, int> fds_;
};

static void read_exact(int fd, uint64_t offset, std::vector<uint8_t> &buf) {
  size_t done = 0;
  while (done < buf.size()) {
    ssize_t n = ::pread(fd, buf.data() + done, buf.size() - done, static_cast<off_t>(offset + done));
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error(std::string("pread failed: ") + std::strerror(errno));
    }
    if (n == 0) throw std::runtime_error("short pread at EOF");
    done += static_cast<size_t>(n);
  }
}

struct RunStats {
  uint64_t bytes_read = 0;
  uint64_t read_calls = 0;
  uint64_t cache_hits = 0;
  uint64_t cache_misses = 0;
  uint64_t evictions = 0;
  uint64_t checksum = 1469598103934665603ull;
};

struct CacheEntry {
  uint64_t freq = 0;
  uint64_t last_use = 0;
  uint64_t nbytes = 0;
};

static std::vector<int> route_experts(int layer, int token, int topk, const std::string &pattern) {
  std::vector<int> out;
  std::unordered_set<int> seen;
  int seed = 0;
  if (pattern == "fixed") seed = 0;
  else if (pattern == "hot") seed = (token % 2) * 17 + layer * 13;
  else seed = token * 17 + layer * 13;
  for (int k = 0; static_cast<int>(out.size()) < topk; ++k) {
    int expert = (seed + k * 31) % 192;
    if (seen.insert(expert).second) out.push_back(expert);
  }
  return out;
}

static void touch_or_load(
    const Span &span,
    const Args &args,
    FdCache &fds,
    std::vector<uint8_t> &buf,
    std::unordered_map<int, std::unordered_map<int, CacheEntry>> &cache,
    uint64_t tick,
    RunStats &stats) {
  auto &layer_cache = cache[span.layer];
  auto hit = layer_cache.find(span.expert);
  if (hit != layer_cache.end()) {
    hit->second.freq += 1;
    hit->second.last_use = tick;
    stats.cache_hits += 1;
    return;
  }

  stats.cache_misses += 1;
  if (static_cast<int>(layer_cache.size()) >= args.slot_bank) {
    auto victim = layer_cache.begin();
    for (auto it = layer_cache.begin(); it != layer_cache.end(); ++it) {
      bool take = false;
      if (args.policy == "lru") {
        take = it->second.last_use < victim->second.last_use;
      } else {
        take = (it->second.freq < victim->second.freq) ||
               (it->second.freq == victim->second.freq && it->second.last_use < victim->second.last_use);
      }
      if (take) victim = it;
    }
    layer_cache.erase(victim);
    stats.evictions += 1;
  }

  if (!args.no_read) {
    int fd = fds.get(span.file);
    buf.resize(static_cast<size_t>(span.nbytes));
    read_exact(fd, span.offset, buf);
    if (args.checksum == "full") stats.checksum = fnv1a_update(stats.checksum, buf);
    else if (args.checksum == "sample") stats.checksum = fnv1a_update_sample(stats.checksum, buf);
    stats.bytes_read += span.nbytes;
    stats.read_calls += 1;
  }
  layer_cache[span.expert] = CacheEntry{1, tick, span.nbytes};
}

static uint64_t cache_bytes(const std::unordered_map<int, std::unordered_map<int, CacheEntry>> &cache) {
  uint64_t total = 0;
  for (const auto &[_, layer_cache] : cache) {
    for (const auto &[__, entry] : layer_cache) total += entry.nbytes;
  }
  return total;
}

struct TraceEvent {
  uint64_t event = 0;
  int layer = 0;
  int batch = 0;
  int token = 0;
  std::vector<int> experts;
};

static std::vector<TraceEvent> load_trace_events(const fs::path &trace_path) {
  FILE *f = std::fopen(trace_path.c_str(), "r");
  if (!f) throw std::runtime_error("failed to open trace: " + trace_path.string() + ": " + std::strerror(errno));
  char *line = nullptr;
  size_t cap = 0;
  std::vector<TraceEvent> events;
  while (getline(&line, &cap, f) != -1) {
    std::string s(line);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    if (s.empty() || s[0] == '#') continue;
    if (s.rfind("event\t", 0) == 0) continue;
    auto cols = split(s, '\t');
    if (cols.size() != 5) {
      std::free(line);
      std::fclose(f);
      throw std::runtime_error("bad route trace row with " + std::to_string(cols.size()) + " cols");
    }
    TraceEvent event;
    event.event = std::stoull(cols[0]);
    event.layer = std::stoi(cols[1]);
    event.batch = std::stoi(cols[2]);
    event.token = std::stoi(cols[3]);
    event.experts = parse_csv_ints(cols[4]);
    if (event.experts.empty()) {
      std::free(line);
      std::fclose(f);
      throw std::runtime_error("route trace row has empty experts list");
    }
    events.push_back(std::move(event));
  }
  std::free(line);
  std::fclose(f);
  if (events.empty()) throw std::runtime_error("route trace has no events");
  std::sort(events.begin(), events.end(), [](const TraceEvent &a, const TraceEvent &b) { return a.event < b.event; });
  return events;
}

struct ParityFixture {
  int layer = 0;
  int topk = 0;
  std::vector<int> experts;
  std::vector<float> route_weights;
  std::vector<float> hidden;
  std::vector<float> expected;
};

static std::string read_text_file(const fs::path &path) {
  FILE *f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("failed to open file: " + path.string() + ": " + std::strerror(errno));
  std::string out;
  char buf[1 << 16];
  while (true) {
    size_t n = std::fread(buf, 1, sizeof(buf), f);
    if (n) out.append(buf, n);
    if (n < sizeof(buf)) {
      if (std::ferror(f)) {
        std::fclose(f);
        throw std::runtime_error("failed reading file: " + path.string());
      }
      break;
    }
  }
  std::fclose(f);
  return out;
}

static size_t find_json_key(const std::string &text, const std::string &key) {
  auto pos = text.find("\"" + key + "\"");
  if (pos == std::string::npos) throw std::runtime_error("fixture missing key: " + key);
  return pos;
}

static int parse_json_int(const std::string &text, const std::string &key) {
  auto pos = find_json_key(text, key);
  pos = text.find(':', pos);
  if (pos == std::string::npos) throw std::runtime_error("bad scalar key: " + key);
  char *end = nullptr;
  long v = std::strtol(text.c_str() + pos + 1, &end, 10);
  if (end == text.c_str() + pos + 1) throw std::runtime_error("bad int for key: " + key);
  return static_cast<int>(v);
}

static std::string json_array_body(const std::string &text, const std::string &key) {
  auto pos = find_json_key(text, key);
  auto start = text.find('[', pos);
  if (start == std::string::npos) throw std::runtime_error("bad array key: " + key);
  int depth = 0;
  for (size_t i = start; i < text.size(); ++i) {
    if (text[i] == '[') ++depth;
    else if (text[i] == ']') {
      --depth;
      if (depth == 0) return text.substr(start + 1, i - start - 1);
    }
  }
  throw std::runtime_error("unterminated array for key: " + key);
}

static std::vector<float> parse_json_float_array(const std::string &text, const std::string &key) {
  std::string body = json_array_body(text, key);
  std::vector<float> out;
  const char *p = body.c_str();
  while (*p) {
    while (*p && (*p == ',' || std::isspace(static_cast<unsigned char>(*p)))) ++p;
    if (!*p) break;
    char *end = nullptr;
    double v = std::strtod(p, &end);
    if (end == p) throw std::runtime_error("bad float in array: " + key);
    out.push_back(static_cast<float>(v));
    p = end;
  }
  return out;
}

static std::vector<int> parse_json_int_array(const std::string &text, const std::string &key) {
  std::string body = json_array_body(text, key);
  std::vector<int> out;
  const char *p = body.c_str();
  while (*p) {
    while (*p && (*p == ',' || std::isspace(static_cast<unsigned char>(*p)))) ++p;
    if (!*p) break;
    char *end = nullptr;
    long v = std::strtol(p, &end, 10);
    if (end == p) throw std::runtime_error("bad int in array: " + key);
    out.push_back(static_cast<int>(v));
    p = end;
  }
  return out;
}

static ParityFixture load_parity_fixture(const fs::path &path) {
  std::string text = read_text_file(path);
  ParityFixture fx;
  fx.layer = parse_json_int(text, "layer");
  fx.topk = parse_json_int(text, "topk");
  fx.experts = parse_json_int_array(text, "experts");
  fx.route_weights = parse_json_float_array(text, "route_weights");
  fx.hidden = parse_json_float_array(text, "hidden");
  fx.expected = parse_json_float_array(text, "expected_routed");
  if (fx.experts.size() != fx.route_weights.size()) throw std::runtime_error("fixture experts/route_weights length mismatch");
  if (fx.experts.size() != static_cast<size_t>(fx.topk)) throw std::runtime_error("fixture topk does not match experts length");
  if (fx.hidden.size() != 4096) throw std::runtime_error("fixture hidden must have 4096 floats");
  if (fx.expected.size() != 4096) throw std::runtime_error("fixture expected_routed must have 4096 floats");
  return fx;
}

struct TensorSlice {
  uint64_t offset = 0;
  uint64_t nbytes = 0;
  const uint8_t *ptr = nullptr;
};

struct ExpertBank {
  int layer = 0;
  int expert = 0;
  std::vector<uint8_t> raw;
  TensorSlice up_w, up_s, up_b;
  TensorSlice gate_w, gate_s, gate_b;
  TensorSlice down_w, down_s, down_b;
};

static const Entry *find_entry(const std::vector<Entry> &entries, int layer, int expert, const std::string &family, const std::string &kind) {
  for (const auto &entry : entries) {
    if (entry.layer == layer && entry.expert == expert && entry.family == family && entry.kind == kind) return &entry;
  }
  return nullptr;
}

static void set_slice(TensorSlice &slice, const Entry *entry, const Span &span, const std::vector<uint8_t> &raw) {
  if (!entry) throw std::runtime_error("missing tensor entry while materializing expert");
  if (entry->offset < span.offset || entry->offset + entry->nbytes > span.offset + span.nbytes) {
    throw std::runtime_error("tensor entry outside expert span");
  }
  slice.offset = entry->offset - span.offset;
  slice.nbytes = entry->nbytes;
  slice.ptr = raw.data() + slice.offset;
}

static ExpertBank load_expert_bank(const std::vector<Entry> &entries, const std::unordered_map<uint64_t, Span> &spans, FdCache &fds, int layer, int expert) {
  auto span_it = spans.find(span_key(layer, expert));
  if (span_it == spans.end()) throw std::runtime_error("missing expert span for parity fixture");
  const Span &span = span_it->second;
  ExpertBank bank;
  bank.layer = layer;
  bank.expert = expert;
  bank.raw.resize(static_cast<size_t>(span.nbytes));
  int fd = fds.get(span.file);
  read_exact(fd, span.offset, bank.raw);
  set_slice(bank.up_w, find_entry(entries, layer, expert, "up_proj", "weight"), span, bank.raw);
  set_slice(bank.up_s, find_entry(entries, layer, expert, "up_proj", "scales"), span, bank.raw);
  set_slice(bank.up_b, find_entry(entries, layer, expert, "up_proj", "biases"), span, bank.raw);
  set_slice(bank.gate_w, find_entry(entries, layer, expert, "gate_proj", "weight"), span, bank.raw);
  set_slice(bank.gate_s, find_entry(entries, layer, expert, "gate_proj", "scales"), span, bank.raw);
  set_slice(bank.gate_b, find_entry(entries, layer, expert, "gate_proj", "biases"), span, bank.raw);
  set_slice(bank.down_w, find_entry(entries, layer, expert, "down_proj", "weight"), span, bank.raw);
  set_slice(bank.down_s, find_entry(entries, layer, expert, "down_proj", "scales"), span, bank.raw);
  set_slice(bank.down_b, find_entry(entries, layer, expert, "down_proj", "biases"), span, bank.raw);
  return bank;
}

static uint32_t load_u32_le(const uint8_t *p) {
  uint32_t v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

static uint16_t load_u16_le(const uint8_t *p) {
  uint16_t v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

static float bf16_to_float(uint16_t bf) {
  uint32_t bits = static_cast<uint32_t>(bf) << 16;
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

static float dequant_q4_affine(const TensorSlice &w, const TensorSlice &s, const TensorSlice &b, int out, int in, int packed_words, int groups) {
  int word_index = out * packed_words + (in >> 3);
  int nibble = in & 7;
  uint32_t word = load_u32_le(w.ptr + static_cast<size_t>(word_index) * 4);
  uint32_t q = (word >> (nibble * 4)) & 0xFu;
  int group = in >> 6;
  if (group >= groups) throw std::runtime_error("quant group out of bounds");
  size_t sb_index = static_cast<size_t>(out * groups + group) * 2;
  float scale = bf16_to_float(load_u16_le(s.ptr + sb_index));
  float bias = bf16_to_float(load_u16_le(b.ptr + sb_index));
  return static_cast<float>(q) * scale + bias;
}

static void qlinear(const std::vector<float> &x, const TensorSlice &w, const TensorSlice &s, const TensorSlice &b, int out_dim, int in_dim, int packed_words, int groups, std::vector<float> &out) {
  out.assign(out_dim, 0.0f);
  for (int o = 0; o < out_dim; ++o) {
    float acc = 0.0f;
    for (int i = 0; i < in_dim; ++i) acc += x[static_cast<size_t>(i)] * dequant_q4_affine(w, s, b, o, i, packed_words, groups);
    out[static_cast<size_t>(o)] = acc;
  }
}

static float silu(float x) {
  return x / (1.0f + std::exp(-x));
}

static std::vector<float> compute_routed_fixture(const ParityFixture &fx, const std::vector<Entry> &entries, const std::unordered_map<uint64_t, Span> &spans, const Args &args, uint64_t &bytes_read, int &read_calls) {
  FdCache fds(args.root);
  std::vector<float> routed(4096, 0.0f), up, gate, hidden, down;
  bytes_read = 0;
  read_calls = 0;
  for (size_t eidx = 0; eidx < fx.experts.size(); ++eidx) {
    ExpertBank bank = load_expert_bank(entries, spans, fds, fx.layer, fx.experts[eidx]);
    bytes_read += bank.raw.size();
    ++read_calls;
    qlinear(fx.hidden, bank.up_w, bank.up_s, bank.up_b, 1536, 4096, 512, 64, up);
    qlinear(fx.hidden, bank.gate_w, bank.gate_s, bank.gate_b, 1536, 4096, 512, 64, gate);
    hidden.resize(1536);
    for (int i = 0; i < 1536; ++i) hidden[static_cast<size_t>(i)] = silu(gate[static_cast<size_t>(i)]) * up[static_cast<size_t>(i)];
    qlinear(hidden, bank.down_w, bank.down_s, bank.down_b, 4096, 1536, 192, 24, down);
    float weight = fx.route_weights[eidx];
    for (int i = 0; i < 4096; ++i) routed[static_cast<size_t>(i)] += weight * down[static_cast<size_t>(i)];
  }
  return routed;
}

struct ErrorStats {
  double max_abs = 0.0;
  double mean_abs = 0.0;
  double rmse = 0.0;
  double expected_max_abs = 0.0;
  double max_rel_to_expected = 0.0;
  int max_index = 0;
};

static ErrorStats compare_vectors(const std::vector<float> &actual, const std::vector<float> &expected) {
  if (actual.size() != expected.size()) throw std::runtime_error("compare_vectors size mismatch");
  ErrorStats stats;
  double sum_abs = 0.0;
  double sum_sq = 0.0;
  for (size_t i = 0; i < actual.size(); ++i) {
    double diff = static_cast<double>(actual[i]) - static_cast<double>(expected[i]);
    double ad = std::fabs(diff);
    stats.expected_max_abs = std::max(stats.expected_max_abs, std::fabs(static_cast<double>(expected[i])));
    if (ad > stats.max_abs) {
      stats.max_abs = ad;
      stats.max_index = static_cast<int>(i);
    }
    sum_abs += ad;
    sum_sq += diff * diff;
  }
  stats.mean_abs = sum_abs / static_cast<double>(actual.size());
  stats.rmse = std::sqrt(sum_sq / static_cast<double>(actual.size()));
  stats.max_rel_to_expected = stats.max_abs / std::max(stats.expected_max_abs, 1.0e-12);
  return stats;
}

static bool parity_passes(const ErrorStats &stats) {
  return stats.max_abs <= std::max(1.0e-4, 2.0e-2 * stats.expected_max_abs);
}

struct ParityResult {
  fs::path fixture;
  int layer = 0;
  int topk = 0;
  int read_calls = 0;
  uint64_t bytes_read = 0;
  double compute_elapsed_s = 0.0;
  ErrorStats error;
};

static ParityResult run_parity_fixture(const fs::path &fixture_path, const std::vector<Entry> &entries, const std::unordered_map<uint64_t, Span> &spans, const Args &args) {
  ParityFixture fx = load_parity_fixture(fixture_path);
  auto t0 = std::chrono::steady_clock::now();
  uint64_t bytes_read = 0;
  int read_calls = 0;
  std::vector<float> actual = compute_routed_fixture(fx, entries, spans, args, bytes_read, read_calls);
  auto t1 = std::chrono::steady_clock::now();
  ParityResult result;
  result.fixture = fixture_path;
  result.layer = fx.layer;
  result.topk = fx.topk;
  result.read_calls = read_calls;
  result.bytes_read = bytes_read;
  result.compute_elapsed_s = std::chrono::duration<double>(t1 - t0).count();
  result.error = compare_vectors(actual, fx.expected);
  return result;
}

static std::vector<fs::path> load_fixture_list(const fs::path &list_path) {
  std::string text = read_text_file(list_path);
  std::stringstream ss(text);
  std::string line;
  std::vector<fs::path> paths;
  while (std::getline(ss, line)) {
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || std::isspace(static_cast<unsigned char>(line.back())))) line.pop_back();
    size_t start = 0;
    while (start < line.size() && std::isspace(static_cast<unsigned char>(line[start]))) ++start;
    line = line.substr(start);
    if (line.empty() || line[0] == '#') continue;
    fs::path path(line);
    if (path.is_relative()) path = list_path.parent_path() / path;
    paths.push_back(path);
  }
  if (paths.empty()) throw std::runtime_error("fixture list is empty: " + list_path.string());
  return paths;
}

static std::string json_escape(const std::string &s) {
  std::string out;
  for (char c : s) {
    if (c == '"' || c == '\\') { out.push_back('\\'); out.push_back(c); }
    else if (c == '\n') out += "\\n";
    else out.push_back(c);
  }
  return out;
}

int main(int argc, char **argv) {
  try {
    Args args = parse_args(argc, argv);
    auto t_index0 = std::chrono::steady_clock::now();
    std::vector<Entry> entries = load_entries(args.index);
    auto spans = build_spans(entries);
    auto t_index1 = std::chrono::steady_clock::now();

    std::vector<Span> plan;
    if (args.layer.has_value()) {
      for (int expert : args.experts) {
        auto it = spans.find(span_key(*args.layer, expert));
        if (it == spans.end()) throw std::runtime_error("missing span for layer/expert");
        plan.push_back(it->second);
      }
    } else {
      for (int layer : args.explicit_layers) {
        for (int expert : args.experts) {
          auto it = spans.find(span_key(layer, expert));
          if (it == spans.end()) throw std::runtime_error("missing span for layer/expert");
          plan.push_back(it->second);
        }
      }
    }

    uint64_t planned_bytes = 0;
    for (const auto &s : plan) planned_bytes += s.nbytes;

    auto index_elapsed = std::chrono::duration<double>(t_index1 - t_index0).count();

    if (!args.fixture.empty()) {
      ParityResult result = run_parity_fixture(args.fixture, entries, spans, args);
      std::cout << "{\n";
      std::cout << "  \"ok\": true,\n";
      std::cout << "  \"mode\": \"parity-fixture\",\n";
      std::cout << "  \"fixture\": \"" << json_escape(args.fixture.string()) << "\",\n";
      std::cout << "  \"index\": \"" << json_escape(args.index.string()) << "\",\n";
      std::cout << "  \"root\": \"" << json_escape(args.root.string()) << "\",\n";
      std::cout << "  \"layer\": " << result.layer << ",\n";
      std::cout << "  \"topk\": " << result.topk << ",\n";
      std::cout << "  \"read_calls\": " << result.read_calls << ",\n";
      std::cout << "  \"bytes_read\": " << result.bytes_read << ",\n";
      std::cout << "  \"gib_read\": " << static_cast<double>(result.bytes_read) / (1024.0 * 1024.0 * 1024.0) << ",\n";
      std::cout << "  \"index_load_s\": " << index_elapsed << ",\n";
      std::cout << "  \"compute_elapsed_s\": " << result.compute_elapsed_s << ",\n";
      std::cout << "  \"max_abs_error\": " << result.error.max_abs << ",\n";
      std::cout << "  \"mean_abs_error\": " << result.error.mean_abs << ",\n";
      std::cout << "  \"rmse\": " << result.error.rmse << ",\n";
      std::cout << "  \"expected_max_abs\": " << result.error.expected_max_abs << ",\n";
      std::cout << "  \"max_rel_to_expected\": " << result.error.max_rel_to_expected << ",\n";
      std::cout << "  \"parity_abs_floor\": 0.0001,\n";
      std::cout << "  \"parity_rel_threshold\": 0.02,\n";
      std::cout << "  \"parity_pass\": " << (parity_passes(result.error) ? "true" : "false") << ",\n";
      std::cout << "  \"max_error_index\": " << result.error.max_index << "\n";
      std::cout << "}\n";
      return 0;
    }

    if (!args.fixture_list.empty()) {
      auto fixture_paths = load_fixture_list(args.fixture_list);
      std::vector<ParityResult> results;
      results.reserve(fixture_paths.size());
      auto t0 = std::chrono::steady_clock::now();
      uint64_t total_bytes = 0;
      int total_read_calls = 0;
      double max_abs = 0.0;
      double max_rel = 0.0;
      double worst_mean_abs = 0.0;
      double worst_rmse = 0.0;
      int worst_layer = -1;
      int worst_rel_layer = -1;
      bool all_pass = true;
      for (const auto &fixture_path : fixture_paths) {
        ParityResult result = run_parity_fixture(fixture_path, entries, spans, args);
        total_bytes += result.bytes_read;
        total_read_calls += result.read_calls;
        all_pass = all_pass && parity_passes(result.error);
        if (result.error.max_abs > max_abs) {
          max_abs = result.error.max_abs;
          worst_mean_abs = result.error.mean_abs;
          worst_rmse = result.error.rmse;
          worst_layer = result.layer;
        }
        if (result.error.max_rel_to_expected > max_rel) {
          max_rel = result.error.max_rel_to_expected;
          worst_rel_layer = result.layer;
        }
        results.push_back(std::move(result));
      }
      auto t1 = std::chrono::steady_clock::now();
      double elapsed = std::chrono::duration<double>(t1 - t0).count();
      std::cout << "{\n";
      std::cout << "  \"ok\": true,\n";
      std::cout << "  \"mode\": \"parity-fixture-list\",\n";
      std::cout << "  \"fixture_list\": \"" << json_escape(args.fixture_list.string()) << "\",\n";
      std::cout << "  \"index\": \"" << json_escape(args.index.string()) << "\",\n";
      std::cout << "  \"root\": \"" << json_escape(args.root.string()) << "\",\n";
      std::cout << "  \"fixtures\": " << results.size() << ",\n";
      std::cout << "  \"read_calls\": " << total_read_calls << ",\n";
      std::cout << "  \"bytes_read\": " << total_bytes << ",\n";
      std::cout << "  \"gib_read\": " << static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0) << ",\n";
      std::cout << "  \"index_load_s\": " << index_elapsed << ",\n";
      std::cout << "  \"compute_elapsed_s\": " << elapsed << ",\n";
      std::cout << "  \"max_abs_error\": " << max_abs << ",\n";
      std::cout << "  \"max_rel_to_expected\": " << max_rel << ",\n";
      std::cout << "  \"worst_mean_abs_error\": " << worst_mean_abs << ",\n";
      std::cout << "  \"worst_rmse\": " << worst_rmse << ",\n";
      std::cout << "  \"worst_layer\": " << worst_layer << ",\n";
      std::cout << "  \"worst_rel_layer\": " << worst_rel_layer << ",\n";
      std::cout << "  \"parity_abs_floor\": 0.0001,\n";
      std::cout << "  \"parity_rel_threshold\": 0.02,\n";
      std::cout << "  \"parity_pass\": " << (all_pass ? "true" : "false") << ",\n";
      std::cout << "  \"layers\": [";
      for (size_t i = 0; i < results.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << results[i].layer;
      }
      std::cout << "],\n";
      std::cout << "  \"per_layer\": [\n";
      for (size_t i = 0; i < results.size(); ++i) {
        const auto &row = results[i];
        std::cout << "    {\"layer\": " << row.layer
                  << ", \"topk\": " << row.topk
                  << ", \"read_calls\": " << row.read_calls
                  << ", \"bytes_read\": " << row.bytes_read
                  << ", \"compute_elapsed_s\": " << row.compute_elapsed_s
                  << ", \"max_abs_error\": " << row.error.max_abs
                  << ", \"mean_abs_error\": " << row.error.mean_abs
                  << ", \"rmse\": " << row.error.rmse
                  << ", \"expected_max_abs\": " << row.error.expected_max_abs
                  << ", \"max_rel_to_expected\": " << row.error.max_rel_to_expected
                  << ", \"parity_pass\": " << (parity_passes(row.error) ? "true" : "false")
                  << "}" << (i + 1 == results.size() ? "\n" : ",\n");
      }
      std::cout << "  ]\n";
      std::cout << "}\n";
      return 0;
    }

    if (!args.trace.empty()) {
      auto events = load_trace_events(args.trace);
      FdCache fds(args.root);
      std::vector<uint8_t> buf;
      std::unordered_map<int, std::unordered_map<int, CacheEntry>> cache;
      std::unordered_set<int> trace_layers;
      RunStats stats;
      uint64_t tick = 0;
      uint64_t selections = 0;
      size_t max_k = 0;
      auto t_read0 = std::chrono::steady_clock::now();
      for (int iter = 0; iter < args.iterations; ++iter) {
        for (const auto &event : events) {
          trace_layers.insert(event.layer);
          max_k = std::max(max_k, event.experts.size());
          selections += event.experts.size();
          for (int expert : event.experts) {
            auto it = spans.find(span_key(event.layer, expert));
            if (it == spans.end()) throw std::runtime_error("missing span for trace layer/expert");
            touch_or_load(it->second, args, fds, buf, cache, ++tick, stats);
          }
        }
      }
      auto t_read1 = std::chrono::steady_clock::now();
      auto elapsed = std::chrono::duration<double>(t_read1 - t_read0).count();
      double gib = static_cast<double>(stats.bytes_read) / (1024.0 * 1024.0 * 1024.0);
      double gib_s = elapsed > 0.0 ? gib / elapsed : 0.0;
      uint64_t final_cache_bytes = cache_bytes(cache);
      std::cout << "{\n";
      std::cout << "  \"ok\": true,\n";
      std::cout << "  \"mode\": \"route-trace\",\n";
      std::cout << "  \"trace\": \"" << json_escape(args.trace.string()) << "\",\n";
      std::cout << "  \"index\": \"" << json_escape(args.index.string()) << "\",\n";
      std::cout << "  \"root\": \"" << json_escape(args.root.string()) << "\",\n";
      std::cout << "  \"index_entries\": " << entries.size() << ",\n";
      std::cout << "  \"expert_spans\": " << spans.size() << ",\n";
      std::cout << "  \"events\": " << events.size() << ",\n";
      std::cout << "  \"layers\": " << trace_layers.size() << ",\n";
      std::cout << "  \"selected_experts\": " << selections << ",\n";
      std::cout << "  \"max_k\": " << max_k << ",\n";
      std::cout << "  \"slot_bank\": " << args.slot_bank << ",\n";
      std::cout << "  \"policy\": \"" << json_escape(args.policy) << "\",\n";
      std::cout << "  \"iterations\": " << args.iterations << ",\n";
      std::cout << "  \"cache_hits\": " << stats.cache_hits << ",\n";
      std::cout << "  \"cache_misses\": " << stats.cache_misses << ",\n";
      std::cout << "  \"evictions\": " << stats.evictions << ",\n";
      std::cout << "  \"read_calls\": " << stats.read_calls << ",\n";
      std::cout << "  \"bytes_read\": " << stats.bytes_read << ",\n";
      std::cout << "  \"gib_read\": " << gib << ",\n";
      std::cout << "  \"final_cache_bytes\": " << final_cache_bytes << ",\n";
      std::cout << "  \"final_cache_gib\": " << static_cast<double>(final_cache_bytes) / (1024.0 * 1024.0 * 1024.0) << ",\n";
      std::cout << "  \"index_load_s\": " << index_elapsed << ",\n";
      std::cout << "  \"read_elapsed_s\": " << elapsed << ",\n";
      std::cout << "  \"gib_per_s\": " << gib_s << ",\n";
      std::cout << "  \"checksum_fnv1a64\": \"0x" << std::hex << stats.checksum << std::dec << "\",\n";
      std::cout << "  \"checksum_mode\": \"" << json_escape(args.checksum) << "\",\n";
      std::cout << "  \"no_read\": " << (args.no_read ? "true" : "false") << "\n";
      std::cout << "}\n";
      return 0;
    }

    if (args.simulate_tokens > 0) {
      FdCache fds(args.root);
      std::vector<uint8_t> buf;
      std::unordered_map<int, std::unordered_map<int, CacheEntry>> cache;
      RunStats stats;
      uint64_t tick = 0;
      auto t_read0 = std::chrono::steady_clock::now();
      for (int iter = 0; iter < args.iterations; ++iter) {
        for (int token = 0; token < args.simulate_tokens; ++token) {
          for (int layer : args.explicit_layers) {
            for (int expert : route_experts(layer, token, args.topk, args.route_pattern)) {
              auto it = spans.find(span_key(layer, expert));
              if (it == spans.end()) throw std::runtime_error("missing span for simulated layer/expert");
              touch_or_load(it->second, args, fds, buf, cache, ++tick, stats);
            }
          }
        }
      }
      auto t_read1 = std::chrono::steady_clock::now();
      auto elapsed = std::chrono::duration<double>(t_read1 - t_read0).count();
      double gib = static_cast<double>(stats.bytes_read) / (1024.0 * 1024.0 * 1024.0);
      double gib_s = elapsed > 0.0 ? gib / elapsed : 0.0;
      uint64_t final_cache_bytes = cache_bytes(cache);
      std::cout << "{\n";
      std::cout << "  \"ok\": true,\n";
      std::cout << "  \"mode\": \"cache-sim\",\n";
      std::cout << "  \"index\": \"" << json_escape(args.index.string()) << "\",\n";
      std::cout << "  \"root\": \"" << json_escape(args.root.string()) << "\",\n";
      std::cout << "  \"index_entries\": " << entries.size() << ",\n";
      std::cout << "  \"expert_spans\": " << spans.size() << ",\n";
      std::cout << "  \"layers\": " << args.explicit_layers.size() << ",\n";
      std::cout << "  \"topk\": " << args.topk << ",\n";
      std::cout << "  \"simulate_tokens\": " << args.simulate_tokens << ",\n";
      std::cout << "  \"slot_bank\": " << args.slot_bank << ",\n";
      std::cout << "  \"policy\": \"" << json_escape(args.policy) << "\",\n";
      std::cout << "  \"route_pattern\": \"" << json_escape(args.route_pattern) << "\",\n";
      std::cout << "  \"iterations\": " << args.iterations << ",\n";
      std::cout << "  \"cache_hits\": " << stats.cache_hits << ",\n";
      std::cout << "  \"cache_misses\": " << stats.cache_misses << ",\n";
      std::cout << "  \"evictions\": " << stats.evictions << ",\n";
      std::cout << "  \"read_calls\": " << stats.read_calls << ",\n";
      std::cout << "  \"bytes_read\": " << stats.bytes_read << ",\n";
      std::cout << "  \"gib_read\": " << gib << ",\n";
      std::cout << "  \"final_cache_bytes\": " << final_cache_bytes << ",\n";
      std::cout << "  \"final_cache_gib\": " << static_cast<double>(final_cache_bytes) / (1024.0 * 1024.0 * 1024.0) << ",\n";
      std::cout << "  \"index_load_s\": " << index_elapsed << ",\n";
      std::cout << "  \"read_elapsed_s\": " << elapsed << ",\n";
      std::cout << "  \"gib_per_s\": " << gib_s << ",\n";
      std::cout << "  \"checksum_fnv1a64\": \"0x" << std::hex << stats.checksum << std::dec << "\",\n";
      std::cout << "  \"checksum_mode\": \"" << json_escape(args.checksum) << "\",\n";
      std::cout << "  \"no_read\": " << (args.no_read ? "true" : "false") << "\n";
      std::cout << "}\n";
      return 0;
    }

    FdCache fds(args.root);
    uint64_t checksum = 1469598103934665603ull;
    uint64_t bytes_read = 0;
    uint64_t read_calls = 0;
    std::vector<uint8_t> buf;
    auto t_read0 = std::chrono::steady_clock::now();
    for (int iter = 0; iter < args.iterations; ++iter) {
      for (const auto &s : plan) {
        if (args.no_read) continue;
        bytes_read += s.nbytes;
        read_calls += 1;
        int fd = fds.get(s.file);
        buf.resize(static_cast<size_t>(s.nbytes));
        read_exact(fd, s.offset, buf);
        if (args.checksum == "full") checksum = fnv1a_update(checksum, buf);
        else if (args.checksum == "sample") checksum = fnv1a_update_sample(checksum, buf);
      }
    }
    auto t_read1 = std::chrono::steady_clock::now();

    auto elapsed = std::chrono::duration<double>(t_read1 - t_read0).count();
    double gib = static_cast<double>(bytes_read) / (1024.0 * 1024.0 * 1024.0);
    double gib_s = elapsed > 0.0 ? gib / elapsed : 0.0;

    std::cout << "{\n";
    std::cout << "  \"ok\": true,\n";
    std::cout << "  \"index\": \"" << json_escape(args.index.string()) << "\",\n";
    std::cout << "  \"root\": \"" << json_escape(args.root.string()) << "\",\n";
    std::cout << "  \"index_entries\": " << entries.size() << ",\n";
    std::cout << "  \"expert_spans\": " << spans.size() << ",\n";
    std::cout << "  \"planned_spans\": " << plan.size() << ",\n";
    std::cout << "  \"planned_bytes_per_iter\": " << planned_bytes << ",\n";
    std::cout << "  \"iterations\": " << args.iterations << ",\n";
    std::cout << "  \"read_calls\": " << read_calls << ",\n";
    std::cout << "  \"bytes_read\": " << bytes_read << ",\n";
    std::cout << "  \"gib_read\": " << gib << ",\n";
    std::cout << "  \"index_load_s\": " << index_elapsed << ",\n";
    std::cout << "  \"read_elapsed_s\": " << elapsed << ",\n";
    std::cout << "  \"gib_per_s\": " << gib_s << ",\n";
    std::cout << "  \"checksum_fnv1a64\": \"0x" << std::hex << checksum << std::dec << "\",\n";
    std::cout << "  \"checksum_mode\": \"" << json_escape(args.checksum) << "\",\n";
    std::cout << "  \"no_read\": " << (args.no_read ? "true" : "false") << "\n";
    std::cout << "}\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "hy3_sidecar_io: " << e.what() << "\n";
    usage();
    return 2;
  }
}
