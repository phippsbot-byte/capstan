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

#include "hy3_expert_bank.h"
#include "hy3_routed_mlp.h"

namespace fs = std::filesystem;

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
  bool layer_major = false;
  bool route_exec = false;
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
               "[--layer-major] [--route-exec] [--iterations N] [--checksum sample|full|none] [--no-read]\n";
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
    else if (key == "--layer-major") args.layer_major = true;
    else if (key == "--route-exec") args.route_exec = true;
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
  if (args.route_exec) {
    if (args.fixture_list.empty()) throw std::runtime_error("--route-exec requires --fixture-list");
    args.layer_major = true;
  }
  if (args.layer_major && args.fixture.empty() && args.fixture_list.empty()) {
    throw std::runtime_error("--layer-major requires --fixture or --fixture-list");
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
      ParityResult result = run_parity_fixture(args.fixture, entries, spans, args.root, args.layer_major);
      std::cout << "{\n";
      std::cout << "  \"ok\": true,\n";
      std::cout << "  \"mode\": \"" << (args.layer_major ? "parity-fixture-layer-major" : "parity-fixture") << "\",\n";
      std::cout << "  \"fixture\": \"" << json_escape(args.fixture.string()) << "\",\n";
      std::cout << "  \"index\": \"" << json_escape(args.index.string()) << "\",\n";
      std::cout << "  \"root\": \"" << json_escape(args.root.string()) << "\",\n";
      std::cout << "  \"layer\": " << result.layer << ",\n";
      std::cout << "  \"topk\": " << result.topk << ",\n";
      std::cout << "  \"seq_len\": " << result.seq_len << ",\n";
      std::cout << "  \"layer_major\": " << (result.layer_major ? "true" : "false") << ",\n";
      std::cout << "  \"read_calls\": " << result.read_calls << ",\n";
      std::cout << "  \"naive_read_calls\": " << result.naive_read_calls << ",\n";
      std::cout << "  \"unique_expert_spans\": " << result.unique_expert_spans << ",\n";
      std::cout << "  \"dedup_saved_reads\": " << result.dedup_saved_reads << ",\n";
      std::cout << "  \"bytes_read\": " << result.bytes_read << ",\n";
      std::cout << "  \"naive_bytes_read\": " << result.naive_bytes_read << ",\n";
      std::cout << "  \"dedup_saved_bytes\": " << result.dedup_saved_bytes << ",\n";
      std::cout << "  \"gib_read\": " << static_cast<double>(result.bytes_read) / (1024.0 * 1024.0 * 1024.0) << ",\n";
      std::cout << "  \"naive_gib_read\": " << static_cast<double>(result.naive_bytes_read) / (1024.0 * 1024.0 * 1024.0) << ",\n";
      std::cout << "  \"dedup_saved_gib\": " << static_cast<double>(result.dedup_saved_bytes) / (1024.0 * 1024.0 * 1024.0) << ",\n";
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

    if (!args.fixture_list.empty() && args.route_exec) {
      auto fixture_paths = load_fixture_list(args.fixture_list);
      std::vector<ParityResult> results;
      results.reserve(fixture_paths.size());
      auto t0 = std::chrono::steady_clock::now();
      uint64_t total_bytes = 0;
      uint64_t total_naive_bytes = 0;
      uint64_t total_dedup_saved_bytes = 0;
      int total_read_calls = 0;
      int total_naive_read_calls = 0;
      int total_unique_expert_spans = 0;
      int total_dedup_saved_reads = 0;
      int token_layer_events = 0;
      int seq_len = -1;
      int topk = -1;
      double max_abs = 0.0;
      double max_rel = 0.0;
      double worst_mean_abs = 0.0;
      double worst_rmse = 0.0;
      int worst_layer = -1;
      int worst_rel_layer = -1;
      bool all_pass = true;
      for (const auto &fixture_path : fixture_paths) {
        ParityResult result = run_parity_fixture(fixture_path, entries, spans, args.root, true);
        if (seq_len < 0) seq_len = result.seq_len;
        if (topk < 0) topk = result.topk;
        if (result.seq_len != seq_len) throw std::runtime_error("--route-exec fixture list has mixed seq_len values");
        if (result.topk != topk) throw std::runtime_error("--route-exec fixture list has mixed topk values");
        total_bytes += result.bytes_read;
        total_naive_bytes += result.naive_bytes_read;
        total_dedup_saved_bytes += result.dedup_saved_bytes;
        total_read_calls += result.read_calls;
        total_naive_read_calls += result.naive_read_calls;
        total_unique_expert_spans += result.unique_expert_spans;
        total_dedup_saved_reads += result.dedup_saved_reads;
        token_layer_events += result.seq_len;
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
      std::cout << "  \"mode\": \"prompt-route-exec\",\n";
      std::cout << "  \"execution_scope\": \"routed_moe_only\",\n";
      std::cout << "  \"input_schema\": \"hy3-routed-layer-parity-v2-fixture-list\",\n";
      std::cout << "  \"fixture_list\": \"" << json_escape(args.fixture_list.string()) << "\",\n";
      std::cout << "  \"index\": \"" << json_escape(args.index.string()) << "\",\n";
      std::cout << "  \"root\": \"" << json_escape(args.root.string()) << "\",\n";
      std::cout << "  \"fixtures\": " << results.size() << ",\n";
      std::cout << "  \"layers\": " << results.size() << ",\n";
      std::cout << "  \"seq_len\": " << seq_len << ",\n";
      std::cout << "  \"topk\": " << topk << ",\n";
      std::cout << "  \"token_layer_events\": " << token_layer_events << ",\n";
      std::cout << "  \"selected_routes\": " << total_naive_read_calls << ",\n";
      std::cout << "  \"read_calls\": " << total_read_calls << ",\n";
      std::cout << "  \"naive_read_calls\": " << total_naive_read_calls << ",\n";
      std::cout << "  \"unique_expert_spans\": " << total_unique_expert_spans << ",\n";
      std::cout << "  \"dedup_saved_reads\": " << total_dedup_saved_reads << ",\n";
      std::cout << "  \"bytes_read\": " << total_bytes << ",\n";
      std::cout << "  \"naive_bytes_read\": " << total_naive_bytes << ",\n";
      std::cout << "  \"dedup_saved_bytes\": " << total_dedup_saved_bytes << ",\n";
      std::cout << "  \"gib_read\": " << static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0) << ",\n";
      std::cout << "  \"naive_gib_read\": " << static_cast<double>(total_naive_bytes) / (1024.0 * 1024.0 * 1024.0) << ",\n";
      std::cout << "  \"dedup_saved_gib\": " << static_cast<double>(total_dedup_saved_bytes) / (1024.0 * 1024.0 * 1024.0) << ",\n";
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
      std::cout << "  \"parity_pass\": " << (all_pass ? "true" : "false") << "\n";
      std::cout << "}\n";
      return 0;
    }

    if (!args.fixture_list.empty()) {
      auto fixture_paths = load_fixture_list(args.fixture_list);
      std::vector<ParityResult> results;
      results.reserve(fixture_paths.size());
      auto t0 = std::chrono::steady_clock::now();
      uint64_t total_bytes = 0;
      uint64_t total_naive_bytes = 0;
      uint64_t total_dedup_saved_bytes = 0;
      int total_read_calls = 0;
      int total_naive_read_calls = 0;
      int total_unique_expert_spans = 0;
      int total_dedup_saved_reads = 0;
      double max_abs = 0.0;
      double max_rel = 0.0;
      double worst_mean_abs = 0.0;
      double worst_rmse = 0.0;
      int worst_layer = -1;
      int worst_rel_layer = -1;
      bool all_pass = true;
      for (const auto &fixture_path : fixture_paths) {
        ParityResult result = run_parity_fixture(fixture_path, entries, spans, args.root, args.layer_major);
        total_bytes += result.bytes_read;
        total_naive_bytes += result.naive_bytes_read;
        total_dedup_saved_bytes += result.dedup_saved_bytes;
        total_read_calls += result.read_calls;
        total_naive_read_calls += result.naive_read_calls;
        total_unique_expert_spans += result.unique_expert_spans;
        total_dedup_saved_reads += result.dedup_saved_reads;
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
      std::cout << "  \"mode\": \"" << (args.layer_major ? "parity-fixture-list-layer-major" : "parity-fixture-list") << "\",\n";
      std::cout << "  \"fixture_list\": \"" << json_escape(args.fixture_list.string()) << "\",\n";
      std::cout << "  \"index\": \"" << json_escape(args.index.string()) << "\",\n";
      std::cout << "  \"root\": \"" << json_escape(args.root.string()) << "\",\n";
      std::cout << "  \"fixtures\": " << results.size() << ",\n";
      std::cout << "  \"seq_len\": [";
      for (size_t i = 0; i < results.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << results[i].seq_len;
      }
      std::cout << "],\n";
      std::cout << "  \"layer_major\": " << (args.layer_major ? "true" : "false") << ",\n";
      std::cout << "  \"read_calls\": " << total_read_calls << ",\n";
      std::cout << "  \"naive_read_calls\": " << total_naive_read_calls << ",\n";
      std::cout << "  \"unique_expert_spans\": " << total_unique_expert_spans << ",\n";
      std::cout << "  \"dedup_saved_reads\": " << total_dedup_saved_reads << ",\n";
      std::cout << "  \"bytes_read\": " << total_bytes << ",\n";
      std::cout << "  \"naive_bytes_read\": " << total_naive_bytes << ",\n";
      std::cout << "  \"dedup_saved_bytes\": " << total_dedup_saved_bytes << ",\n";
      std::cout << "  \"gib_read\": " << static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0) << ",\n";
      std::cout << "  \"naive_gib_read\": " << static_cast<double>(total_naive_bytes) / (1024.0 * 1024.0 * 1024.0) << ",\n";
      std::cout << "  \"dedup_saved_gib\": " << static_cast<double>(total_dedup_saved_bytes) / (1024.0 * 1024.0 * 1024.0) << ",\n";
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
                  << ", \"seq_len\": " << row.seq_len
                  << ", \"layer_major\": " << (row.layer_major ? "true" : "false")
                  << ", \"read_calls\": " << row.read_calls
                  << ", \"naive_read_calls\": " << row.naive_read_calls
                  << ", \"unique_expert_spans\": " << row.unique_expert_spans
                  << ", \"dedup_saved_reads\": " << row.dedup_saved_reads
                  << ", \"bytes_read\": " << row.bytes_read
                  << ", \"naive_bytes_read\": " << row.naive_bytes_read
                  << ", \"dedup_saved_bytes\": " << row.dedup_saved_bytes
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
