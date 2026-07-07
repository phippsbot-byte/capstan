#include "hy3_routed_mlp.h"

#include "hy3_q4_affine.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>

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

static bool has_json_key(const std::string &text, const std::string &key) {
  return text.find("\"" + key + "\"") != std::string::npos;
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

static int parse_json_int_default(const std::string &text, const std::string &key, int fallback) {
  return has_json_key(text, key) ? parse_json_int(text, key) : fallback;
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

ParityFixture load_parity_fixture(const fs::path &path) {
  std::string text = read_text_file(path);
  ParityFixture fx;
  fx.layer = parse_json_int(text, "layer");
  fx.topk = parse_json_int(text, "topk");
  fx.seq_len = parse_json_int_default(text, "seq_len", 1);
  fx.experts_flat = has_json_key(text, "experts_flat") ? parse_json_int_array(text, "experts_flat") : parse_json_int_array(text, "experts");
  fx.route_weights_flat = has_json_key(text, "route_weights_flat") ? parse_json_float_array(text, "route_weights_flat") : parse_json_float_array(text, "route_weights");
  fx.hidden_flat = has_json_key(text, "hidden_tokens") ? parse_json_float_array(text, "hidden_tokens") : parse_json_float_array(text, "hidden");
  fx.expected_flat = has_json_key(text, "expected_routed_tokens") ? parse_json_float_array(text, "expected_routed_tokens") : parse_json_float_array(text, "expected_routed");
  if (fx.seq_len < 1) throw std::runtime_error("fixture seq_len must be >= 1");
  if (fx.experts_flat.size() != static_cast<size_t>(fx.seq_len * fx.topk)) throw std::runtime_error("fixture experts length mismatch");
  if (fx.route_weights_flat.size() != static_cast<size_t>(fx.seq_len * fx.topk)) throw std::runtime_error("fixture route_weights length mismatch");
  if (fx.hidden_flat.size() != static_cast<size_t>(fx.seq_len * 4096)) throw std::runtime_error("fixture hidden must have seq_len*4096 floats");
  if (fx.expected_flat.size() != static_cast<size_t>(fx.seq_len * 4096)) throw std::runtime_error("fixture expected_routed must have seq_len*4096 floats");
  return fx;
}

std::vector<fs::path> load_fixture_list(const fs::path &list_path) {
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

std::vector<float> compute_routed_fixture(const ParityFixture &fx, const std::vector<Entry> &entries, const std::unordered_map<uint64_t, Span> &spans, const fs::path &root, uint64_t &bytes_read, int &read_calls) {
  FdCache fds(root);
  std::vector<float> routed(static_cast<size_t>(fx.seq_len * 4096), 0.0f), up, gate, hidden, down;
  bytes_read = 0;
  read_calls = 0;
  for (int token = 0; token < fx.seq_len; ++token) {
    std::vector<float> x(fx.hidden_flat.begin() + static_cast<size_t>(token) * 4096, fx.hidden_flat.begin() + static_cast<size_t>(token + 1) * 4096);
    for (int k = 0; k < fx.topk; ++k) {
      size_t route = static_cast<size_t>(token * fx.topk + k);
      ExpertBank bank = load_expert_bank(entries, spans, fds, fx.layer, fx.experts_flat[route]);
      bytes_read += bank.raw.size();
      ++read_calls;
      qlinear(x, bank.up_w, bank.up_s, bank.up_b, 1536, 4096, 512, 64, up);
      qlinear(x, bank.gate_w, bank.gate_s, bank.gate_b, 1536, 4096, 512, 64, gate);
      hidden.resize(1536);
      for (int i = 0; i < 1536; ++i) hidden[static_cast<size_t>(i)] = silu(gate[static_cast<size_t>(i)]) * up[static_cast<size_t>(i)];
      qlinear(hidden, bank.down_w, bank.down_s, bank.down_b, 4096, 1536, 192, 24, down);
      float weight = fx.route_weights_flat[route];
      for (int i = 0; i < 4096; ++i) routed[static_cast<size_t>(token) * 4096 + static_cast<size_t>(i)] += weight * down[static_cast<size_t>(i)];
    }
  }
  return routed;
}

ErrorStats compare_vectors(const std::vector<float> &actual, const std::vector<float> &expected) {
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

bool parity_passes(const ErrorStats &stats) {
  return stats.max_abs <= std::max(1.0e-4, 2.0e-2 * stats.expected_max_abs);
}

ParityResult run_parity_fixture(const fs::path &fixture_path, const std::vector<Entry> &entries, const std::unordered_map<uint64_t, Span> &spans, const fs::path &root) {
  ParityFixture fx = load_parity_fixture(fixture_path);
  auto t0 = std::chrono::steady_clock::now();
  uint64_t bytes_read = 0;
  int read_calls = 0;
  std::vector<float> actual = compute_routed_fixture(fx, entries, spans, root, bytes_read, read_calls);
  auto t1 = std::chrono::steady_clock::now();
  ParityResult result;
  result.fixture = fixture_path;
  result.layer = fx.layer;
  result.topk = fx.topk;
  result.seq_len = fx.seq_len;
  result.read_calls = read_calls;
  result.bytes_read = bytes_read;
  result.compute_elapsed_s = std::chrono::duration<double>(t1 - t0).count();
  result.error = compare_vectors(actual, fx.expected_flat);
  return result;
}
