#include "hy3_expert_bank.h"
#include "hy3_q4_affine.h"

#include <algorithm>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

static constexpr int HIDDEN_DIM = 4096;
static constexpr int INTER_DIM = 1536;

static uint32_t load_u32_le_local(const uint8_t *p) {
  uint32_t v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

static uint16_t load_u16_le_local(const uint8_t *p) {
  uint16_t v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

static float bf16_to_float_local(uint16_t bf) {
  uint32_t bits = static_cast<uint32_t>(bf) << 16;
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
static inline float hsum_f32(float32x4_t v) {
#if defined(__aarch64__)
  return vaddvq_f32(v);
#else
  float tmp[4];
  vst1q_f32(tmp, v);
  return tmp[0] + tmp[1] + tmp[2] + tmp[3];
#endif
}
#endif

static inline float qdot64(const uint8_t *words, const float *x) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  float32x4_t acc = vdupq_n_f32(0.0f);
  for (int j = 0; j < 8; ++j) {
    const uint32_t word = load_u32_le_local(words + static_cast<size_t>(j) * 4);
    const uint32_t q0_raw[4] = {
        word & 0xFu,
        (word >> 4) & 0xFu,
        (word >> 8) & 0xFu,
        (word >> 12) & 0xFu,
    };
    const uint32_t q1_raw[4] = {
        (word >> 16) & 0xFu,
        (word >> 20) & 0xFu,
        (word >> 24) & 0xFu,
        (word >> 28) & 0xFu,
    };
    acc = vmlaq_f32(acc, vcvtq_f32_u32(vld1q_u32(q0_raw)), vld1q_f32(x + static_cast<size_t>(j) * 8));
    acc = vmlaq_f32(acc, vcvtq_f32_u32(vld1q_u32(q1_raw)), vld1q_f32(x + static_cast<size_t>(j) * 8 + 4));
  }
  return hsum_f32(acc);
#else
  float acc = 0.0f;
  for (int i = 0; i < 64; ++i) {
    uint32_t word = load_u32_le_local(words + static_cast<size_t>(i >> 3) * 4);
    uint32_t q = (word >> ((i & 7) * 4)) & 0xFu;
    acc += static_cast<float>(q) * x[i];
  }
  return acc;
#endif
}

static void compute_group_sums(const std::vector<float> &x, int groups, std::vector<float> &group_sums) {
  group_sums.assign(groups, 0.0f);
  for (int g = 0; g < groups; ++g) {
    float sum = 0.0f;
    for (int i = 0; i < 64; ++i) sum += x[static_cast<size_t>(g) * 64 + i];
    group_sums[static_cast<size_t>(g)] = sum;
  }
}

static void qlinear_direct(
    const std::vector<float> &x,
    const TensorSlice &w,
    const TensorSlice &s,
    const TensorSlice &b,
    int out_dim,
    int packed_words,
    int groups,
    std::vector<float> &out,
    std::vector<float> &group_sums) {
  compute_group_sums(x, groups, group_sums);
  out.assign(out_dim, 0.0f);
  for (int o = 0; o < out_dim; ++o) {
    const uint8_t *wrow = w.ptr + static_cast<size_t>(o) * static_cast<size_t>(packed_words) * 4;
    const uint8_t *srow = s.ptr + static_cast<size_t>(o) * static_cast<size_t>(groups) * 2;
    const uint8_t *brow = b.ptr + static_cast<size_t>(o) * static_cast<size_t>(groups) * 2;
    float acc = 0.0f;
    for (int g = 0; g < groups; ++g) {
      const float scale = bf16_to_float_local(load_u16_le_local(srow + static_cast<size_t>(g) * 2));
      const float bias = bf16_to_float_local(load_u16_le_local(brow + static_cast<size_t>(g) * 2));
      const float qx = qdot64(wrow + static_cast<size_t>(g) * 8 * 4, x.data() + static_cast<size_t>(g) * 64);
      acc += scale * qx + bias * group_sums[static_cast<size_t>(g)];
    }
    out[static_cast<size_t>(o)] = acc;
  }
}

static void qlinear_pair_swiglu_direct(
    const std::vector<float> &x,
    const TensorSlice &up_w,
    const TensorSlice &up_s,
    const TensorSlice &up_b,
    const TensorSlice &gate_w,
    const TensorSlice &gate_s,
    const TensorSlice &gate_b,
    std::vector<float> &hidden,
    std::vector<float> &group_sums) {
  constexpr int out_dim = INTER_DIM;
  constexpr int packed_words = 512;
  constexpr int groups = 64;
  compute_group_sums(x, groups, group_sums);
  hidden.assign(out_dim, 0.0f);
  for (int o = 0; o < out_dim; ++o) {
    const uint8_t *up_wrow = up_w.ptr + static_cast<size_t>(o) * packed_words * 4;
    const uint8_t *gate_wrow = gate_w.ptr + static_cast<size_t>(o) * packed_words * 4;
    const uint8_t *up_srow = up_s.ptr + static_cast<size_t>(o) * groups * 2;
    const uint8_t *up_brow = up_b.ptr + static_cast<size_t>(o) * groups * 2;
    const uint8_t *gate_srow = gate_s.ptr + static_cast<size_t>(o) * groups * 2;
    const uint8_t *gate_brow = gate_b.ptr + static_cast<size_t>(o) * groups * 2;
    float acc_up = 0.0f;
    float acc_gate = 0.0f;
    for (int g = 0; g < groups; ++g) {
      const float sum_x = group_sums[static_cast<size_t>(g)];
      const float up_scale = bf16_to_float_local(load_u16_le_local(up_srow + static_cast<size_t>(g) * 2));
      const float up_bias = bf16_to_float_local(load_u16_le_local(up_brow + static_cast<size_t>(g) * 2));
      const float gate_scale = bf16_to_float_local(load_u16_le_local(gate_srow + static_cast<size_t>(g) * 2));
      const float gate_bias = bf16_to_float_local(load_u16_le_local(gate_brow + static_cast<size_t>(g) * 2));
      const float *xg = x.data() + static_cast<size_t>(g) * 64;
      acc_up += up_scale * qdot64(up_wrow + static_cast<size_t>(g) * 8 * 4, xg) + up_bias * sum_x;
      acc_gate += gate_scale * qdot64(gate_wrow + static_cast<size_t>(g) * 8 * 4, xg) + gate_bias * sum_x;
    }
    hidden[static_cast<size_t>(o)] = silu(acc_gate) * acc_up;
  }
}

struct Fixture {
  int layer = 0;
  int topk = 0;
  int seq_len = 0;
  std::vector<int> experts_flat;
  std::vector<float> route_weights_flat;
  std::vector<float> hidden_flat;
  std::vector<float> expected_flat;
};

static std::string read_text_file(const fs::path &path) {
  FILE *f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("failed to open " + path.string());
  std::string out;
  char buf[1 << 16];
  while (true) {
    size_t n = std::fread(buf, 1, sizeof(buf), f);
    if (n) out.append(buf, n);
    if (n < sizeof(buf)) break;
  }
  std::fclose(f);
  return out;
}

static size_t find_json_key(const std::string &text, const std::string &key) {
  auto pos = text.find("\"" + key + "\"");
  if (pos == std::string::npos) throw std::runtime_error("fixture missing key: " + key);
  return pos;
}

static bool has_json_key(const std::string &text, const std::string &key) {
  return text.find("\"" + key + "\"") != std::string::npos;
}

static int parse_json_int(const std::string &text, const std::string &key) {
  auto pos = find_json_key(text, key);
  pos = text.find(':', pos);
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

static Fixture load_fixture(const fs::path &path) {
  std::string text = read_text_file(path);
  Fixture fx;
  fx.layer = parse_json_int(text, "layer");
  fx.topk = parse_json_int(text, "topk");
  fx.seq_len = has_json_key(text, "seq_len") ? parse_json_int(text, "seq_len") : 1;
  fx.experts_flat = has_json_key(text, "experts_flat") ? parse_json_int_array(text, "experts_flat") : parse_json_int_array(text, "experts");
  fx.route_weights_flat = has_json_key(text, "route_weights_flat") ? parse_json_float_array(text, "route_weights_flat") : parse_json_float_array(text, "route_weights");
  fx.hidden_flat = has_json_key(text, "hidden_tokens") ? parse_json_float_array(text, "hidden_tokens") : parse_json_float_array(text, "hidden");
  fx.expected_flat = has_json_key(text, "expected_routed_tokens") ? parse_json_float_array(text, "expected_routed_tokens") : parse_json_float_array(text, "expected_routed");
  return fx;
}

struct DenseExpertBankLocal {
  DenseQ4Affine up;
  DenseQ4Affine gate;
  DenseQ4Affine down;
};

static DenseExpertBankLocal dequantize_bank(const ExpertBank &bank) {
  DenseExpertBankLocal dense;
  qlinear_dequantize(bank.up_w, bank.up_s, bank.up_b, INTER_DIM, HIDDEN_DIM, 512, 64, dense.up);
  qlinear_dequantize(bank.gate_w, bank.gate_s, bank.gate_b, INTER_DIM, HIDDEN_DIM, 512, 64, dense.gate);
  qlinear_dequantize(bank.down_w, bank.down_s, bank.down_b, HIDDEN_DIM, INTER_DIM, 192, 24, dense.down);
  return dense;
}

static void dense_route(const DenseExpertBankLocal &bank, const std::vector<float> &x, std::vector<float> &up, std::vector<float> &gate, std::vector<float> &hidden, std::vector<float> &down) {
  qlinear_dense(x, bank.up, up);
  qlinear_dense(x, bank.gate, gate);
  hidden.resize(INTER_DIM);
  for (int i = 0; i < INTER_DIM; ++i) hidden[static_cast<size_t>(i)] = silu(gate[static_cast<size_t>(i)]) * up[static_cast<size_t>(i)];
  qlinear_dense(hidden, bank.down, down);
}

static void direct_route(const ExpertBank &bank, const std::vector<float> &x, std::vector<float> &hidden, std::vector<float> &down, std::vector<float> &group_sums) {
  qlinear_pair_swiglu_direct(x, bank.up_w, bank.up_s, bank.up_b, bank.gate_w, bank.gate_s, bank.gate_b, hidden, group_sums);
  qlinear_direct(hidden, bank.down_w, bank.down_s, bank.down_b, HIDDEN_DIM, 192, 24, down, group_sums);
}

struct ErrorStats {
  double max_abs = 0.0;
  double mean_abs = 0.0;
  double rmse = 0.0;
  double expected_max_abs = 0.0;
  double max_rel_to_expected = 0.0;
};

static ErrorStats compare_vectors(const std::vector<float> &actual, const std::vector<float> &expected) {
  if (actual.size() != expected.size()) throw std::runtime_error("compare size mismatch");
  ErrorStats st;
  double sum_abs = 0.0;
  double sum_sq = 0.0;
  for (size_t i = 0; i < actual.size(); ++i) {
    double diff = static_cast<double>(actual[i]) - static_cast<double>(expected[i]);
    double ad = std::fabs(diff);
    st.max_abs = std::max(st.max_abs, ad);
    st.expected_max_abs = std::max(st.expected_max_abs, std::fabs(static_cast<double>(expected[i])));
    sum_abs += ad;
    sum_sq += diff * diff;
  }
  st.mean_abs = sum_abs / static_cast<double>(actual.size());
  st.rmse = std::sqrt(sum_sq / static_cast<double>(actual.size()));
  st.max_rel_to_expected = st.max_abs / std::max(st.expected_max_abs, 1.0e-12);
  return st;
}

static std::vector<int> unique_experts_in_order(const Fixture &fx) {
  std::vector<int> out;
  std::unordered_set<int> seen;
  for (int expert : fx.experts_flat) {
    if (seen.insert(expert).second) out.push_back(expert);
  }
  return out;
}

using Clock = std::chrono::steady_clock;
static double elapsed_s(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

struct RunOutput {
  std::vector<float> actual;
  double load_s = 0.0;
  double dequant_s = 0.0;
  double compute_s = 0.0;
};

static RunOutput run_dense(const Fixture &fx, const std::vector<Entry> &entries, const std::unordered_map<uint64_t, Span> &spans, const fs::path &root) {
  FdCache fds(root);
  RunOutput out;
  out.actual.assign(static_cast<size_t>(fx.seq_len) * HIDDEN_DIM, 0.0f);
  auto unique = unique_experts_in_order(fx);
  std::unordered_map<int, ExpertBank> raw;
  std::unordered_map<int, DenseExpertBankLocal> dense;
  auto t0 = Clock::now();
  for (int expert : unique) raw.emplace(expert, load_expert_bank(entries, spans, fds, fx.layer, expert));
  auto t1 = Clock::now();
  for (int expert : unique) dense.emplace(expert, dequantize_bank(raw.at(expert)));
  auto t2 = Clock::now();
  std::vector<float> up, gate, hidden, down;
  std::vector<float> route_outputs(static_cast<size_t>(fx.seq_len) * fx.topk * HIDDEN_DIM, 0.0f);
  for (int expert : unique) {
    const auto &bank = dense.at(expert);
    for (int token = 0; token < fx.seq_len; ++token) {
      std::vector<float> x(fx.hidden_flat.begin() + static_cast<size_t>(token) * HIDDEN_DIM, fx.hidden_flat.begin() + static_cast<size_t>(token + 1) * HIDDEN_DIM);
      for (int k = 0; k < fx.topk; ++k) {
        size_t route = static_cast<size_t>(token * fx.topk + k);
        if (fx.experts_flat[route] != expert) continue;
        dense_route(bank, x, up, gate, hidden, down);
        std::copy(down.begin(), down.end(), route_outputs.begin() + route * HIDDEN_DIM);
      }
    }
  }
  for (int token = 0; token < fx.seq_len; ++token) {
    for (int k = 0; k < fx.topk; ++k) {
      size_t route = static_cast<size_t>(token * fx.topk + k);
      float weight = fx.route_weights_flat[route];
      for (int i = 0; i < HIDDEN_DIM; ++i) out.actual[static_cast<size_t>(token) * HIDDEN_DIM + i] += weight * route_outputs[route * HIDDEN_DIM + i];
    }
  }
  auto t3 = Clock::now();
  out.load_s = elapsed_s(t0, t1);
  out.dequant_s = elapsed_s(t1, t2);
  out.compute_s = elapsed_s(t2, t3);
  return out;
}

static RunOutput run_direct(const Fixture &fx, const std::vector<Entry> &entries, const std::unordered_map<uint64_t, Span> &spans, const fs::path &root) {
  FdCache fds(root);
  RunOutput out;
  out.actual.assign(static_cast<size_t>(fx.seq_len) * HIDDEN_DIM, 0.0f);
  auto unique = unique_experts_in_order(fx);
  std::unordered_map<int, ExpertBank> raw;
  auto t0 = Clock::now();
  for (int expert : unique) raw.emplace(expert, load_expert_bank(entries, spans, fds, fx.layer, expert));
  auto t1 = Clock::now();
  std::vector<float> hidden, down, group_sums;
  std::vector<float> route_outputs(static_cast<size_t>(fx.seq_len) * fx.topk * HIDDEN_DIM, 0.0f);
  for (int expert : unique) {
    const auto &bank = raw.at(expert);
    for (int token = 0; token < fx.seq_len; ++token) {
      std::vector<float> x(fx.hidden_flat.begin() + static_cast<size_t>(token) * HIDDEN_DIM, fx.hidden_flat.begin() + static_cast<size_t>(token + 1) * HIDDEN_DIM);
      for (int k = 0; k < fx.topk; ++k) {
        size_t route = static_cast<size_t>(token * fx.topk + k);
        if (fx.experts_flat[route] != expert) continue;
        direct_route(bank, x, hidden, down, group_sums);
        std::copy(down.begin(), down.end(), route_outputs.begin() + route * HIDDEN_DIM);
      }
    }
  }
  for (int token = 0; token < fx.seq_len; ++token) {
    for (int k = 0; k < fx.topk; ++k) {
      size_t route = static_cast<size_t>(token * fx.topk + k);
      float weight = fx.route_weights_flat[route];
      for (int i = 0; i < HIDDEN_DIM; ++i) out.actual[static_cast<size_t>(token) * HIDDEN_DIM + i] += weight * route_outputs[route * HIDDEN_DIM + i];
    }
  }
  auto t2 = Clock::now();
  out.load_s = elapsed_s(t0, t1);
  out.compute_s = elapsed_s(t1, t2);
  return out;
}

static std::vector<fs::path> load_fixture_args(int argc, char **argv, fs::path &index, fs::path &root) {
  std::vector<fs::path> fixtures;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    auto need = [&](const char *name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing ") + name);
      return argv[++i];
    };
    if (key == "--index") index = need("--index");
    else if (key == "--root") root = need("--root");
    else if (key == "--fixture") fixtures.emplace_back(need("--fixture"));
    else if (key == "--fixture-list") {
      fs::path list_path = need("--fixture-list");
      std::string text = read_text_file(list_path);
      std::stringstream ss(text);
      std::string line;
      while (std::getline(ss, line)) {
        while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) line.pop_back();
        size_t start = 0;
        while (start < line.size() && std::isspace(static_cast<unsigned char>(line[start]))) ++start;
        line = line.substr(start);
        if (line.empty() || line[0] == '#') continue;
        fs::path p(line);
        if (p.is_relative()) p = list_path.parent_path() / p;
        fixtures.push_back(p);
      }
    } else {
      throw std::runtime_error("unknown arg: " + key);
    }
  }
  if (index.empty()) throw std::runtime_error("--index required");
  if (root.empty()) throw std::runtime_error("--root required");
  if (fixtures.empty()) throw std::runtime_error("--fixture or --fixture-list required");
  return fixtures;
}

int main(int argc, char **argv) {
  try {
    fs::path index;
    fs::path root;
    auto fixtures = load_fixture_args(argc, argv, index, root);
    auto entries = load_entries(index);
    auto spans = build_spans(entries);
    std::cout << "{\n  \"schema\": \"hy3-cpp-q4-direct-dot-spike-v1\",\n";
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    std::cout << "  \"kernel\": \"direct_q4_group_affine_neon\",\n";
#else
    std::cout << "  \"kernel\": \"direct_q4_group_affine_scalar\",\n";
#endif
    std::cout << "  \"fixtures\": [\n";
    bool first = true;
    for (const auto &fixture_path : fixtures) {
      Fixture fx = load_fixture(fixture_path);
      RunOutput dense = run_dense(fx, entries, spans, root);
      RunOutput direct = run_direct(fx, entries, spans, root);
      ErrorStats dense_err = compare_vectors(dense.actual, fx.expected_flat);
      ErrorStats direct_err = compare_vectors(direct.actual, fx.expected_flat);
      ErrorStats direct_vs_dense = compare_vectors(direct.actual, dense.actual);
      const double dense_total = dense.dequant_s + dense.compute_s;
      const double direct_total = direct.compute_s;
      if (!first) std::cout << ",\n";
      first = false;
      std::cout << "    {\n";
      std::cout << "      \"fixture\": \"" << fixture_path.string() << "\",\n";
      std::cout << "      \"layer\": " << fx.layer << ", \"seq_len\": " << fx.seq_len << ", \"topk\": " << fx.topk << ",\n";
      std::cout << "      \"routes\": " << (fx.seq_len * fx.topk) << ", \"unique_experts\": " << unique_experts_in_order(fx).size() << ",\n";
      std::cout << "      \"dense\": {\"load_s\": " << dense.load_s << ", \"dequant_s\": " << dense.dequant_s << ", \"compute_s\": " << dense.compute_s << ", \"total_excluding_load_s\": " << dense_total << ", \"max_rel_to_expected\": " << dense_err.max_rel_to_expected << "},\n";
      std::cout << "      \"direct\": {\"load_s\": " << direct.load_s << ", \"compute_s\": " << direct.compute_s << ", \"total_excluding_load_s\": " << direct_total << ", \"max_rel_to_expected\": " << direct_err.max_rel_to_expected << "},\n";
      std::cout << "      \"direct_vs_dense\": {\"max_rel_to_dense\": " << direct_vs_dense.max_rel_to_expected << ", \"mean_abs\": " << direct_vs_dense.mean_abs << "},\n";
      std::cout << "      \"direct_speedup_vs_dense_total\": " << (dense_total / std::max(direct_total, 1.0e-12)) << ",\n";
      std::cout << "      \"direct_speedup_vs_dense_compute_only\": " << (dense.compute_s / std::max(direct.compute_s, 1.0e-12)) << "\n";
      std::cout << "    }";
    }
    std::cout << "\n  ]\n}\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
