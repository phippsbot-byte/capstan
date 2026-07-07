#pragma once

#include "hy3_expert_bank.h"

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

struct ParityFixture {
  int layer = 0;
  int topk = 0;
  int seq_len = 1;
  std::vector<int> experts_flat;
  std::vector<float> route_weights_flat;
  std::vector<float> hidden_flat;
  std::vector<float> expected_flat;
};

struct ErrorStats {
  double max_abs = 0.0;
  double mean_abs = 0.0;
  double rmse = 0.0;
  double expected_max_abs = 0.0;
  double max_rel_to_expected = 0.0;
  int max_index = 0;
};

struct ParityResult {
  fs::path fixture;
  int layer = 0;
  int topk = 0;
  int seq_len = 1;
  bool layer_major = false;
  int read_calls = 0;
  int naive_read_calls = 0;
  int unique_expert_spans = 0;
  int dedup_saved_reads = 0;
  uint64_t bytes_read = 0;
  uint64_t naive_bytes_read = 0;
  uint64_t dedup_saved_bytes = 0;
  double compute_elapsed_s = 0.0;
  ErrorStats error;
};

ParityFixture load_parity_fixture(const fs::path &path);
std::vector<fs::path> load_fixture_list(const fs::path &list_path);
struct RoutedComputeResult {
  std::vector<float> actual;
  int read_calls = 0;
  int naive_read_calls = 0;
  int unique_expert_spans = 0;
  int dedup_saved_reads = 0;
  uint64_t bytes_read = 0;
  uint64_t naive_bytes_read = 0;
  uint64_t dedup_saved_bytes = 0;
};

RoutedComputeResult compute_routed_fixture(
    const ParityFixture &fx,
    const std::vector<Entry> &entries,
    const std::unordered_map<uint64_t, Span> &spans,
    const fs::path &root,
    bool layer_major = false);
ErrorStats compare_vectors(const std::vector<float> &actual, const std::vector<float> &expected);
bool parity_passes(const ErrorStats &stats);
ParityResult run_parity_fixture(
    const fs::path &fixture_path,
    const std::vector<Entry> &entries,
    const std::unordered_map<uint64_t, Span> &spans,
    const fs::path &root,
    bool layer_major = false);
