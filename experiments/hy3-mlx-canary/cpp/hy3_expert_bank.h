#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

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

class FdCache {
 public:
  explicit FdCache(fs::path root);
  ~FdCache();
  FdCache(const FdCache &) = delete;
  FdCache &operator=(const FdCache &) = delete;
  int get(const std::string &rel);

 private:
  fs::path root_;
  std::unordered_map<std::string, int> fds_;
};

uint64_t span_key(int layer, int expert);
std::vector<Entry> load_entries(const fs::path &index_path);
std::unordered_map<uint64_t, Span> build_spans(const std::vector<Entry> &entries);
void read_exact(int fd, uint64_t offset, std::vector<uint8_t> &buf);
ExpertBank load_expert_bank(
    const std::vector<Entry> &entries,
    const std::unordered_map<uint64_t, Span> &spans,
    FdCache &fds,
    int layer,
    int expert);
