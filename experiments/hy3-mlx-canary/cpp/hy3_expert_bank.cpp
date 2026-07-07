#include "hy3_expert_bank.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#include <utility>

static std::vector<std::string> split_local(const std::string &s, char delim) {
  std::vector<std::string> out;
  std::string item;
  std::stringstream ss(s);
  while (std::getline(ss, item, delim)) out.push_back(item);
  return out;
}

FdCache::FdCache(fs::path root) : root_(std::move(root)) {}

FdCache::~FdCache() {
  for (auto &[_, fd] : fds_) {
    if (fd >= 0) ::close(fd);
  }
}

int FdCache::get(const std::string &rel) {
  auto it = fds_.find(rel);
  if (it != fds_.end()) return it->second;
  fs::path p = root_ / rel;
  int fd = ::open(p.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("open failed for " + p.string() + ": " + std::strerror(errno));
  fds_[rel] = fd;
  return fd;
}

uint64_t span_key(int layer, int expert) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32) |
         static_cast<uint32_t>(expert);
}

std::vector<Entry> load_entries(const fs::path &index_path) {
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
    auto cols = split_local(s, '\t');
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

std::unordered_map<uint64_t, Span> build_spans(const std::vector<Entry> &entries) {
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

void read_exact(int fd, uint64_t offset, std::vector<uint8_t> &buf) {
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

ExpertBank load_expert_bank(const std::vector<Entry> &entries, const std::unordered_map<uint64_t, Span> &spans, FdCache &fds, int layer, int expert) {
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
