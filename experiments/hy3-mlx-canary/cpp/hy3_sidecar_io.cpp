#include <algorithm>
#include <chrono>
#include <cstdint>
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
  std::optional<int> layer;
  std::vector<int> explicit_layers;
  std::vector<int> experts;
  int topk = 0;
  int iterations = 1;
  bool no_read = false;
  std::string checksum = "sample";
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
               "(--layer N --experts a,b,c | --layers A-B --topk K) "
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
    else if (key == "--layer") args.layer = std::stoi(need_value("--layer"));
    else if (key == "--layers") args.explicit_layers = parse_layers(need_value("--layers"));
    else if (key == "--experts") args.experts = parse_csv_ints(need_value("--experts"));
    else if (key == "--topk") args.topk = std::stoi(need_value("--topk"));
    else if (key == "--iterations") args.iterations = std::stoi(need_value("--iterations"));
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
    auto index_elapsed = std::chrono::duration<double>(t_index1 - t_index0).count();
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
