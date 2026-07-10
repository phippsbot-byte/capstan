#include "hy3_expert_bank.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

ExpertBank make_bank(int layer, int expert, size_t bytes, uint8_t seed) {
  ExpertBank bank;
  bank.layer = layer;
  bank.expert = expert;
  bank.raw.resize(bytes);
  for (size_t i = 0; i < bytes; ++i) bank.raw[i] = static_cast<uint8_t>(seed + i);

  std::vector<TensorSlice *> slices = {
      &bank.up_w, &bank.up_s, &bank.up_b,
      &bank.gate_w, &bank.gate_s, &bank.gate_b,
      &bank.down_w, &bank.down_s, &bank.down_b,
  };
  for (size_t i = 0; i < slices.size(); ++i) {
    slices[i]->offset = i * 4;
    slices[i]->nbytes = 4;
    slices[i]->ptr = bank.raw.data() + slices[i]->offset;
  }
  return bank;
}

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

void require_slices_bound_to_raw(const ExpertBank &bank) {
  std::vector<const TensorSlice *> slices = {
      &bank.up_w, &bank.up_s, &bank.up_b,
      &bank.gate_w, &bank.gate_s, &bank.gate_b,
      &bank.down_w, &bank.down_s, &bank.down_b,
  };
  for (const TensorSlice *slice : slices) {
    require(slice->ptr == bank.raw.data() + slice->offset, "slice pointer was not rebound to cached raw storage");
    require(slice->offset + slice->nbytes <= bank.raw.size(), "slice exceeds cached raw storage");
  }
}

void test_move_rebinds_slices() {
  ExpertBank source = make_bank(1, 7, 96, 11);
  const uint8_t expected = source.raw[source.down_b.offset];
  ExpertBank moved = std::move(source);
  require_slices_bound_to_raw(moved);
  require(*moved.down_b.ptr == expected, "moved slice data changed");

  ExpertBank replacement = make_bank(2, 8, 128, 23);
  const uint8_t replacement_expected = replacement.raw[replacement.gate_w.offset];
  moved = std::move(replacement);
  require_slices_bound_to_raw(moved);
  require(*moved.gate_w.ptr == replacement_expected, "move-assigned slice data changed");

  ExpertBank empty;
  ExpertBank moved_empty = std::move(empty);
  require(moved_empty.raw.empty(), "empty bank move created raw storage");
  require(moved_empty.up_w.ptr == nullptr, "empty bank move produced a non-null slice");
}

void test_cache_is_byte_bounded_and_lru() {
  PackedExpertCache cache(192);
  ExpertBank first_bank = make_bank(1, 1, 96, 1);
  const ExpertBank *first = cache.insert(std::move(first_bank));
  require(first != nullptr, "cache rejected a bank within budget");
  require_slices_bound_to_raw(*first);
  ExpertBank second_bank = make_bank(1, 2, 96, 2);
  require(cache.insert(std::move(second_bank)) != nullptr, "cache rejected second bank within budget");
  require(cache.entries() == 2, "cache did not retain two banks within budget");
  require(cache.bytes() == 192, "cache byte accounting is wrong");

  require(cache.find(1, 1) != nullptr, "expected first bank cache hit");
  ExpertBank third_bank = make_bank(1, 3, 96, 3);
  require(cache.insert(std::move(third_bank)) != nullptr, "cache rejected replacement bank");

  require(cache.entries() == 2, "cache exceeded entry budget after eviction");
  require(cache.bytes() <= cache.max_bytes(), "cache exceeded byte budget");
  require(cache.find(1, 2) == nullptr, "least-recently-used bank was not evicted");
  require(cache.find(1, 1) != nullptr, "recently-used bank was incorrectly evicted");
  const ExpertBank *third = cache.find(1, 3);
  require(third != nullptr, "new bank missing after insertion");
  require_slices_bound_to_raw(*third);
  require(cache.evictions() == 1, "eviction counter mismatch");
  require(cache.hits() == 3, "hit counter mismatch");
  require(cache.misses() == 1, "miss counter mismatch");
}

void test_duplicate_key_replaces_accounting() {
  PackedExpertCache cache(192);
  ExpertBank original = make_bank(4, 6, 96, 7);
  require(cache.insert(std::move(original)) != nullptr, "cache rejected original bank");
  ExpertBank replacement = make_bank(4, 6, 64, 19);
  require(cache.insert(std::move(replacement)) != nullptr, "cache rejected duplicate replacement");
  require(cache.entries() == 1, "duplicate key created a second cache entry");
  require(cache.bytes() == 64, "duplicate key replacement corrupted byte accounting");
}

void test_disabled_and_oversized_entries_stay_caller_owned() {
  PackedExpertCache disabled(0);
  ExpertBank uncached = make_bank(2, 4, 96, 9);
  require(disabled.insert(std::move(uncached)) == nullptr, "disabled cache accepted an entry");
  require_slices_bound_to_raw(uncached);
  require(disabled.entries() == 0, "disabled cache retained an entry");
  require(disabled.bytes() == 0, "disabled cache reported resident bytes");
  require(disabled.find(2, 4) == nullptr, "disabled cache returned a hit");

  PackedExpertCache too_small(64);
  ExpertBank oversized = make_bank(3, 5, 96, 13);
  require(too_small.insert(std::move(oversized)) == nullptr, "undersized cache accepted an oversized bank");
  require_slices_bound_to_raw(oversized);
  require(too_small.entries() == 0, "oversized bank was retained");
  require(too_small.bytes() == 0, "oversized bank counted against resident bytes");
}

}  // namespace

int main() {
  try {
    test_move_rebinds_slices();
    test_cache_is_byte_bounded_and_lru();
    test_duplicate_key_replaces_accounting();
    test_disabled_and_oversized_entries_stay_caller_owned();
    std::cout << "hy3_expert_cache_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "hy3_expert_cache_test: FAIL: " << error.what() << "\n";
    return 1;
  }
}
