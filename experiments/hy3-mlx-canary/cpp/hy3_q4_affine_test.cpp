#include "hy3_q4_affine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

uint16_t float_to_bf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t rounding_bias = 0x7FFFu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

void store_u16_le(std::vector<uint8_t> &dst, size_t offset, uint16_t value) {
  std::memcpy(dst.data() + offset, &value, sizeof(value));
}

void store_u32_le(std::vector<uint8_t> &dst, size_t offset, uint32_t value) {
  std::memcpy(dst.data() + offset, &value, sizeof(value));
}

struct QuantMatrix {
  int out_dim;
  int in_dim;
  int packed_words;
  int groups;
  std::vector<uint8_t> weights;
  std::vector<uint8_t> scales;
  std::vector<uint8_t> biases;

  QuantMatrix(int out, int in, int seed)
      : out_dim(out),
        in_dim(in),
        packed_words((in + 7) / 8),
        groups((in + 63) / 64),
        weights(static_cast<size_t>(out_dim * packed_words) * 4, 0),
        scales(static_cast<size_t>(out_dim * groups) * 2, 0),
        biases(static_cast<size_t>(out_dim * groups) * 2, 0) {
    if (in_dim % 64 != 0) throw std::runtime_error("test matrix input width must be a multiple of 64");
    for (int o = 0; o < out_dim; ++o) {
      for (int word_index = 0; word_index < packed_words; ++word_index) {
        uint32_t packed = 0;
        for (int nibble = 0; nibble < 8; ++nibble) {
          const int i = word_index * 8 + nibble;
          const uint32_t q = static_cast<uint32_t>((seed + o * 7 + i * 5) & 0xF);
          packed |= q << (nibble * 4);
        }
        const size_t offset = static_cast<size_t>(o * packed_words + word_index) * 4;
        store_u32_le(weights, offset, packed);
      }
      for (int g = 0; g < groups; ++g) {
        const float scale = 0.03125f * static_cast<float>(1 + ((seed + o + g) % 4));
        const float bias = 0.0625f * static_cast<float>(((seed + o * 3 + g) % 5) - 2);
        const size_t offset = static_cast<size_t>(o * groups + g) * 2;
        store_u16_le(scales, offset, float_to_bf16(scale));
        store_u16_le(biases, offset, float_to_bf16(bias));
      }
    }
  }

  TensorSlice weight_slice() const { return TensorSlice{0, weights.size(), weights.data()}; }
  TensorSlice scale_slice() const { return TensorSlice{0, scales.size(), scales.data()}; }
  TensorSlice bias_slice() const { return TensorSlice{0, biases.size(), biases.data()}; }
};

void require_close(
    const std::vector<float> &actual,
    const std::vector<float> &expected,
    float abs_tolerance,
    float rel_tolerance,
    const std::string &label) {
  if (actual.size() != expected.size()) throw std::runtime_error(label + ": size mismatch");
  for (size_t i = 0; i < actual.size(); ++i) {
    const float diff = std::fabs(actual[i] - expected[i]);
    const float limit = abs_tolerance + rel_tolerance * std::fabs(expected[i]);
    if (!std::isfinite(actual[i]) || diff > limit) {
      throw std::runtime_error(
          label + ": mismatch at " + std::to_string(i) +
          " actual=" + std::to_string(actual[i]) +
          " expected=" + std::to_string(expected[i]) +
          " diff=" + std::to_string(diff) +
          " limit=" + std::to_string(limit));
    }
  }
}

void test_direct_and_dense_match_reference() {
  QuantMatrix matrix(11, 128, 3);
  std::vector<float> x(static_cast<size_t>(matrix.in_dim));
  for (int i = 0; i < matrix.in_dim; ++i) {
    x[static_cast<size_t>(i)] = static_cast<float>((i * 11) % 29 - 14) / 32.0f;
  }

  std::vector<float> reference;
  std::vector<float> direct;
  std::vector<float> dense_out;
  std::vector<float> group_sums;
  DenseQ4Affine dense;

  qlinear_reference(
      x, matrix.weight_slice(), matrix.scale_slice(), matrix.bias_slice(),
      matrix.out_dim, matrix.in_dim, matrix.packed_words, matrix.groups, reference);
  qlinear_direct(
      x, matrix.weight_slice(), matrix.scale_slice(), matrix.bias_slice(),
      matrix.out_dim, matrix.in_dim, matrix.packed_words, matrix.groups, direct, group_sums);
  qlinear_dequantize(
      matrix.weight_slice(), matrix.scale_slice(), matrix.bias_slice(),
      matrix.out_dim, matrix.in_dim, matrix.packed_words, matrix.groups, dense);
  qlinear_dense(x, dense, dense_out);

  require_close(direct, reference, 2.0e-5f, 2.0e-6f, "direct vs reference");
  require_close(dense_out, reference, 2.0e-5f, 2.0e-6f, "dense vs reference");
}

void test_pair_swiglu_matches_separate_reference() {
  QuantMatrix up(13, 128, 5);
  QuantMatrix gate(13, 128, 17);
  std::vector<float> x(static_cast<size_t>(up.in_dim));
  for (int i = 0; i < up.in_dim; ++i) {
    x[static_cast<size_t>(i)] = static_cast<float>((i * 13) % 31 - 15) / 64.0f;
  }

  std::vector<float> up_reference;
  std::vector<float> gate_reference;
  std::vector<float> expected(static_cast<size_t>(up.out_dim));
  std::vector<float> actual;
  std::vector<float> group_sums;

  qlinear_reference(
      x, up.weight_slice(), up.scale_slice(), up.bias_slice(),
      up.out_dim, up.in_dim, up.packed_words, up.groups, up_reference);
  qlinear_reference(
      x, gate.weight_slice(), gate.scale_slice(), gate.bias_slice(),
      gate.out_dim, gate.in_dim, gate.packed_words, gate.groups, gate_reference);
  for (int i = 0; i < up.out_dim; ++i) {
    expected[static_cast<size_t>(i)] = silu(gate_reference[static_cast<size_t>(i)]) * up_reference[static_cast<size_t>(i)];
  }

  qlinear_pair_swiglu_direct(
      x,
      up.weight_slice(), up.scale_slice(), up.bias_slice(),
      gate.weight_slice(), gate.scale_slice(), gate.bias_slice(),
      up.out_dim, up.in_dim, up.packed_words, up.groups, actual, group_sums);

  require_close(actual, expected, 3.0e-4f, 4.0e-6f, "paired swiglu vs reference");
}

void test_short_input_is_rejected() {
  QuantMatrix matrix(2, 64, 1);
  std::vector<float> short_x(63, 0.0f);
  std::vector<float> out;
  std::vector<float> group_sums;
  bool threw = false;
  try {
    qlinear_direct(
        short_x, matrix.weight_slice(), matrix.scale_slice(), matrix.bias_slice(),
        matrix.out_dim, matrix.in_dim, matrix.packed_words, matrix.groups, out, group_sums);
  } catch (const std::runtime_error &) {
    threw = true;
  }
  if (!threw) throw std::runtime_error("short qlinear_direct input was accepted");
}

}  // namespace

int main() {
  try {
    test_direct_and_dense_match_reference();
    test_pair_swiglu_matches_separate_reference();
    test_short_input_is_rejected();
    std::cout << "hy3_q4_affine_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "hy3_q4_affine_test: FAIL: " << error.what() << "\n";
    return 1;
  }
}
