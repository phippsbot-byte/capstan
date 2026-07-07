#include "hy3_q4_affine.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

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

void qlinear(const std::vector<float> &x, const TensorSlice &w, const TensorSlice &s, const TensorSlice &b, int out_dim, int in_dim, int packed_words, int groups, std::vector<float> &out) {
  out.assign(out_dim, 0.0f);
  for (int o = 0; o < out_dim; ++o) {
    float acc = 0.0f;
    for (int i = 0; i < in_dim; ++i) acc += x[static_cast<size_t>(i)] * dequant_q4_affine(w, s, b, o, i, packed_words, groups);
    out[static_cast<size_t>(o)] = acc;
  }
}

float silu(float x) {
  return x / (1.0f + std::exp(-x));
}
