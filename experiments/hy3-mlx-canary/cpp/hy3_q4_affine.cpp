#include "hy3_q4_affine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#endif

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

void qlinear_reference(const std::vector<float> &x, const TensorSlice &w, const TensorSlice &s, const TensorSlice &b, int out_dim, int in_dim, int packed_words, int groups, std::vector<float> &out) {
  out.assign(out_dim, 0.0f);
  for (int o = 0; o < out_dim; ++o) {
    float acc = 0.0f;
    for (int i = 0; i < in_dim; ++i) acc += x[static_cast<size_t>(i)] * dequant_q4_affine(w, s, b, o, i, packed_words, groups);
    out[static_cast<size_t>(o)] = acc;
  }
}

#ifdef __APPLE__
static void dequant_q4_affine_dense(const TensorSlice &w, const TensorSlice &s, const TensorSlice &b, int out_dim, int in_dim, int packed_words, int groups, std::vector<float> &dense) {
  if (in_dim > packed_words * 8) throw std::runtime_error("qlinear packed weight width too small");
  if (groups < ((in_dim + 63) / 64)) throw std::runtime_error("qlinear scale group count too small");
  dense.assign(static_cast<size_t>(out_dim) * static_cast<size_t>(in_dim), 0.0f);
  for (int o = 0; o < out_dim; ++o) {
    const uint8_t *wrow = w.ptr + static_cast<size_t>(o) * static_cast<size_t>(packed_words) * 4;
    const uint8_t *srow = s.ptr + static_cast<size_t>(o) * static_cast<size_t>(groups) * 2;
    const uint8_t *brow = b.ptr + static_cast<size_t>(o) * static_cast<size_t>(groups) * 2;
    float *dense_row = dense.data() + static_cast<size_t>(o) * static_cast<size_t>(in_dim);
    for (int g = 0; g < groups; ++g) {
      const int group_start = g * 64;
      if (group_start >= in_dim) break;
      const int group_end = std::min(group_start + 64, in_dim);
      const float scale = bf16_to_float(load_u16_le(srow + static_cast<size_t>(g) * 2));
      const float bias = bf16_to_float(load_u16_le(brow + static_cast<size_t>(g) * 2));
      for (int i = group_start; i < group_end; ++i) {
        const uint32_t word = load_u32_le(wrow + static_cast<size_t>(i >> 3) * 4);
        const uint32_t q = (word >> ((i & 7) * 4)) & 0xFu;
        dense_row[i] = static_cast<float>(q) * scale + bias;
      }
    }
  }
}
#endif

void qlinear(const std::vector<float> &x, const TensorSlice &w, const TensorSlice &s, const TensorSlice &b, int out_dim, int in_dim, int packed_words, int groups, std::vector<float> &out) {
#ifdef __APPLE__
  thread_local std::vector<float> dense;
  dequant_q4_affine_dense(w, s, b, out_dim, in_dim, packed_words, groups, dense);
  out.assign(out_dim, 0.0f);
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
  cblas_sgemv(CblasRowMajor, CblasNoTrans, out_dim, in_dim, 1.0f, dense.data(), in_dim, x.data(), 1, 0.0f, out.data(), 1);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#else
  qlinear_reference(x, w, s, b, out_dim, in_dim, packed_words, groups, out);
#endif
}

float silu(float x) {
  return x / (1.0f + std::exp(-x));
}
