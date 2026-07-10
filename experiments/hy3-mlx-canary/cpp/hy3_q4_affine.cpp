#include "hy3_q4_affine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
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
    const uint32_t word = load_u32_le(words + static_cast<size_t>(j) * 4);
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
    uint32_t word = load_u32_le(words + static_cast<size_t>(i >> 3) * 4);
    uint32_t q = (word >> ((i & 7) * 4)) & 0xFu;
    acc += static_cast<float>(q) * x[i];
  }
  return acc;
#endif
}

static void compute_group_sums(const std::vector<float> &x, int in_dim, int groups, std::vector<float> &group_sums) {
  if (static_cast<int>(x.size()) < in_dim) throw std::runtime_error("qlinear_direct input too short");
  if (groups < ((in_dim + 63) / 64)) throw std::runtime_error("qlinear_direct group count too small");
  group_sums.assign(groups, 0.0f);
  for (int g = 0; g < groups; ++g) {
    const int start = g * 64;
    if (start >= in_dim) break;
    const int end = std::min(start + 64, in_dim);
    float sum = 0.0f;
    for (int i = start; i < end; ++i) sum += x[static_cast<size_t>(i)];
    group_sums[static_cast<size_t>(g)] = sum;
  }
}

void qlinear_direct(
    const std::vector<float> &x,
    const TensorSlice &w,
    const TensorSlice &s,
    const TensorSlice &b,
    int out_dim,
    int in_dim,
    int packed_words,
    int groups,
    std::vector<float> &out,
    std::vector<float> &group_sums) {
  if (in_dim > packed_words * 8) throw std::runtime_error("qlinear_direct packed weight width too small");
  compute_group_sums(x, in_dim, groups, group_sums);
  out.assign(out_dim, 0.0f);
  for (int o = 0; o < out_dim; ++o) {
    const uint8_t *wrow = w.ptr + static_cast<size_t>(o) * static_cast<size_t>(packed_words) * 4;
    const uint8_t *srow = s.ptr + static_cast<size_t>(o) * static_cast<size_t>(groups) * 2;
    const uint8_t *brow = b.ptr + static_cast<size_t>(o) * static_cast<size_t>(groups) * 2;
    float acc = 0.0f;
    for (int g = 0; g < groups; ++g) {
      const int group_start = g * 64;
      if (group_start >= in_dim) break;
      const float scale = bf16_to_float(load_u16_le(srow + static_cast<size_t>(g) * 2));
      const float bias = bf16_to_float(load_u16_le(brow + static_cast<size_t>(g) * 2));
      const float qx = qdot64(wrow + static_cast<size_t>(g) * 8 * 4, x.data() + static_cast<size_t>(group_start));
      acc += scale * qx + bias * group_sums[static_cast<size_t>(g)];
    }
    out[static_cast<size_t>(o)] = acc;
  }
}

void qlinear_pair_swiglu_direct(
    const std::vector<float> &x,
    const TensorSlice &up_w,
    const TensorSlice &up_s,
    const TensorSlice &up_b,
    const TensorSlice &gate_w,
    const TensorSlice &gate_s,
    const TensorSlice &gate_b,
    int out_dim,
    int in_dim,
    int packed_words,
    int groups,
    std::vector<float> &hidden,
    std::vector<float> &group_sums) {
  if (in_dim > packed_words * 8) throw std::runtime_error("qlinear_pair_swiglu_direct packed weight width too small");
  compute_group_sums(x, in_dim, groups, group_sums);
  hidden.assign(out_dim, 0.0f);
  for (int o = 0; o < out_dim; ++o) {
    const uint8_t *up_wrow = up_w.ptr + static_cast<size_t>(o) * static_cast<size_t>(packed_words) * 4;
    const uint8_t *gate_wrow = gate_w.ptr + static_cast<size_t>(o) * static_cast<size_t>(packed_words) * 4;
    const uint8_t *up_srow = up_s.ptr + static_cast<size_t>(o) * static_cast<size_t>(groups) * 2;
    const uint8_t *up_brow = up_b.ptr + static_cast<size_t>(o) * static_cast<size_t>(groups) * 2;
    const uint8_t *gate_srow = gate_s.ptr + static_cast<size_t>(o) * static_cast<size_t>(groups) * 2;
    const uint8_t *gate_brow = gate_b.ptr + static_cast<size_t>(o) * static_cast<size_t>(groups) * 2;
    float acc_up = 0.0f;
    float acc_gate = 0.0f;
    for (int g = 0; g < groups; ++g) {
      const int group_start = g * 64;
      if (group_start >= in_dim) break;
      const float sum_x = group_sums[static_cast<size_t>(g)];
      const float *xg = x.data() + static_cast<size_t>(group_start);
      const float up_scale = bf16_to_float(load_u16_le(up_srow + static_cast<size_t>(g) * 2));
      const float up_bias = bf16_to_float(load_u16_le(up_brow + static_cast<size_t>(g) * 2));
      const float gate_scale = bf16_to_float(load_u16_le(gate_srow + static_cast<size_t>(g) * 2));
      const float gate_bias = bf16_to_float(load_u16_le(gate_brow + static_cast<size_t>(g) * 2));
      acc_up += up_scale * qdot64(up_wrow + static_cast<size_t>(g) * 8 * 4, xg) + up_bias * sum_x;
      acc_gate += gate_scale * qdot64(gate_wrow + static_cast<size_t>(g) * 8 * 4, xg) + gate_bias * sum_x;
    }
    hidden[static_cast<size_t>(o)] = silu(acc_gate) * acc_up;
  }
}

void qlinear_reference(const std::vector<float> &x, const TensorSlice &w, const TensorSlice &s, const TensorSlice &b, int out_dim, int in_dim, int packed_words, int groups, std::vector<float> &out) {
  out.assign(out_dim, 0.0f);
  for (int o = 0; o < out_dim; ++o) {
    float acc = 0.0f;
    for (int i = 0; i < in_dim; ++i) acc += x[static_cast<size_t>(i)] * dequant_q4_affine(w, s, b, o, i, packed_words, groups);
    out[static_cast<size_t>(o)] = acc;
  }
}

void qlinear_dequantize(const TensorSlice &w, const TensorSlice &s, const TensorSlice &b, int out_dim, int in_dim, int packed_words, int groups, DenseQ4Affine &dense) {
  if (in_dim > packed_words * 8) throw std::runtime_error("qlinear packed weight width too small");
  if (groups < ((in_dim + 63) / 64)) throw std::runtime_error("qlinear scale group count too small");
  dense.out_dim = out_dim;
  dense.in_dim = in_dim;
  dense.weights.resize(static_cast<size_t>(out_dim) * static_cast<size_t>(in_dim));
  for (int o = 0; o < out_dim; ++o) {
    const uint8_t *wrow = w.ptr + static_cast<size_t>(o) * static_cast<size_t>(packed_words) * 4;
    const uint8_t *srow = s.ptr + static_cast<size_t>(o) * static_cast<size_t>(groups) * 2;
    const uint8_t *brow = b.ptr + static_cast<size_t>(o) * static_cast<size_t>(groups) * 2;
    float *dense_row = dense.weights.data() + static_cast<size_t>(o) * static_cast<size_t>(in_dim);
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

void qlinear_dense(const std::vector<float> &x, const DenseQ4Affine &dense, std::vector<float> &out) {
  if (static_cast<int>(x.size()) < dense.in_dim) throw std::runtime_error("qlinear_dense input too short");
  out.assign(dense.out_dim, 0.0f);
#ifdef __APPLE__
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
  cblas_sgemv(CblasRowMajor, CblasNoTrans, dense.out_dim, dense.in_dim, 1.0f, dense.weights.data(), dense.in_dim, x.data(), 1, 0.0f, out.data(), 1);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#else
  for (int o = 0; o < dense.out_dim; ++o) {
    const float *row = dense.weights.data() + static_cast<size_t>(o) * static_cast<size_t>(dense.in_dim);
    float acc = 0.0f;
    for (int i = 0; i < dense.in_dim; ++i) acc += x[static_cast<size_t>(i)] * row[i];
    out[static_cast<size_t>(o)] = acc;
  }
#endif
}

void qlinear(const std::vector<float> &x, const TensorSlice &w, const TensorSlice &s, const TensorSlice &b, int out_dim, int in_dim, int packed_words, int groups, std::vector<float> &out) {
  thread_local DenseQ4Affine dense;
  qlinear_dequantize(w, s, b, out_dim, in_dim, packed_words, groups, dense);
  qlinear_dense(x, dense, out);
}

float silu(float x) {
  return x / (1.0f + std::exp(-x));
}
