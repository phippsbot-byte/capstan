#pragma once

#include "hy3_expert_bank.h"

#include <vector>

struct DenseQ4Affine {
  int out_dim = 0;
  int in_dim = 0;
  std::vector<float> weights;
};

float silu(float x);
void qlinear_dequantize(
    const TensorSlice &w,
    const TensorSlice &s,
    const TensorSlice &b,
    int out_dim,
    int in_dim,
    int packed_words,
    int groups,
    DenseQ4Affine &dense);
void qlinear_dense(
    const std::vector<float> &x,
    const DenseQ4Affine &dense,
    std::vector<float> &out);
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
    std::vector<float> &group_sums);
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
    std::vector<float> &group_sums);
void qlinear_reference(
    const std::vector<float> &x,
    const TensorSlice &w,
    const TensorSlice &s,
    const TensorSlice &b,
    int out_dim,
    int in_dim,
    int packed_words,
    int groups,
    std::vector<float> &out);
void qlinear(
    const std::vector<float> &x,
    const TensorSlice &w,
    const TensorSlice &s,
    const TensorSlice &b,
    int out_dim,
    int in_dim,
    int packed_words,
    int groups,
    std::vector<float> &out);
