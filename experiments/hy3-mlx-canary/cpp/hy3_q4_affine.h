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
