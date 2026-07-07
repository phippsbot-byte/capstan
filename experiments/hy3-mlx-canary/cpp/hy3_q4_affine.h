#pragma once

#include "hy3_expert_bank.h"

#include <vector>

float silu(float x);
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
