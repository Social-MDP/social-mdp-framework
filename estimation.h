#pragma once

#include <array>
#include <memory>

#include "smdp.h"

template <int N> std::array<float, N> generateUniform() {
    std::array<float, N> ret;
    std::fill(ret.begin(), ret.end(), 1.0f / N);
    return ret;
}

template <int N> struct MultiPolicyEstimator {
    std::array<float, N> probs;

    MultiPolicyEstimator(const State &initial) : probs{generateUniform<N>()} {}

    float prob(int goal) const { return probs[goal]; }

    template <typename Iter> void update(const State &s1, const State &s2, Iter begin, Iter end) {
        assert(end - begin == N);
        auto sum = 0.0f;
        auto temp = begin;
        for (auto i = 0; i < N; i++) {
            probs[i] *= (*temp++)->prob(s1, s2);
            sum += probs[i];
        }
        for (auto i = 0; i < N; i++)
            probs[i] /= sum;
    }
};