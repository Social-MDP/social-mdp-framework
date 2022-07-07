/**
 * @file smdp.cu
 * @author Chengyuan Ma
 * @brief GPU-accelerated MDP solver with value iteration.
 */

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <memory>
#include <random>
#include <tuple>

#include <cub/cub.cuh>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include "progress.h"
#include "smdp.h"

using namespace std::chrono;

inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fmt::print("GPU error at {}:{}: {}.\n", file, line, cudaGetErrorString(code));
        cudaDeviceReset();
        exit(code);
    }
}

#define gpuCheck(expr) gpuAssert((expr), __FILE__, __LINE__)

__host__ __device__ State decodeState(const Context *ctx, IntState encoded) {
    State ret;
    for (auto i = N_COMPONENTS - 1; i >= 0; i--) {
        ret[i] = encoded % ctx->nSide;
        encoded /= ctx->nSide;
    }
    return ret;
}

__host__ __device__ IntState encodeState(const Context *ctx, const State &decoded) {
    IntState ret = 0;
    for (auto i = 0; i < N_COMPONENTS; i++)
        ret = ret * ctx->nSide + decoded[i];
    return ret;
}

__host__ __device__ bool isValid(const State &s) {
    for (auto i = 0; i < N_COORDS - 1; i++)
        for (auto j = i + 1; j < N_COORDS; j++)
            if (X(s, i) == X(s, j) && Y(s, i) == Y(s, j))
                return false;
    return true;
}

__host__ __device__ int anotherAgent(int p) { return p ^ 1; }
__host__ __device__ int anotherResource(int p) { return p ^ 1; }

__host__ __device__ State step(const Context *ctx, const State &s, int p, int a) {
    const auto [dx, dy] = ctx->actions[a];
    if (dx == 0 && dy == 0)
        return s;
    State ret{s};
    const auto pp = anotherAgent(p);
    const auto ppx = X(s, pp), ppy = Y(s, pp);
    auto tx = X(s, p), ty = Y(s, p);

#define CHECK_AND_UPDATE(i)                                                                        \
    do {                                                                                           \
        tx += dx;                                                                                  \
        ty += dy;                                                                                  \
        if (tx < 0 || ctx->nSide <= tx || ty < 0 || ctx->nSide <= ty)                              \
            return s;                                                                              \
        if (tx == ppx && ty == ppy)                                                                \
            return s;                                                                              \
        X(ret, i) = tx;                                                                            \
        Y(ret, i) = ty;                                                                            \
    } while (0)

    CHECK_AND_UPDATE(p);
    for (auto i = 2; i <= 3; i++) {
        if (X(s, i) != tx || Y(s, i) != ty)
            continue;
        CHECK_AND_UPDATE(i);
        auto j = anotherResource(i);
        if (X(s, j) == tx && Y(s, j) == ty)
            CHECK_AND_UPDATE(j);
        break;
    }
    return ret;
}

__host__ __device__ float calculateRewardSpecific(const Context *ctx, int g, const State &s,
                                                  int p) {
    const auto [gx, gy] = ctx->allGoals[g];
    const auto dx = gx - X(s, p + 2);
    const auto dy = gy - Y(s, p + 2);
    // return dx == 0 && dy == 0 ? 0 : expf(-hypotf(dx, dy));
    return dx == 0 && dy == 0 ? 0 : max(0.0, 1 - hypotf(dx, dy) / 5.0);
}

__host__ __device__ float calculateReward(const Context *ctx, const State &s, int p) {
    float sum = 0;
    for (auto i = 0; i < N_GOALS; i++)
        sum += ctx->goals[p][i] * calculateRewardSpecific(ctx, i, s, p);
    return sum;
}

__host__ __device__ float substituteXiL1(SocialGoal xi, float ri, float rj) {
    switch (xi) {
    case COOPERATE:
        return rj;
    case CONFLICT:
    case COMPETE:
        return -rj;
    case COERCE:
        return ri;
    }
    return 0.0;
}

__host__ __device__ float substituteXiL2(SocialGoal xi, SocialGoal xi2, float ri, float Rj,
                                         float rj, bool sameGoal) {
    switch (xi) {
    case COOPERATE:
        return Rj;
    case CONFLICT:
        return -Rj;
    case COMPETE:
        switch (xi2) {
        case COOPERATE:
        case EXCHANGE:
            return Rj;
        case CONFLICT:
        case COMPETE:
            return -Rj;
        case COERCE:
            return sameGoal ? Rj : -Rj;
        }
    case COERCE:
        switch (xi2) {
        case COOPERATE:
        case EXCHANGE:
            return Rj + ri;
        case CONFLICT:
        case COMPETE:
            return -Rj + ri;
        case COERCE:
            return sameGoal ? Rj : -Rj + ri;
        }
    case EXCHANGE:
        if (xi2 == EXCHANGE)
            return 2 * rj + Rj;
    }
    return 0.0;
}

__host__ __device__ inline float actionCost(int action) { return action == 0 ? 0 : 0.05; }

__global__ void kernelReward(const Context *ctx, float *result, int p) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = blockDim.x * gridDim.x;
    for (auto i = index; i < ctx->nStates(); i += stride) {
        auto state = decodeState(ctx, i);
        result[i] = isValid(state) ? calculateReward(ctx, state, p) : 0;
    }
}

__global__ void kernelValueIterateL0(const Context *ctx, float *next, float *delta, float *prev,
                                     int p) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = blockDim.x * gridDim.x;
    const auto pp = anotherAgent(p);
    for (auto i = index; i < ctx->nStates(); i += stride) {
        const auto state = decodeState(ctx, i);
        if (!isValid(state)) {
            delta[i] = 0;
            continue;
        }
        float maxQ = 0;
        for (auto a = 0; a < N_ACTIONS; a++) {
            const auto temp = step(ctx, state, p, a);
            float q = -actionCost(a);
            for (auto b = 0; b < N_ACTIONS; b++) {
                const auto next = step(ctx, temp, pp, b);
                q += prev[encodeState(ctx, next)] / N_ACTIONS;
            }
            maxQ = fmax(maxQ, q);
        }
        next[i] = calculateReward(ctx, state, p) + ctx->gamma * maxQ;
        delta[i] = fabs(next[i] - prev[i]);
    }
}

__global__ void kernelValueIterateL1(const Context *ctx, float *next, float *delta, float *prev,
                                     int p, float *pi, SocialGoal xi) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = blockDim.x * gridDim.x;
    const auto pp = anotherAgent(p);
    for (auto i = index; i < ctx->nStates(); i += stride) {
        const auto state = decodeState(ctx, i);
        if (!isValid(state)) {
            delta[i] = 0;
            continue;
        }
        float maxQ = 0;
        for (auto a = 0; a < N_ACTIONS; a++) {
            const auto temp = step(ctx, state, p, a);
            float numer = 0;
            float denom = 0;
            for (auto b = 0; b < N_ACTIONS; b++) {
                const auto next = step(ctx, temp, pp, b);
                const auto enc = encodeState(ctx, next);
                const auto w = pi[enc];
                numer += w * prev[enc];
                denom += w;
            }
            maxQ = fmax(maxQ, numer / denom - actionCost(a));
        }
        const float ri = calculateReward(ctx, state, p);
        const float rj = calculateReward(ctx, state, pp);
        next[i] = calculateReward(ctx, state, p) + substituteXiL1(xi, ri, rj) + ctx->gamma * maxQ;
        delta[i] = fabs(next[i] - prev[i]);
    }
}

__global__ void kernelValueIterateL2(const Context *ctx, float *next, float *delta, float *prev,
                                     int p, float *pi, SocialGoal xi, const Context *ctx2,
                                     const float *pXis) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = blockDim.x * gridDim.x;
    const auto pp = anotherAgent(p);
    for (auto i = index; i < ctx->nStates(); i += stride) {
        const auto state = decodeState(ctx, i);
        if (!isValid(state)) {
            delta[i] = 0;
            continue;
        }
        float maxQ = 0;
        for (auto a = 0; a < N_ACTIONS; a++) {
            const auto temp = step(ctx, state, p, a);
            float numer = 0;
            float denom = 0;
            for (auto b = 0; b < N_ACTIONS; b++) {
                const auto next = step(ctx, temp, pp, b);
                const auto enc = encodeState(ctx, next);
                const auto w = pi[enc];
                numer += w * prev[enc];
                denom += w;
            }
            maxQ = fmax(maxQ, numer / denom - actionCost(a));
        }
        float Xi = 0;
        for (auto gi = 0; gi < N_GOALS; gi++) {
            const float ri = calculateRewardSpecific(ctx, gi, state, p);
            for (auto gj = 0; gj < N_GOALS; gj++) {
                const float rj = calculateRewardSpecific(ctx, gj, state, pp);
                const float ri2 = calculateRewardSpecific(ctx2, gj, state, p);
                for (auto xi2 = 0; xi2 < N_SOCIAL_GOALS; xi2++) {
                    const float Rj = substituteXiL1((SocialGoal)xi2, rj, ri2);
                    Xi += ctx->goals[p][gi] * ctx->goals[pp][gj] * pXis[xi2] *
                          substituteXiL2(xi, (SocialGoal)xi2, ri, Rj, rj, gi == gj);
                }
            }
        }
        next[i] = calculateReward(ctx, state, p) + Xi + ctx->gamma * maxQ;
        delta[i] = fabs(next[i] - prev[i]);
    }
}

__global__ void kernelValueToPolicy(const Context *ctx, float *value, float offset) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = blockDim.x * gridDim.x;
    for (auto i = index; i < ctx->nStates(); i += stride)
        value[i] = expf(value[i] * ctx->tau - offset);
}

__global__ void kernelCheckNaN(const Context *ctx, float *value, bool *hasNan) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = blockDim.x * gridDim.x;
    for (auto i = index; i < ctx->nStates(); i += stride)
        if (isnan(value[i]))
            *hasNan = true;
}

__global__ void kernelPolicyValueCompose(const Context *ctx, float *result, const float *pi1,
                                         float p1, const float *pi2, float p2) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = blockDim.x * gridDim.x;
    for (auto i = index; i < ctx->nStates(); i += stride)
        result[i] = expf(p1 * logf(pi1[i]) + p2 * logf(pi2[i]));
}

__global__ void kernelPolicyLogitCompose(const Context *ctx, float *result, const float *pi1,
                                         float p1, const float *pi2, float p2) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = blockDim.x * gridDim.x;
    for (auto i = index; i < ctx->nStates(); i += stride)
        result[i] = p1 * pi1[i] + p2 * pi2[i];
}

float optimalOffset(const Context &ctx, float maxValue) {
    const auto prod = ctx.tau * maxValue;
    return std::max(0.0f, prod - 80);
}

Policy valueIterateL0(const Context &ctx, int iters, int agent) {
    float *devA = nullptr;
    float *devB = nullptr;
    float *devDelta = nullptr;
    void *devTemp = nullptr;
    float *devMaxDelta = nullptr;
    Context *devCtx = nullptr;
    size_t tempSize = 0;
    float maxDelta = 0;

    const auto blockSize = 256;
    const auto nBlocks = (ctx.nStates() + blockSize - 1) / blockSize;

    // gpuCheck(cudaSetDevice(0));
    gpuCheck(cudaMalloc((void **)&devA, ctx.nStates() * sizeof(float)));
    gpuCheck(cudaMalloc((void **)&devB, ctx.nStates() * sizeof(float)));
    gpuCheck(cudaMalloc((void **)&devDelta, ctx.nStates() * sizeof(float)));
    gpuCheck(cudaMalloc((void **)&devCtx, sizeof(Context)));
    gpuCheck(cudaMemcpy(devCtx, &ctx, sizeof(Context), cudaMemcpyHostToDevice));

    cub::DeviceReduce::Max(devTemp, tempSize, devDelta, devMaxDelta, ctx.nStates());
    gpuCheck(cudaMalloc((void **)&devTemp, tempSize));
    gpuCheck(cudaMalloc((void **)&devMaxDelta, sizeof(float)));

    gpuCheck(cudaMemset(devA, 0, ctx.nStates() * sizeof(float)));

    ProgressBar bar{fmt::format("L0 policy for {}: ", agent), "", iters};
    for (int i = 0; i < iters; i++) {
        kernelValueIterateL0<<<nBlocks, blockSize>>>(devCtx, devB, devDelta, devA, agent);
        gpuCheck(cudaGetLastError());

        cub::DeviceReduce::Max(devTemp, tempSize, devDelta, devMaxDelta, ctx.nStates());
        gpuCheck(cudaMemcpy(&maxDelta, devMaxDelta, sizeof(float), cudaMemcpyDeviceToHost));
        bar.suffix = fmt::format("MD: {:.3}", maxDelta);
        bar.tick();
        std::swap(devB, devA);
    }

    // Reuse the maxDelta to store the max of values, which helps prevent expf overflow.
    cub::DeviceReduce::Max(devTemp, tempSize, devA, devMaxDelta, ctx.nStates());
    gpuCheck(cudaMemcpy(&maxDelta, devMaxDelta, sizeof(float), cudaMemcpyDeviceToHost));
    kernelValueToPolicy<<<nBlocks, blockSize>>>(devCtx, devA, optimalOffset(ctx, maxDelta));
    gpuCheck(cudaGetLastError());

    gpuCheck(cudaDeviceSynchronize());
    gpuCheck(cudaFree(devB));
    gpuCheck(cudaFree(devDelta));
    gpuCheck(cudaFree(devCtx));
    gpuCheck(cudaFree(devTemp));
    gpuCheck(cudaFree(devMaxDelta));

    return Policy(ctx, agent, nullptr, devA, fmt::format("l0({},{})", agent, ctx.goals));
}

Policy valueIterateL1(const Context &ctx, int iters, int agent, Policy &pi, SocialGoal xi) {
    pi.ensureOnGpu();
    float *devA = nullptr;
    float *devB = nullptr;
    float *devDelta = nullptr;
    void *devTemp = nullptr;
    float *devMaxDelta = nullptr;
    Context *devCtx = nullptr;
    size_t tempSize = 0;
    float maxDelta = 0;

    const auto blockSize = 256;
    const auto nBlocks = (ctx.nStates() + blockSize - 1) / blockSize;

    // gpuCheck(cudaSetDevice(0));
    gpuCheck(cudaMalloc((void **)&devA, ctx.nStates() * sizeof(float)));
    gpuCheck(cudaMalloc((void **)&devB, ctx.nStates() * sizeof(float)));
    gpuCheck(cudaMalloc((void **)&devDelta, ctx.nStates() * sizeof(float)));
    gpuCheck(cudaMalloc((void **)&devCtx, sizeof(Context)));
    gpuCheck(cudaMemcpy(devCtx, &ctx, sizeof(Context), cudaMemcpyHostToDevice));

    cub::DeviceReduce::Max(devTemp, tempSize, devDelta, devMaxDelta, ctx.nStates());
    gpuCheck(cudaMalloc((void **)&devTemp, tempSize));
    gpuCheck(cudaMalloc((void **)&devMaxDelta, sizeof(float)));

    gpuCheck(cudaMemset(devA, 0, ctx.nStates() * sizeof(float)));

    ProgressBar bar{fmt::format("L1 policy for {}: ", agent), "", iters};
    for (int i = 0; i < iters; i++) {
        kernelValueIterateL1<<<nBlocks, blockSize>>>(devCtx, devB, devDelta, devA, agent, pi.onGpu,
                                                     xi);
        gpuCheck(cudaGetLastError());

        cub::DeviceReduce::Max(devTemp, tempSize, devDelta, devMaxDelta, ctx.nStates());
        gpuCheck(cudaMemcpy(&maxDelta, devMaxDelta, sizeof(float), cudaMemcpyDeviceToHost));
        bar.suffix = fmt::format("MD: {:.3}", maxDelta);
        bar.tick();
        std::swap(devB, devA);
    }

    // Reuse the maxDelta to store the max of values, which helps prevent expf overflow.
    cub::DeviceReduce::Max(devTemp, tempSize, devA, devMaxDelta, ctx.nStates());
    gpuCheck(cudaMemcpy(&maxDelta, devMaxDelta, sizeof(float), cudaMemcpyDeviceToHost));
    kernelValueToPolicy<<<nBlocks, blockSize>>>(devCtx, devA, optimalOffset(ctx, maxDelta));
    gpuCheck(cudaGetLastError());

    gpuCheck(cudaDeviceSynchronize());
    gpuCheck(cudaFree(devB));
    gpuCheck(cudaFree(devDelta));
    gpuCheck(cudaFree(devCtx));
    gpuCheck(cudaFree(devTemp));
    gpuCheck(cudaFree(devMaxDelta));

    return Policy(ctx, agent, nullptr, devA,
                  fmt::format("l1({},{},{},{})", agent, ctx.goals, xi, pi.desc));
}

Policy valueIterateL2(const Context &ctx, int iters, int agent, Policy &pi, SocialGoal xi,
                      const Context &ctx2, std::array<float, N_SOCIAL_GOALS> &pXis) {
    pi.ensureOnGpu();
    float *devA = nullptr;
    float *devB = nullptr;
    float *devDelta = nullptr;
    float *devPXis = nullptr;
    void *devTemp = nullptr;
    float *devMaxDelta = nullptr;
    Context *devCtx = nullptr;
    Context *devCtx2 = nullptr;
    size_t tempSize = 0;
    float maxDelta = 0;

    const auto blockSize = 256;
    const auto nBlocks = (ctx.nStates() + blockSize - 1) / blockSize;

    // gpuCheck(cudaSetDevice(0));
    gpuCheck(cudaMalloc((void **)&devA, ctx.nStates() * sizeof(float)));
    gpuCheck(cudaMalloc((void **)&devB, ctx.nStates() * sizeof(float)));
    gpuCheck(cudaMalloc((void **)&devDelta, ctx.nStates() * sizeof(float)));
    gpuCheck(cudaMalloc((void **)&devPXis, N_SOCIAL_GOALS * sizeof(float)));
    gpuCheck(cudaMalloc((void **)&devCtx, sizeof(Context)));
    gpuCheck(cudaMalloc((void **)&devCtx2, sizeof(Context)));
    gpuCheck(cudaMemcpy(devCtx, &ctx, sizeof(Context), cudaMemcpyHostToDevice));
    gpuCheck(cudaMemcpy(devCtx2, &ctx2, sizeof(Context), cudaMemcpyHostToDevice));
    gpuCheck(
        cudaMemcpy(devPXis, pXis.data(), N_SOCIAL_GOALS * sizeof(float), cudaMemcpyHostToDevice));

    cub::DeviceReduce::Max(devTemp, tempSize, devDelta, devMaxDelta, ctx.nStates());
    gpuCheck(cudaMalloc((void **)&devTemp, tempSize));
    gpuCheck(cudaMalloc((void **)&devMaxDelta, sizeof(float)));

    gpuCheck(cudaMemset(devA, 0, ctx.nStates() * sizeof(float)));

    ProgressBar bar{fmt::format("L2 policy for {}: ", agent), "", iters};
    for (int i = 0; i < iters; i++) {
        kernelValueIterateL2<<<nBlocks, blockSize>>>(devCtx, devB, devDelta, devA, agent, pi.onGpu,
                                                     xi, devCtx2, devPXis);
        gpuCheck(cudaGetLastError());

        cub::DeviceReduce::Max(devTemp, tempSize, devDelta, devMaxDelta, ctx.nStates());
        gpuCheck(cudaMemcpy(&maxDelta, devMaxDelta, sizeof(float), cudaMemcpyDeviceToHost));
        bar.suffix = fmt::format("MD: {:.3}", maxDelta);
        bar.tick();
        std::swap(devB, devA);
    }

    cub::DeviceReduce::Max(devTemp, tempSize, devA, devMaxDelta, ctx.nStates());
    gpuCheck(cudaMemcpy(&maxDelta, devMaxDelta, sizeof(float), cudaMemcpyDeviceToHost));
    kernelValueToPolicy<<<nBlocks, blockSize>>>(devCtx, devA, optimalOffset(ctx, maxDelta));
    gpuCheck(cudaGetLastError());

    gpuCheck(cudaDeviceSynchronize());
    gpuCheck(cudaFree(devB));
    gpuCheck(cudaFree(devPXis));
    gpuCheck(cudaFree(devDelta));
    gpuCheck(cudaFree(devCtx));
    gpuCheck(cudaFree(devCtx2));
    gpuCheck(cudaFree(devTemp));
    gpuCheck(cudaFree(devMaxDelta));

    return Policy(ctx, agent, nullptr, devA,
                  fmt::format("l2({},{},{},{},{})", agent, ctx.goals, xi, pi.desc, pXis));
}

std::atomic_int unfreedPolicies = 0;

Policy::~Policy() {
    if (onGpu)
        gpuCheck(cudaFree(onGpu));
    if (onCpu)
        delete[] onCpu;
    if (onGpu || onCpu)
        unfreedPolicies--;
}

void Policy::ensureOnCpu() {
    if (!onCpu) {
        assert(onGpu);
        onCpu = new float[ctx.nStates()];
        gpuCheck(cudaMemcpy(onCpu, onGpu, ctx.nStates() * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

void Policy::ensureOnGpu() {
    if (!onGpu) {
        assert(onCpu);
        gpuCheck(cudaMalloc((void **)&onGpu, ctx.nStates() * sizeof(float)));
        gpuCheck(cudaMemcpy(onGpu, onCpu, ctx.nStates() * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void Policy::freeCpu() {
    ensureOnGpu();
    if (onCpu) {
        delete[] onCpu;
        onCpu = nullptr;
    }
}

void Policy::freeGpu() {
    ensureOnCpu();
    if (onGpu) {
        gpuCheck(cudaFree(onGpu));
        onGpu = nullptr;
    }
}

float Policy::prob(const State &s, int action) {
    ensureOnCpu();
    float numer = 0;
    float denom = 0;
    for (auto a = 0; a < N_ACTIONS; a++) {
        const auto next = step(&ctx, s, agent, a);
        const auto l = onCpu[encodeState(&ctx, next)];
        if (a == action)
            numer += l;
        denom += l;
    }
    return denom <= 1e-8 ? 0 : numer / denom;
}

float Policy::prob(const State &s1, const State &s2) {
    ensureOnCpu();
    float numer = 0;
    float denom = 0;
    for (auto a = 0; a < N_ACTIONS; a++) {
        const auto next = step(&ctx, s1, agent, a);
        const auto l = onCpu[encodeState(&ctx, next)];
        if (next == s2)
            numer += l;
        denom += l;
    }
    return denom <= 1e-8 ? 0 : numer / denom;
}

int Policy::action(const State &s) {
    static std::mt19937 rng; // Use default seed for reproducibility.

    std::array<float, N_ACTIONS> probs;
    for (auto a = 0; a < N_ACTIONS; a++)
        probs[a] = prob(s, a);
    const auto maxP = *std::max_element(probs.begin(), probs.end());
    std::vector<int> actions;
    for (auto a = 0; a < N_ACTIONS; a++)
        if (maxP - probs[a] <= 1e-7)
            actions.push_back(a);

    std::uniform_int_distribution<size_t> dist{0, actions.size() - 1};
    const auto choice = dist(rng);
    assert(choice < actions.size());
    return actions[choice];
}

State Policy::act(const State &s, int &dstAction) {
    dstAction = action(s);
    return step(&ctx, s, agent, dstAction);
}

bool checkNaN(const Context &ctx, float *dev) {
    Context *devCtx = nullptr;
    bool *devHasNaN = nullptr;
    bool hasNaN;
    gpuCheck(cudaMalloc((void **)&devCtx, sizeof(Context)));
    gpuCheck(cudaMalloc((void **)&devHasNaN, sizeof(bool)));
    gpuCheck(cudaMemcpy(devCtx, &ctx, sizeof(Context), cudaMemcpyHostToDevice));
    gpuCheck(cudaMemset(devHasNaN, 0, sizeof(bool)));

    const auto blockSize = 256;
    const auto nBlocks = (ctx.nStates() + blockSize - 1) / blockSize;
    kernelCheckNaN<<<nBlocks, blockSize>>>(devCtx, dev, devHasNaN);
    gpuCheck(cudaMemcpy(&hasNaN, devHasNaN, sizeof(bool), cudaMemcpyDeviceToHost));
    gpuCheck(cudaFree(devCtx));
    gpuCheck(cudaFree(devHasNaN));
    return hasNaN;
}

bool Policy::hasNaN() {
    ensureOnGpu();
    return checkNaN(ctx, onGpu);
}

Policy Policy::valueCompose(Policy &pi1, float p1, Policy &pi2, float p2) {
    assert(pi1.ctx.nSide == pi2.ctx.nSide);
    assert(pi1.agent == pi2.agent);
    pi1.ensureOnGpu();
    pi2.ensureOnGpu();
    float *result = nullptr;
    Context *devCtx = nullptr;
    gpuCheck(cudaMalloc((void **)&result, pi1.ctx.nStates() * sizeof(float)));
    gpuCheck(cudaMalloc((void **)&devCtx, sizeof(Context)));
    gpuCheck(cudaMemcpy(devCtx, &pi1.ctx, sizeof(Context), cudaMemcpyHostToDevice));

    const auto blockSize = 256;
    const auto nBlocks = (pi1.ctx.nStates() + blockSize - 1) / blockSize;
    kernelPolicyValueCompose<<<nBlocks, blockSize>>>(devCtx, result, pi1.onGpu, p1, pi2.onGpu, p2);

    gpuCheck(cudaGetLastError());
    gpuCheck(cudaDeviceSynchronize());
    gpuCheck(cudaFree(devCtx));

    return Policy(pi1.ctx, pi1.agent, nullptr, result, fmt::format("v({},{})", pi1.desc, pi2.desc));
}

Policy Policy::logitCompose(Policy &pi1, float p1, Policy &pi2, float p2) {
    assert(pi1.ctx.nSide == pi2.ctx.nSide);
    assert(pi1.agent == pi2.agent);
    pi1.ensureOnGpu();
    pi2.ensureOnGpu();
    float *result = nullptr;
    Context *devCtx = nullptr;
    gpuCheck(cudaMalloc((void **)&result, pi1.ctx.nStates() * sizeof(float)));
    gpuCheck(cudaMalloc((void **)&devCtx, sizeof(Context)));
    gpuCheck(cudaMemcpy(devCtx, &pi1.ctx, sizeof(Context), cudaMemcpyHostToDevice));

    const auto blockSize = 256;
    const auto nBlocks = (pi1.ctx.nStates() + blockSize - 1) / blockSize;
    kernelPolicyLogitCompose<<<nBlocks, blockSize>>>(devCtx, result, pi1.onGpu, p1, pi2.onGpu, p2);

    gpuCheck(cudaGetLastError());
    gpuCheck(cudaDeviceSynchronize());
    gpuCheck(cudaFree(devCtx));

    return Policy(pi1.ctx, pi1.agent, nullptr, result);
}

void reportMemoryStatus() {
    size_t freeBytes;
    size_t totalBytes;
    gpuCheck(cudaMemGetInfo(&freeBytes, &totalBytes));
    fmt::print("Unfreed memory about {}MB; ", (totalBytes - freeBytes) / 1e6);
    fmt::print("{} unfreed policies\n", unfreedPolicies);
}