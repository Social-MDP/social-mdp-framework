#pragma once

#include <array>
#include <string>
#include <atomic>

constexpr int N_COORDS = 4;
constexpr int N_COMPONENTS = 2 * N_COORDS;
constexpr int N_ACTIONS = 5;
constexpr int N_GOALS = 2;
using Component = int;
using State = std::array<Component, N_COMPONENTS>;
using IntState = int;

#define X(s, i) s[2 * (i)]
#define Y(s, i) s[2 * (i) + 1]

/**
 * @brief A context for a two-agent grid world game.
 */
struct Context {
    Component nSide;
    float gamma;
    float tau;
    std::array<std::pair<Component, Component>, N_GOALS> allGoals;
    std::array<std::array<float, N_GOALS>, 2> goals;
    std::array<std::pair<Component, Component>, N_ACTIONS> actions{
        {{0, 0}, {0, 1}, {0, -1}, {1, 0}, {-1, 0}}};

    // clang-format off
#ifdef __CUDACC__
    __host__ __device__
#endif
    IntState nStates() const {
        // Power by exponentiation. Given N_COMPONENTS is a constant this makes the compiler
        // generate the optimal code.
        int ret = 1, tmp = nSide;
        for (auto i = N_COMPONENTS; i; i >>= 1, tmp *= tmp)
            (i & 1) && (ret *= tmp);
        return ret;
    }
    // clang-format on
};

extern std::atomic_int unfreedPolicies;
void reportMemoryStatus();

/**
 * @brief Stores a softmax policy. The raw logits array are either stored on CPU or on GPU (or both)
 * to reduce time for memory copying operations.
 */
struct Policy {
    Context ctx;
    int agent;
    float *onCpu;
    float *onGpu;
    std::string desc = "";

    Policy(const Policy &) = delete;
    Policy(Policy &&other)
        : ctx{other.ctx}, agent{other.agent}, onCpu{other.onCpu}, onGpu{other.onGpu}, desc{other.desc} {
        other.onCpu = nullptr;
        other.onGpu = nullptr;
    }
    Policy &operator=(const Policy &) = delete;
    Policy &operator=(Policy &&other) {
        this->ctx = std::move(other.ctx);
        this->agent = other.agent;
        this->onCpu = other.onCpu;
        this->onGpu = other.onGpu;
        this->desc = std::move(other.desc);
        other.onCpu = nullptr;
        other.onGpu = nullptr;
        return *this;
    }
    Policy(const Context &ctx, int agent, float *onCpu, float *onGpu, const std::string &desc = "")
        : ctx{ctx}, agent{agent}, onCpu{onCpu}, onGpu{onGpu}, desc{desc} {
        unfreedPolicies++;
    }
    ~Policy();

    static Policy valueCompose(Policy &pi1, float p1, Policy &pi2, float p2);
    static Policy logitCompose(Policy &pi1, float p1, Policy &pi2, float p2);

    void ensureOnCpu();
    void ensureOnGpu();
    void freeCpu();
    void freeGpu();
    bool hasNaN();

    float prob(const State &s, int action);
    float prob(const State &s1, const State &s2);
    int action(const State &s);
    State act(const State &s, int &dstAction);
};

Policy valueIterateL0(const Context &ctx, int iters, int agent);
Policy valueIterateL1(const Context &ctx, int iters, int agent, Policy &pi, float chi);
Policy valueIterateL2(const Context &ctx, int iters, int agent, Policy &pi, float chi,
                      const Context &ctx2, float chi2);