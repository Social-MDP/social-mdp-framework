#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <optional>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

#include <nlohmann/json.hpp>

#include "estimation.h"
#include "visualize.h"

using namespace std::chrono;
using json = nlohmann::json;

bool finished = false;

void to_json(json &j, const Context &ctx) {
    j = json{{"nSide", ctx.nSide},
             {"gamma", ctx.gamma},
             {"tau", ctx.tau},
             {"allGoals", ctx.allGoals},
             {"goals", ctx.goals}};
}

void from_json(const json &j, Context &ctx) {
    j.at("nSide").get_to(ctx.nSide);
    j.at("gamma").get_to(ctx.gamma);
    j.at("tau").get_to(ctx.tau);
    j.at("allGoals").get_to(ctx.allGoals);
    j.at("goals").get_to(ctx.goals);
}

struct RecordStep {
    State state;
    int actionA;
    int actionB;
    std::array<float, N_ACTIONS> probActionA;
    std::array<float, N_ACTIONS> probActionB;
    std::array<float, N_GOALS> l0GoalEstimationA;
    std::array<float, N_GOALS> l0GoalEstimationB;
    std::array<float, N_SOCIAL_GOALS> l1GoalEstimationA;
    std::array<float, N_SOCIAL_GOALS> l1GoalEstimationB;
    std::string extra;
};

void to_json(json &j, const RecordStep &s) {
    j = json{{"state", s.state},
             {"actionA", s.actionA},
             {"actionB", s.actionB},
             {"probActionA", s.probActionA},
             {"probActionB", s.probActionB},
             {"l0GoalEstimationA", s.l0GoalEstimationA},
             {"l0GoalEstimationB", s.l0GoalEstimationB},
             {"l1GoalEstimationA", s.l1GoalEstimationA},
             {"l1GoalEstimationB", s.l1GoalEstimationB},
             {"extra", s.extra}};
}

void from_json(const json &j, RecordStep &s) {
    j.at("state").get_to(s.state);
    j.at("actionA").get_to(s.actionA);
    j.at("actionB").get_to(s.actionB);
    j.at("probActionA").get_to(s.probActionA);
    j.at("probActionB").get_to(s.probActionB);
    j.at("l0GoalEstimationA").get_to(s.l0GoalEstimationA);
    j.at("l0GoalEstimationB").get_to(s.l0GoalEstimationB);
    j.at("l1GoalEstimationA").get_to(s.l1GoalEstimationA);
    j.at("l1GoalEstimationB").get_to(s.l1GoalEstimationB);
    j.at("extra").get_to(s.extra);
}

struct AgentConfig {
    int goal, level, xi;
    bool forcePGE, forceSGE;
};

void to_json(json &j, const AgentConfig &c) {
    j = json{{"goal", c.goal},
             {"level", c.level},
             {"xi", c.xi},
             {"forcePGE", c.forcePGE},
             {"forceSGE", c.forceSGE}};
}

void from_json(const json &j, AgentConfig &c) {
    j.at("goal").get_to(c.goal);
    j.at("xi").get_to(c.xi);
    j.at("level").get_to(c.level);
    j.at("forcePGE").get_to(c.forcePGE);
    j.at("forceSGE").get_to(c.forceSGE);
}

struct InputConfig {
    int nSides;
    int nIterations;
    float gamma;
    float tau;
    State initialState;
    std::array<std::pair<Component, Component>, N_GOALS> allGoals;
    std::array<AgentConfig, 2> agents;
} config;

void to_json(json &j, const InputConfig &c) {
    j = json{{"nSides", c.nSides}, {"nIterations", c.nIterations},   {"gamma", c.gamma},
             {"tau", c.tau},       {"initialState", c.initialState}, {"allGoals", c.allGoals},
             {"agents", c.agents}};
}

void from_json(const json &j, InputConfig &c) {
    j.at("agents").get_to(c.agents);
    j.at("nSides").get_to(c.nSides);
    j.at("nIterations").get_to(c.nIterations);
    j.at("gamma").get_to(c.gamma);
    j.at("tau").get_to(c.tau);
    j.at("initialState").get_to(c.initialState);
    j.at("allGoals").get_to(c.allGoals);
}

std::string generateFileName() {
    const auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream ss;
    ss << "./records/";
    ss << "A" << config.agents[0].goal << config.agents[0].level << config.agents[0].xi;
    ss << "B" << config.agents[1].goal << config.agents[1].level << config.agents[1].xi;
    ss << "G";
    for (const auto &x : config.allGoals)
        ss << x.first << x.second;
    ss << "I";
    for (const auto x : config.initialState)
        ss << x;
    ss << std::setprecision(2) << "y" << config.gamma << "t" << config.tau << ".json";
    fmt::print("writing to {}\n", ss.str());
    return ss.str();
}

void play() {
    int N_ITERS = config.nIterations;
    int G_A = config.agents[0].goal;
    int G_B = config.agents[1].goal;

    // Level of agents.
    int L_A = config.agents[0].level;
    int L_B = config.agents[1].level;

    const int CHI_A = config.agents[0].xi;
    const int CHI_B = config.agents[1].xi;
    fmt::print("Chi indices: A {}, B {}", CHI_A, CHI_B);

    // Is it necessary to do Social Goal Estimation on A / B?
    bool SGE_A = L_B == 2 || config.agents[0].forceSGE;
    bool SGE_B = L_A == 2 || config.agents[1].forceSGE;

    // Is it necessary to do Physical Goal Estimation on A / B?
    bool PGE_A = SGE_B || L_B == 1 || config.agents[0].forcePGE;
    bool PGE_B = SGE_A || L_A == 1 || config.agents[1].forcePGE;

    // All possible deterministic contexts.
    std::array<std::array<Context, N_GOALS>, N_GOALS> ctxs;
    for (auto i = 0; i < N_GOALS; i++) {
        for (auto j = 0; j < N_GOALS; j++) {
            ctxs[i][j] = Context{
                .nSide = 10,
                .gamma = 0.95,
                .tau = 5,
                .allGoals = config.allGoals,
            };
            ctxs[i][j].goals[0][i] = 1.0;
            ctxs[i][j].goals[1][j] = 1.0;
        }
    }
    const auto ctx = ctxs[G_A][G_B];

    json record{{"ctx", ctx}, {"agents", config.agents}};

    // Pre-compute level 0 policies.
    std::vector<std::shared_ptr<Policy>> piAL0;
    std::vector<std::shared_ptr<Policy>> piBL0;
    {
        const auto start = high_resolution_clock::now();
        for (auto i = 0; i < N_GOALS; i++) {
            // We need to compute a goal-i policy for A if
            // - i is our actual goal;
            // - or we need to do physical goal inference on A.
            const auto doA = i == G_A || PGE_A;
            // The same goes for B.
            const auto doB = i == G_B || PGE_B;
            piAL0.push_back(doA ? std::make_shared<Policy>(valueIterateL0(ctxs[i][G_B], N_ITERS, 0))
                                : nullptr);
            piBL0.push_back(doB ? std::make_shared<Policy>(valueIterateL0(ctxs[G_A][i], N_ITERS, 1))
                                : nullptr);
        }
        const auto end = high_resolution_clock::now();
        const duration<float> elapsed = end - start;
        fmt::print("Level 0 value iteration took {}.\n", elapsed);
    }

    // State s{{9, 2, 9, 4, 4, 5, 4, 7}};
    // State s{{7, 2, 7, 7, 5, 3, 5, 6}};
    // State s{{5, 3, 5, 6, 7, 3, 7, 6}};
    // State s{{9, 2, 9, 5, 4, 5, 4, 7}};
    State s = config.initialState;
    MultiPolicyEstimator<N_GOALS> pgeA(s);
    MultiPolicyEstimator<N_GOALS> pgeB(s);
    MultiPolicyEstimator<N_SOCIAL_GOALS> sgeA(s);
    MultiPolicyEstimator<N_SOCIAL_GOALS> sgeB(s);
    record["states"].push_back(RecordStep{.state = s,
                                          .actionA = 0,
                                          .actionB = 0,
                                          .probActionA = generateUniform<N_ACTIONS>(),
                                          .probActionB = generateUniform<N_ACTIONS>(),
                                          .l0GoalEstimationA = generateUniform<N_GOALS>(),
                                          .l0GoalEstimationB = generateUniform<N_GOALS>(),
                                          .extra = "Initial"});
    renderState(ctx, s);

    std::vector<std::shared_ptr<Policy>> piAL1;
    std::vector<std::shared_ptr<Policy>> piBL1;

    // Always start with level 0 policies.
    auto piA = piAL0[G_A];
    auto piB = piBL0[G_B];

    for (auto t = 0; t < 20; t++) {
        RecordStep step{.state = s};

        // Take actions and store the resulting action into step.action*.
        const auto s1 = piA->act(s, step.actionA);
        const auto s2 = piB->act(s1, step.actionB);

        for (auto i = 0; i < N_ACTIONS; i++) {
            step.probActionA[i] = piA->prob(s, i);
            step.probActionB[i] = piB->prob(s1, i);
        }

        fmt::print("A: {}\n", step.probActionA);
        fmt::print("B: {}\n", step.probActionB);

        // Update physical goal estimation.
        if (PGE_A) {
            pgeA.update(s, s1, piAL0.begin(), piAL0.end());
            for (auto i = 0; i < N_GOALS; i++)
                step.l0GoalEstimationA[i] = pgeA.prob(i);
        }
        if (PGE_B) {
            pgeB.update(s1, s2, piBL0.begin(), piBL0.end());
            for (auto i = 0; i < N_GOALS; i++)
                step.l0GoalEstimationB[i] = pgeB.prob(i);
        }
        fmt::print("Time {}: \nPG A: {}, PG B: {};\n", t, step.l0GoalEstimationA,
                   step.l0GoalEstimationB);

        // Update social goal estimation.
        // Temporary context based on which L1 policies for A & B are computed, incorporating an
        // estimate of the physical goal of the other agent.
        Context ctxA = ctx;
        Context ctxB = ctx;
        for (auto i = 0; i < N_GOALS; i++) {
            ctxA.goals[1][i] = pgeB.prob(i);
            ctxB.goals[0][i] = pgeA.prob(i);
        }

        piAL1.clear();
        piBL1.clear();

        auto piHatA = PGE_A ? std::make_shared<Policy>(Policy::valueCompose(
                                  *piAL0[0], pgeA.prob(0), *piAL0[1], pgeA.prob(1)))
                            : nullptr;
        auto piHatB = PGE_B ? std::make_shared<Policy>(Policy::valueCompose(
                                  *piBL0[0], pgeB.prob(0), *piBL0[1], pgeB.prob(1)))
                            : nullptr;
        for (auto i = 0; i < N_SOCIAL_GOALS; i++) {
            const auto doA = L_A == 1 && CHI_A == i || SGE_A;
            const auto doB = L_B == 1 && CHI_B == i || SGE_B;
            piAL1.push_back(doA ? std::make_shared<Policy>(
                                      valueIterateL1(ctxA, N_ITERS, 0, *piHatB, (SocialGoal)i))
                                : nullptr);
            piBL1.push_back(doB ? std::make_shared<Policy>(
                                      valueIterateL1(ctxB, N_ITERS, 1, *piHatA, (SocialGoal)i))
                                : nullptr);
        }

        if (SGE_A) {
            sgeA.update(s, s1, piAL1.begin(), piAL1.end());
            for (auto i = 0; i < N_SOCIAL_GOALS; i++)
                step.l1GoalEstimationA[i] = sgeA.prob(i);
        }
        if (SGE_B) {
            sgeB.update(s1, s2, piBL1.begin(), piBL1.end());
            for (auto i = 0; i < N_SOCIAL_GOALS; i++)
                step.l1GoalEstimationB[i] = sgeB.prob(i);
        }

        {
            float chiB = 0;
            switch (L_A) {
            case 1:
                piA = piAL1[CHI_A];
                break;
            case 2:
                piHatB = std::make_shared<Policy>(
                    Policy::valueCompose(*piBL1[0], sgeB.prob(0), *piBL1[1], sgeB.prob(1)));
                for (auto i = 2; i < N_SOCIAL_GOALS; i++)
                    piHatB = std::make_shared<Policy>(
                        Policy::valueCompose(*piHatB, 1, *piBL1[i], sgeB.prob(i)));
                piA = std::make_shared<Policy>(
                    valueIterateL2(ctxA, N_ITERS, 0, *piHatB, (SocialGoal)CHI_A, ctxB, sgeB.probs));
                break;
            default:
                piA = piAL0[G_A];
                break;
            }
        }

        {
            float chiA = 0;
            switch (L_B) {
            case 1:
                piB = piBL1[CHI_B];
                break;
            case 2:
                piHatA = std::make_shared<Policy>(
                    Policy::valueCompose(*piAL1[0], sgeA.prob(0), *piAL1[1], sgeA.prob(1)));
                for (auto i = 2; i < N_SOCIAL_GOALS; i++)
                    piHatA = std::make_shared<Policy>(
                        Policy::valueCompose(*piHatA, 1, *piAL1[i], sgeA.prob(i)));
                piB = std::make_shared<Policy>(
                    valueIterateL2(ctxB, N_ITERS, 0, *piHatA, (SocialGoal)CHI_B, ctxA, sgeA.probs));
                break;
            default:
                piB = piBL0[G_B];
                break;
            }
        }

        record["states"].push_back(step);
        renderState(ctx, s = s2);

        fmt::print("SG A: {}, SG B: {}\n", step.l1GoalEstimationA, step.l1GoalEstimationB);
        reportMemoryStatus();
    }

    {
        std::ofstream output{generateFileName()};
        output << record.dump(2) << std::endl;
    }

    finished = true;
}

int main(int argc, char **argv) {
    std::string configPath = "config.json";
    if (argc > 1)
        configPath = argv[1];
    std::ifstream configFile{configPath};
    json configJson;
    configFile >> configJson;
    fmt::print("Read JSON config from {}: \n{}\n", configPath, configJson.dump(2));
    config = configJson;

    std::thread thread{play};
    renderLoop();
    thread.join();
    reportMemoryStatus();
    return 0;
}