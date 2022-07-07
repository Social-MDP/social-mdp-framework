#pragma once

#include <cmath>
#include <string>

#include <fmt/core.h>

constexpr int PROGRESS_WIDTH = 100;

/**
 * @brief A very simple progress bar.
 */
struct ProgressBar {
    std::string prefix;
    std::string suffix;

    ProgressBar(const ProgressBar &) = delete;
    ProgressBar(const std::string &prefix, const std::string &suffix, int full)
        : prefix(prefix), suffix(suffix), full(full), progress(0) {}

    void update() {
        const auto p = static_cast<float>(progress) / full;
        const auto percent = fmt::format(" {}% ", static_cast<int>(round(100 * p)));
        const auto bar = PROGRESS_WIDTH - prefix.length() - suffix.length() - percent.length() - 2;
        const auto filled = static_cast<int>(round(p * bar));
        fmt::print("\r{}[{}{}]{}{}", prefix, std::string(filled, '='),
                   std::string(bar - filled, ' '), percent, suffix);
        if (progress == full)
            fmt::print("\n");
    }

    void tick() {
        if (progress < full)
            progress++;
        update();
    }

  private:
    int full;
    int progress;
};