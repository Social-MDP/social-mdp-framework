#define RENDER_WITH_FTXUI

#include <cassert>
#include <chrono>
#include <mutex>
#include <thread>

#ifdef RENDER_WITH_SFML
#include <SFML/Graphics.hpp>
#endif

#ifdef RENDER_WITH_FTXUI
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/screen.hpp>

#include <ftxui/dom/canvas.hpp>
#include <ftxui/dom/node.hpp>
#include <ftxui/screen/color.hpp>
#endif

#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/ranges.h>

#include "visualize.h"

// Using global variables to communicate is a horrible horrible horrible idea,
// but it works.
static std::unique_ptr<Context> ctxToRender = nullptr;
static std::unique_ptr<State> stateToRender = nullptr;
static std::mutex renderMutex;

static std::optional<std::string> savePath = std::nullopt;
static std::mutex captureMutex;
extern bool finished;

void renderState(const Context &ctx, const State &s) {
    renderMutex.lock();
    ctxToRender = std::make_unique<Context>(ctx);
    stateToRender = std::make_unique<State>(s);
    renderMutex.unlock();
}

#ifdef RENDER_WITH_SFML
static sf::RenderWindow *renderWindow = nullptr;
constexpr float ANIMATION_ALPHA = 0.5;

/**
 * @brief An animated sprite. Assumes the texture is square.
 */
struct AnimatedSprite {
    sf::Texture texture;
    sf::Sprite sprite;
    float targetScale = 1.0f;
    sf::Vector2f targetPosition;

    AnimatedSprite(const std::string imagePath) {
        if (!texture.loadFromFile(imagePath))
            exit(1);
        texture.setSmooth(true);
        sprite.setTexture(texture);
        targetPosition = sprite.getPosition();
    }

    void setTargetPosition(sf::Vector2f targetPosition) { this->targetPosition = targetPosition; }

    void setTargetSize(float targetSize) { this->targetScale = targetSize / texture.getSize().x; }

    const sf::Sprite &update() {
        const auto currentPosition = sprite.getPosition();
        const auto nextPosition =
            ANIMATION_ALPHA * currentPosition + (1 - ANIMATION_ALPHA) * targetPosition;
        sprite.setPosition(nextPosition);

        const auto currentScale = sprite.getScale().x;
        const auto nextScale = ANIMATION_ALPHA * currentScale + (1 - ANIMATION_ALPHA) * targetScale;
        sprite.setScale({nextScale, nextScale});
        return sprite;
    }
};
#endif

void saveCapture(const std::string &path) {
    captureMutex.lock();
    savePath = path;
    captureMutex.unlock();
}

void renderLoop() {
#ifdef RENDER_WITH_SFML
    const auto SCREEN_SIZE = 840;

    sf::RenderWindow window{sf::VideoMode(SCREEN_SIZE, SCREEN_SIZE), "Grid World",
                            sf::Style::Titlebar | sf::Style::Close};
    window.setFramerateLimit(120);
    renderWindow = &window;

    AnimatedSprite agentA{"images/yellow_robot.bmp"};
    AnimatedSprite agentB{"images/red_robot.bmp"};
    AnimatedSprite resourceA{"images/yellow_bucket.bmp"};
    AnimatedSprite resourceB{"images/red_bucket.bmp"};
    AnimatedSprite goalA{"images/flower.bmp"};
    AnimatedSprite goalB{"images/tree.bmp"};

    sf::Texture capture;
    capture.create(SCREEN_SIZE, SCREEN_SIZE);

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }
        if (finished) {
            window.close();
        }
        window.clear(sf::Color::White);
        renderMutex.lock();
        if (ctxToRender && stateToRender) {
            const auto cellSize = static_cast<float>(SCREEN_SIZE) / ctxToRender->nSide;
            agentA.setTargetSize(cellSize);
            agentB.setTargetSize(cellSize);
            resourceA.setTargetSize(cellSize);
            resourceB.setTargetSize(cellSize);
            goalA.setTargetSize(cellSize);
            goalB.setTargetSize(cellSize);
            const auto &s = *stateToRender;
            agentA.setTargetPosition({Y(s, 0) * cellSize, X(s, 0) * cellSize});
            agentB.setTargetPosition({Y(s, 1) * cellSize, X(s, 1) * cellSize});
            resourceA.setTargetPosition({Y(s, 2) * cellSize, X(s, 2) * cellSize});
            resourceB.setTargetPosition({Y(s, 3) * cellSize, X(s, 3) * cellSize});
            const auto [gAx, gAy] = ctxToRender->allGoals[0];
            goalA.setTargetPosition({gAy * cellSize, gAx * cellSize});
            const auto [gBx, gBy] = ctxToRender->allGoals[1];
            goalB.setTargetPosition({gBy * cellSize, gBx * cellSize});
            window.draw(goalA.update());
            window.draw(goalB.update());
            window.draw(agentA.update());
            window.draw(agentB.update());
            window.draw(resourceA.update());
            window.draw(resourceB.update());
        }
        renderMutex.unlock();
        captureMutex.lock();
        if (savePath.has_value()) {
            capture.update(window);
            const auto image = capture.copyToImage();
            image.saveToFile(savePath.value());
            savePath = std::nullopt;
        }
        captureMutex.unlock();
        window.display();
    }
    renderWindow = nullptr;
#elif defined(RENDER_WITH_CONSOLE)
    using namespace std::chrono;
    while (!finished) {
        renderMutex.lock();
        if (ctxToRender && stateToRender) {
            fmt::print("┏");
            for (auto c = 0; c < ctxToRender->nSide; c++)
                fmt::print("━━");
            fmt::print("┓\n");
            for (auto r = 0; r < ctxToRender->nSide; r++) {
                fmt::print("┃");
                for (auto c = 0; c < ctxToRender->nSide; c++) {
                    bool special = false;

#define RENDER_GOAL(g, color)                                                                      \
    do {                                                                                           \
        const auto [gx, gy] = ctxToRender->allGoals[g];                                            \
        if (gx == r && gy == c) {                                                                  \
            special = true;                                                                        \
            fmt::print(fmt::fg(color), "<>");                                                      \
        }                                                                                          \
    } while (0)
#define RENDER_STATE(i, color, str)                                                                \
    do {                                                                                           \
        if (stateToRender->at(2 * i) == r && stateToRender->at(2 * i + 1) == c) {                  \
            special = true;                                                                        \
            fmt::print(fmt::fg(color), str);                                                       \
        }                                                                                          \
    } while (0)

                    RENDER_GOAL(0, fmt::color::blue);
                    RENDER_GOAL(1, fmt::color::green);
                    RENDER_STATE(0, fmt::color::yellow, "AA");
                    RENDER_STATE(1, fmt::color::red, "BB");
                    RENDER_STATE(2, fmt::color::yellow, "[]");
                    RENDER_STATE(3, fmt::color::red, "[]");
                    if (!special)
                        fmt::print("  ");
                }
                fmt::print("┃\n");
            }
            fmt::print("┗");
            for (auto c = 0; c < ctxToRender->nSide; c++)
                fmt::print("━━");
            fmt::print("┛\n");
            ctxToRender = nullptr;
            stateToRender = nullptr;
        }
        renderMutex.unlock();
        std::this_thread::sleep_for(milliseconds(16));
    }
#elif defined(RENDER_WITH_FTXUI)
    using namespace std::chrono;
    using namespace ftxui;
    const auto CELL_SIZE = 9;
    const auto RADIUS = CELL_SIZE / 2;

    while (!finished) {
        renderMutex.lock();
        if (ctxToRender && stateToRender) {
            const auto CANVAS_SIZE = ctxToRender->nSide * CELL_SIZE;
            auto c = Canvas(CANVAS_SIZE, CANVAS_SIZE);
#define COORD(r, c) ((c)*CELL_SIZE + RADIUS - 1), ((r)*CELL_SIZE + RADIUS - 1)

            c.DrawPointCircleFilled(
                COORD(ctxToRender->allGoals[0].first, ctxToRender->allGoals[0].second), RADIUS,
                Color::Magenta);
            c.DrawPointCircleFilled(
                COORD(ctxToRender->allGoals[1].first, ctxToRender->allGoals[1].second), RADIUS,
                Color::Blue);
            c.DrawPointCircle(COORD(stateToRender->at(0), stateToRender->at(1)), RADIUS,
                              Color::Yellow);
            c.DrawPointCircle(COORD(stateToRender->at(2), stateToRender->at(3)), RADIUS,
                              Color::Red);
            c.DrawPointCircleFilled(COORD(stateToRender->at(4), stateToRender->at(5)), RADIUS,
                                    Color::Yellow);
            c.DrawPointCircleFilled(COORD(stateToRender->at(6), stateToRender->at(7)), RADIUS,
                                    Color::Red);

            auto document = canvas(&c) | border;
            auto screen = Screen::Create(Dimension::Fit(document));
            Render(screen, document);
            screen.Print();
            fmt::print("\n");
            ctxToRender = nullptr;
            stateToRender = nullptr;
        }
        renderMutex.unlock();
        std::this_thread::sleep_for(milliseconds(16));
    }

#else
#error "A rendering backend must be selected"
#endif
}
