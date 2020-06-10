#include "game.h"

#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>

Game::Tile Game::get_tile(const Game::State &state, const int i) {
    return (state >> (i * Game::BITS_PER_TILE)) & Game::TILE_MASK;
}

Game::Tile Game::get_tile(const Game::State &state, const int y, const int x) {
    return Game::get_tile(state, x + (y * Game::EDGE_SIZE));
}

Game::State Game::set_tile(const Game::State &state, const int i, const Game::Tile &value) {
    return (state & Game::INV_TILE_MASKS[i]) ^ (State(value) << (i * Game::BITS_PER_TILE));
}

Game::State Game::set_tile(const Game::State &state, const int y, const int x, const Game::Tile &value) {
    return Game::set_tile(state, x + (y * Game::EDGE_SIZE), value);
}

Game::State Game::flip(const Game::State &state) {
    Game::State new_state = Game::EMPTY_STATE;
    for (int y = 0; y < Game::EDGE_SIZE; y++) {
        for (int x = 0; x < Game::EDGE_SIZE; x++) {
            new_state = Game::set_tile(new_state, y, x, Game::get_tile(state, y, (Game::EDGE_SIZE - 1) - x));
        }
    }
    return new_state;
}

Game::State Game::rot90(const Game::State &state) {
    Game::State new_state = Game::EMPTY_STATE;
    for (int y = 0; y < Game::EDGE_SIZE; y++) {
        for (int x = 0; x < Game::EDGE_SIZE; x++) {
            new_state = Game::set_tile(new_state, y, x, Game::get_tile(state, (Game::EDGE_SIZE - 1) - x, y));
        }
    }
    return new_state;
}

Game::State Game::rot180(const Game::State &state) {
    Game::State new_state = Game::EMPTY_STATE;
    for (int y = 0; y < Game::EDGE_SIZE; y++) {
        for (int x = 0; x < Game::EDGE_SIZE; x++) {
            new_state = Game::set_tile(new_state, y, x, Game::get_tile(state, (Game::EDGE_SIZE - 1) - y, (Game::EDGE_SIZE - 1) - x));
        }
    }
    return new_state;
}

Game::State Game::rot270(const Game::State &state) {
    Game::State new_state = Game::EMPTY_STATE;
    for (int y = 0; y < Game::EDGE_SIZE; y++) {
        for (int x = 0; x < Game::EDGE_SIZE; x++) {
            new_state = Game::set_tile(new_state, y, x, Game::get_tile(state, x, (Game::EDGE_SIZE - 1) - y));
        }
    }
    return new_state;
}

Game::State Game::slide_left(const Game::State &state) {
    Game::State new_state = Game::EMPTY_STATE;
    for (int y = 0; y < Game::EDGE_SIZE; y++) {
        for (int ox = 0, x = 0; ox < Game::EDGE_SIZE; ox++) {
            const Tile tile = Game::get_tile(state, y, ox);
            if (tile > 0) new_state = Game::set_tile(new_state, y, x++, tile);
        }
    }
    return new_state;
}

Game::State Game::merge_left(const Game::State &state, float &reward) {
    reward = 0;

    // Compact the zeros before merging
    Game::State new_state = Game::slide_left(state);

    // Merge each row
    for (int y = 0, i = 1; y < Game::EDGE_SIZE; y++, i++) {

        // For each tile, merge if it is next to the same tile
        for (int x = 1; x < Game::EDGE_SIZE; x++, i++) {
            const Tile tile = Game::get_tile(new_state, i);
            if ((tile > 0) && (Game::get_tile(new_state, (i - 1)) == tile)) {

                // Reward is the sum of the two tiles merged
                reward += std::pow(2, (tile + 1));
                new_state = Game::set_tile(new_state, (i - 1), 0);
                new_state = Game::set_tile(new_state, i, (tile + 1));
            }
        }
    }

    // Compact the zeros left after merging
    return Game::slide_left(new_state);
}

Game::State Game::merge(const Game::State &state, const Game::Action &action, float &reward) {
    if (action == Game::Action::UP) {
        return Game::rot90(Game::merge_left(Game::rot270(state), reward));
    }
    else if (action == Game::Action::RIGHT) {
        return Game::rot180(Game::merge_left(Game::rot180(state), reward));
    }
    else if (action == Game::Action::DOWN) {
        return Game::rot270(Game::merge_left(Game::rot90(state), reward));
    }
    return Game::merge_left(state, reward);
}

Game::State Game::place_random_tile(const Game::State &state, const Game::State &rand_state, const Game::Tile &new_tile) {
    int max_rand = -1,
        max_i    = -1;

    for (int i = 0; i < Game::BOARD_SIZE; i++) {

        const Tile tile = Game::get_tile(state, i);
        const Tile rand = Game::get_tile(rand_state,  i);

        if ((tile == 0) && ((int) rand > max_rand)) {
            max_rand = (int) rand;
            max_i    = i;
        }
    }

    if (max_rand >= 0) {
        return Game::set_tile(state, max_i, new_tile);
    }
    return state;
}

uint64_t Game::rand64() {
    static std::mt19937_64 engine((std::random_device())());
    static std::uniform_int_distribution<long long int> dist(std::llround(std::pow(2,61)),
                                                             std::llround(std::pow(2,62)));
    return uint64_t(dist(engine));
}

Game::State Game::rand_state() {
    return Game::rand64();
}

Game::Tile Game::rand_tile() {
    return ((Game::rand64() % 10) == 0) ? Game::Tiles::TILE_4 : Game::Tiles::TILE_2;
}

Game::Transition Game::transition(const Game::State &state, const Game::Action &action) {
    float reward;
    const Game::State after_state = Game::merge(state, action, reward);
    return Game::Transition{ state, action, after_state, reward, 0, (state == after_state) };
}

bool Game::terminal(const Game::State &state) {
    float reward;
    return (state == Game::merge(state, Game::Action::UP,    reward)) &&
           (state == Game::merge(state, Game::Action::RIGHT, reward)) &&
           (state == Game::merge(state, Game::Action::DOWN,  reward)) &&
           (state == Game::merge(state, Game::Action::LEFT,  reward));
}

Game::Tile Game::maximum_tile(const Game::State &state) {
    Game::Tile max_tile = 0;
    for (int i = 0; i < Game::BOARD_SIZE; i++) {
        const Game::Tile tile = Game::get_tile(state, i);
        if (tile > max_tile) max_tile = tile;
    }
    return max_tile;
}

bool Game::has_tile(const Game::State &state, const Game::Tile &tile) {
    for (int i = 0; i < Game::BOARD_SIZE; i++) {
        if (Game::get_tile(state, i) == tile) return true;
    }
    return false;
}

void Game::print_state(const Game::State &state) {
    std::cout << '+' << std::string(((Game::EDGE_SIZE * 3) + 1), '-') << '+' << std::endl;
    for (int y = 0; y < Game::EDGE_SIZE; y++) {
        std::cout << "| ";
        for (int x = 0; x < Game::EDGE_SIZE; x++) {
            std::cout << std::setw(2) << int(Game::get_tile(state, y, x)) << ' ';
        }
        std::cout << '|' << std::endl;
    }
    std::cout << '+' << std::string(((Game::EDGE_SIZE * 3) + 1), '-') << '+' << std::endl;
}