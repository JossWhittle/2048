#pragma once

#include <inttypes.h>

namespace Game {

    using State = uint64_t;
    using Tile  = unsigned char;

    constexpr State EMPTY_STATE = 0;

    constexpr int MAX_TILE    = 12;
    constexpr int EDGE_SIZE   = 4;
    constexpr int BOARD_SIZE  = EDGE_SIZE * EDGE_SIZE;
    constexpr int NUM_ACTIONS = 4;

    enum Tiles : Tile {
        TILE_2     = 1,
        TILE_4     = 2,
        TILE_8     = 3,
        TILE_16    = 4,
        TILE_32    = 5,
        TILE_64    = 6,
        TILE_128   = 7,
        TILE_256   = 8,
        TILE_512   = 9,
        TILE_1024  = 10,
        TILE_2048  = 11,
        TILE_4096  = 12,
        TILE_8192  = 13,
        TILE_16384 = 14,
        TILE_32768 = 15,
    };

    enum Action : unsigned char {
        UP    = 0,
        RIGHT = 1,
        DOWN  = 2,
        LEFT  = 3
    };

    constexpr unsigned int TILE_INDEX[EDGE_SIZE][EDGE_SIZE] = {
            {  0,  1,  2,  3 },
            {  4,  5,  6,  7 },
            {  8,  9, 10, 11 },
            { 12, 13, 14, 15 }
    };

    constexpr unsigned int TILE_OFFSETS[EDGE_SIZE][EDGE_SIZE] = {
            {  0,  4,  8, 12 },
            { 16, 20, 24, 28 },
            { 32, 36, 40, 44 },
            { 48, 52, 56, 60 },
    };

    constexpr State TILE_MASK = ((1u << 4u) - 1u);
    constexpr State TILE_MASKS[EDGE_SIZE][EDGE_SIZE] = {
        {
            (TILE_MASK),
            (TILE_MASK << 4u),
            (TILE_MASK << 8u),
            (TILE_MASK << 12u),
        },
        {
            (TILE_MASK << 16u),
            (TILE_MASK << 20u),
            (TILE_MASK << 24u),
            (TILE_MASK << 28u),
        },
        {
            (TILE_MASK << 32u),
            (TILE_MASK << 36u),
            (TILE_MASK << 40u),
            (TILE_MASK << 44u),
        },
        {
            (TILE_MASK << 48u),
            (TILE_MASK << 52u),
            (TILE_MASK << 56u),
            (TILE_MASK << 60u),
        }
    };

    Tile  get_tile(const State &state, const int i);
    Tile  get_tile(const State &state, const int y, const int x);

    State set_tile(const State &state, const int i, const Tile &value);
    State set_tile(const State &state, const int y, const int x, const Tile &value);

    State   flip(const State &state);
    State  rot90(const State &state);
    State rot180(const State &state);
    State rot270(const State &state);

    State  slide_left(const State &state);
    State  merge_left(const State &state, float &reward);
    State       merge(const State &state, const Action &action, float &reward);

    State place_random_tile(const State &state, const Game::State &rand, const Game::Tile &tile);

    State rand64();
    State rand_state();
    Tile  rand_tile();
    Tile  uniform_rand_tile();

    struct Transition {
        State  state;
        Action action;
        State  after_state;
        float  reward;
        bool   terminal;
    };

    Transition transition(const State &state, const Action &action);

    bool terminal(const State &state);

    Tile maximum_tile(const State &state);

    void  print_state(const State &state);
}