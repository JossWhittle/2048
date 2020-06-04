#pragma once

#include <string>
#include <fstream>
#include <ostream>

#include "game.h"
#include "ntuple.h"

namespace Agent {

    constexpr int CPU_THREADS = 18;

    constexpr int NTUPLE_SIZE_0 = 6, NTUPLE_SIZE_1 = 6, NTUPLE_SIZE_2 = 6, NTUPLE_SIZE_3 = 6;

    using NTupleTable_0 = NTupleTable<NTUPLE_SIZE_0, Game::MAX_TILE+1>;
    using NTupleTable_1 = NTupleTable<NTUPLE_SIZE_1, Game::MAX_TILE+1>;
    using NTupleTable_2 = NTupleTable<NTUPLE_SIZE_2, Game::MAX_TILE+1>;
    using NTupleTable_3 = NTupleTable<NTUPLE_SIZE_3, Game::MAX_TILE+1>;

    // Utility structure to keep multiple NTupleTables together
    struct Params {
        NTupleTable_0 table_0;
        NTupleTable_1 table_1;
        NTupleTable_2 table_2;
        NTupleTable_3 table_3;
    };

    // Each NTupleTable has a retina function that extracts the tuple values from a game state
    NTupleTable_0::NTuple retina_0(const Game::State &state);
    NTupleTable_1::NTuple retina_1(const Game::State &state);
    NTupleTable_2::NTuple retina_2(const Game::State &state);
    NTupleTable_3::NTuple retina_3(const Game::State &state);

    struct Trace {
        Game::State after_state;
        Game::State new_state;
    };

    // Approximate value of a state is the sum over the response from each NTupleTable evaluated on all 8 symmetries of the state
    float evaluate_state(const Game::State &state, const Params &params);

    // TD(0) Temporal Difference Learning update params to move estimate for state closer to the target value
//    float update_state_TD0(const Game::State &state, const float &expected_value, const float &learning_rate, Params &params);
    float update_state_TC0(const Game::State &state, const float &expected_value, const float &learning_rate, Params &params);

    // Expectimax search for best action from the current state
    float            expectimax_estimate_chance_value(  const Game::State &state, const int depth, const Params &params);
    float            expectimax_search_max_action_value(const Game::State &state, const int depth, const Params &params);
    Game::Transition expectimax_search_max_transition(  const Game::State &state, const int depth, const Params &params);

    // Play N games on the CPU and update parameters
    float train_agent(const int epoch, const int num_games, const float learning_rate, Params &params, std::ostream &log);

    // Play N games on the CPU and evaluate performance
    void evaluate_agent(const int epoch, const int num_games, const int depth, const Params &params, std::ostream &log);

    // Open a log file and write the column header
    std::ofstream log_training_csv(const std::string &path);
    std::ofstream log_evaluation_csv(const std::string &path);

    // Save and load parameters to binary file
    void save(const std::string &path, const Params &params);
    void load(const std::string &path, Params &params);
}