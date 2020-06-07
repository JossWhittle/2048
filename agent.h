#pragma once

#include <string>
#include <fstream>
#include <ostream>

#include "game.h"
#include "ntuple.h"

namespace Agent {

    constexpr int CPU_THREADS = 16;

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

    constexpr int NUM_PHASES = 15, END_PHASE = (NUM_PHASES + 1);
    using PhaseParams = std::array<Agent::Params, NUM_PHASES+1>;

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
    float evaluate_state(const Game::State &state, const PhaseParams &params);

    // TD(0) Temporal Difference Learning update params to move estimate for state closer to the target value
//    float update_state_TD0(const Game::State &state, const float &expected_value, const float &learning_rate, Params &params);
    float update_state_TC0(const Game::State &state, const float &expected_value, const float &learning_rate, PhaseParams &params);

    // Expectimax search for best action from the current state
    float            expectimax_estimate_chance_value(  const Game::State &state, const int depth, const PhaseParams &params);
    float            expectimax_search_max_action_value(const Game::State &state, const int depth, const PhaseParams &params);
    Game::Transition expectimax_search_max_transition(  const Game::State &state, const int depth, const PhaseParams &params);

    // Determine if a state meets the criteria to end the game
    int phase(const Game::State &state);
    Game::State random_phase_state(const int phase);

    // Play N games on the CPU and update parameters
    float train_agent(const int epoch, const int num_games, const int start_phase, const int end_phase, const float learning_rate, PhaseParams &params, std::ostream &log);

    // Play N games on the CPU and evaluate performance
    void evaluate_agent(const int epoch, const int num_games, const int start_phase, const int end_phase, const int depth, const PhaseParams &params, std::ostream &log);

    // Open a log file and write the column header
    std::ofstream log_training_csv(const std::string &path);
    std::ofstream log_evaluation_csv(const std::string &path);

    void promote_params(const Params &params, Params &new_params);

    // Save and load parameters to binary file
    void save(const std::string &path, const Params &params);
    void save(const std::string &path, const PhaseParams &params);
    void load(const std::string &path, Params &params);
    void load(const std::string &path, PhaseParams &params);
}