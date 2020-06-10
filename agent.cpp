#include <iomanip>
#include <iostream>
#include <chrono>
#include <queue>
#include <tuple>

#include <omp.h>

#include "game.h"
#include "ntuple.h"
#include "agent.h"

Agent::NTupleTable_0::NTuple Agent::retina_0(const Game::State &state) {
    return Agent::NTupleTable_0::NTuple{
        Game::get_tile(state, 0), Game::get_tile(state, 1), Game::get_tile(state, 2),
        Game::get_tile(state, 3), Game::get_tile(state, 4), Game::get_tile(state, 5),
    };
}

Agent::NTupleTable_1::NTuple Agent::retina_1(const Game::State &state) {
    return Agent::NTupleTable_1::NTuple{
        Game::get_tile(state, 4), Game::get_tile(state, 5), Game::get_tile(state, 6),
        Game::get_tile(state, 7), Game::get_tile(state, 8), Game::get_tile(state, 9),
    };
}

Agent::NTupleTable_2::NTuple Agent::retina_2(const Game::State &state) {
    return Agent::NTupleTable_2::NTuple{
        Game::get_tile(state, 0), Game::get_tile(state, 1), Game::get_tile(state, 2),
        Game::get_tile(state, 4), Game::get_tile(state, 5), Game::get_tile(state, 6),
    };
}

Agent::NTupleTable_3::NTuple Agent::retina_3(const Game::State &state) {
    return Agent::NTupleTable_3::NTuple{
        Game::get_tile(state, 4), Game::get_tile(state, 5), Game::get_tile(state, 6),
        Game::get_tile(state, 8), Game::get_tile(state, 9), Game::get_tile(state, 10),
    };
}

float Agent::evaluate_state(const Game::State &state, const Agent::PhaseParams &params) {
    const int phase = Agent::phase(state);
    if (phase >= params.size()) return 0;

    // Unpack Agent parameters as non-modifiable references
    const Agent::NTupleTable_0 &table_0 = params[phase].table_0;
    const Agent::NTupleTable_1 &table_1 = params[phase].table_1;
    const Agent::NTupleTable_2 &table_2 = params[phase].table_2;
    const Agent::NTupleTable_3 &table_3 = params[phase].table_3;

    // Apply each retina and table 8 times for each symmetry
    const Game::State &s_00 = state;
    const Game::State  s_10 = Game::rot90(state);
    const Game::State  s_20 = Game::rot180(state);
    const Game::State  s_30 = Game::rot270(state);
    const Game::State  s_01 = Game::flip(state);
    const Game::State  s_11 = Game::rot90(s_01);
    const Game::State  s_21 = Game::rot180(s_01);
    const Game::State  s_31 = Game::rot270(s_01);

    // Evaluate retina and addressing functions once and use them for both observed value computation and applying updates
    const Agent::NTupleTable_0::NTupleValue
            &t_0_00 = table_0(Agent::retina_0(s_00)), &t_0_10 = table_0(Agent::retina_0(s_10)),
            &t_0_20 = table_0(Agent::retina_0(s_20)), &t_0_30 = table_0(Agent::retina_0(s_30)),
            &t_0_01 = table_0(Agent::retina_0(s_01)), &t_0_11 = table_0(Agent::retina_0(s_11)),
            &t_0_21 = table_0(Agent::retina_0(s_21)), &t_0_31 = table_0(Agent::retina_0(s_31));

    const Agent::NTupleTable_1::NTupleValue
            &t_1_00 = table_1(Agent::retina_1(s_00)), &t_1_10 = table_1(Agent::retina_1(s_10)),
            &t_1_20 = table_1(Agent::retina_1(s_20)), &t_1_30 = table_1(Agent::retina_1(s_30)),
            &t_1_01 = table_1(Agent::retina_1(s_01)), &t_1_11 = table_1(Agent::retina_1(s_11)),
            &t_1_21 = table_1(Agent::retina_1(s_21)), &t_1_31 = table_1(Agent::retina_1(s_31));

    const Agent::NTupleTable_2::NTupleValue
            &t_2_00 = table_2(Agent::retina_2(s_00)), &t_2_10 = table_2(Agent::retina_2(s_10)),
            &t_2_20 = table_2(Agent::retina_2(s_20)), &t_2_30 = table_2(Agent::retina_2(s_30)),
            &t_2_01 = table_2(Agent::retina_2(s_01)), &t_2_11 = table_2(Agent::retina_2(s_11)),
            &t_2_21 = table_2(Agent::retina_2(s_21)), &t_2_31 = table_2(Agent::retina_2(s_31));

    const Agent::NTupleTable_3::NTupleValue
            &t_3_00 = table_3(Agent::retina_3(s_00)), &t_3_10 = table_3(Agent::retina_3(s_10)),
            &t_3_20 = table_3(Agent::retina_3(s_20)), &t_3_30 = table_3(Agent::retina_3(s_30)),
            &t_3_01 = table_3(Agent::retina_3(s_01)), &t_3_11 = table_3(Agent::retina_3(s_11)),
            &t_3_21 = table_3(Agent::retina_3(s_21)), &t_3_31 = table_3(Agent::retina_3(s_31));

    // Response is sum over all retinas
    const float observed_value = (t_0_00 + t_0_10 + t_0_20 + t_0_30) + (t_0_01 + t_0_11 + t_0_21 + t_0_31)
                               + (t_1_00 + t_1_10 + t_1_20 + t_1_30) + (t_1_01 + t_1_11 + t_1_21 + t_1_31)
                               + (t_2_00 + t_2_10 + t_2_20 + t_2_30) + (t_2_01 + t_2_11 + t_2_21 + t_2_31)
                               + (t_3_00 + t_3_10 + t_3_20 + t_3_30) + (t_3_01 + t_3_11 + t_3_21 + t_3_31);

    return observed_value;
}

void Agent::update_state_TC(const Game::State &state, const float &error, const float &learning_rate, Agent::PhaseParams &params) {
    const int phase = Agent::phase(state);
    if (phase >= params.size()) return;

    // Unpack Agent parameters as modifiable references
    Agent::NTupleTable_0 &table_0 = params[phase].table_0;
    Agent::NTupleTable_1 &table_1 = params[phase].table_1;
    Agent::NTupleTable_2 &table_2 = params[phase].table_2;
    Agent::NTupleTable_3 &table_3 = params[phase].table_3;

    // Apply each retina and table 8 times for each symmetry
    const Game::State &s_00 = state;
    const Game::State  s_10 = Game::rot90(state);
    const Game::State  s_20 = Game::rot180(state);
    const Game::State  s_30 = Game::rot270(state);
    const Game::State  s_01 = Game::flip(state);
    const Game::State  s_11 = Game::rot90(s_01);
    const Game::State  s_21 = Game::rot180(s_01);
    const Game::State  s_31 = Game::rot270(s_01);

    // Evaluate retina and addressing functions once and use them for both observed value computation and applying updates
    Agent::NTupleTable_0::NTupleValue
            &t_0_00 = table_0(Agent::retina_0(s_00)), &t_0_10 = table_0(Agent::retina_0(s_10)),
            &t_0_20 = table_0(Agent::retina_0(s_20)), &t_0_30 = table_0(Agent::retina_0(s_30)),
            &t_0_01 = table_0(Agent::retina_0(s_01)), &t_0_11 = table_0(Agent::retina_0(s_11)),
            &t_0_21 = table_0(Agent::retina_0(s_21)), &t_0_31 = table_0(Agent::retina_0(s_31));

    Agent::NTupleTable_1::NTupleValue
            &t_1_00 = table_1(Agent::retina_1(s_00)), &t_1_10 = table_1(Agent::retina_1(s_10)),
            &t_1_20 = table_1(Agent::retina_1(s_20)), &t_1_30 = table_1(Agent::retina_1(s_30)),
            &t_1_01 = table_1(Agent::retina_1(s_01)), &t_1_11 = table_1(Agent::retina_1(s_11)),
            &t_1_21 = table_1(Agent::retina_1(s_21)), &t_1_31 = table_1(Agent::retina_1(s_31));

    Agent::NTupleTable_2::NTupleValue
            &t_2_00 = table_2(Agent::retina_2(s_00)), &t_2_10 = table_2(Agent::retina_2(s_10)),
            &t_2_20 = table_2(Agent::retina_2(s_20)), &t_2_30 = table_2(Agent::retina_2(s_30)),
            &t_2_01 = table_2(Agent::retina_2(s_01)), &t_2_11 = table_2(Agent::retina_2(s_11)),
            &t_2_21 = table_2(Agent::retina_2(s_21)), &t_2_31 = table_2(Agent::retina_2(s_31));

    Agent::NTupleTable_3::NTupleValue
            &t_3_00 = table_3(Agent::retina_3(s_00)), &t_3_10 = table_3(Agent::retina_3(s_10)),
            &t_3_20 = table_3(Agent::retina_3(s_20)), &t_3_30 = table_3(Agent::retina_3(s_30)),
            &t_3_01 = table_3(Agent::retina_3(s_01)), &t_3_11 = table_3(Agent::retina_3(s_11)),
            &t_3_21 = table_3(Agent::retina_3(s_21)), &t_3_31 = table_3(Agent::retina_3(s_31));

    // Update all NTupleTables at all the accessed addresses
    t_0_00.update(error, learning_rate); t_0_10.update(error, learning_rate); t_0_20.update(error, learning_rate); t_0_30.update(error, learning_rate);
    t_0_01.update(error, learning_rate); t_0_11.update(error, learning_rate); t_0_21.update(error, learning_rate); t_0_31.update(error, learning_rate);

    t_1_00.update(error, learning_rate); t_1_10.update(error, learning_rate); t_1_20.update(error, learning_rate); t_1_30.update(error, learning_rate);
    t_1_01.update(error, learning_rate); t_1_11.update(error, learning_rate); t_1_21.update(error, learning_rate); t_1_31.update(error, learning_rate);

    t_2_00.update(error, learning_rate); t_2_10.update(error, learning_rate); t_2_20.update(error, learning_rate); t_2_30.update(error, learning_rate);
    t_2_01.update(error, learning_rate); t_2_11.update(error, learning_rate); t_2_21.update(error, learning_rate); t_2_31.update(error, learning_rate);

    t_3_00.update(error, learning_rate); t_3_10.update(error, learning_rate); t_3_20.update(error, learning_rate); t_3_30.update(error, learning_rate);
    t_3_01.update(error, learning_rate); t_3_11.update(error, learning_rate); t_3_21.update(error, learning_rate); t_3_31.update(error, learning_rate);
}

float Agent::expectimax_estimate_chance_value(const Game::State &state, const int depth, const Agent::Deadline &deadline,
                                              const Agent::PhaseParams &params, Agent::StateValueCache &state_cache) {

    float expectation = 0;

    // Consider each tile on the board
    for (int tile_index = 0; tile_index < Game::BOARD_SIZE; tile_index++) {
        if (deadline.expired()) break;

        // If this tile is not empty then skip
        if (Game::get_tile(state, tile_index) > 0)
            continue;

        // If this tile is empty, generate states with a 2 and 4 tile in that location
        const Game::State state_2 = Game::set_tile(state, tile_index, Game::Tiles::TILE_2),
                          state_4 = Game::set_tile(state, tile_index, Game::Tiles::TILE_4);

        // Sum weighted average over expectations of the chance states.
        // 2 is placed 90% of the time, 4 is placed 10% of the time.
        const float value_2 = Agent::expectimax_search_max_action_value(state_2, depth, deadline, params, state_cache);
        if (value_2 > 0) {
            expectation += 0.9f * value_2;
        }
        const float value_4 = Agent::expectimax_search_max_action_value(state_4, depth, deadline, params, state_cache);
        if (value_4 > 0) {
            expectation += 0.1f * value_4;
        }
    }

    // Weighted average over potential chance states
    return expectation;
}

float Agent::expectimax_afterstate_value(const Game::State &after_state, const int depth, const Agent::Deadline &deadline,
                                         const Agent::PhaseParams &params, Agent::StateValueCache &state_cache) {

    const auto &cached_value = state_cache.find(after_state);
    if (cached_value != state_cache.end()) {
        return cached_value->second;
    }

    float expectation = 0;
    if (depth > 1) {
        expectation = Agent::expectimax_estimate_chance_value(after_state, (depth - 1), deadline, params, state_cache);
    }
    else {
        expectation = Agent::evaluate_state(after_state, params);
    }

    // Cache this transition
    state_cache.insert({ after_state, expectation });

    return expectation;
}

float Agent::expectimax_search_max_action_value(const Game::State &state, const int depth, const Agent::Deadline &deadline,
                                                const Agent::PhaseParams &params, Agent::StateValueCache &state_cache) {

    float max_value = 0;

    for (int action = Game::Action::UP; action < Game::NUM_ACTIONS; action++) {

        // Try out each action and compare the values of the transitions to after_states
        const Game::Transition transition = Game::transition(state, static_cast<Game::Action>(action));

        // Don't consider any action that does not change the game state
        if (transition.terminal) continue;

        const float value = transition.reward + Agent::expectimax_afterstate_value(transition.after_state, depth, deadline, params, state_cache);

        // Select the highest value
        if (value >= max_value) max_value = value;
    }

    return max_value;
}

Game::Transition Agent::expectimax_iterative_search_max_transition(const Game::State &state, const int initial_depth, const int max_depth,
                                                                   const Agent::Deadline &deadline, const Agent::PhaseParams &params) {

    auto start_time = deadline.elapsed();
    Game::Transition max_transition = Agent::expectimax_search_max_transition(state, initial_depth, deadline, params);
    auto prev_duration = deadline.elapsed() - start_time;

    for (int depth = (initial_depth + 1); depth <= max_depth; depth++) {
        if (deadline.remaining() < (2 * prev_duration)) break;

        start_time = deadline.elapsed();
        Game::Transition iterative_transition = Agent::expectimax_search_max_transition(state, depth, deadline, params);
        prev_duration = deadline.elapsed() - start_time;

        if (deadline.expired()) break;

        max_transition = iterative_transition;
    }

    return max_transition;
}

Game::Transition Agent::expectimax_search_max_transition(const Game::State &state, const int depth,
                                                         const Agent::Deadline &deadline, const Agent::PhaseParams &params) {

    Agent::StateValueCache state_cache;

    Game::Transition max_transition{};
    float max_value = 0;

    for (int action = Game::Action::UP; action < Game::NUM_ACTIONS; action++) {
        // Try out each action and compare the values of the transitions to after_states
        const Game::Transition transition = Game::transition(state, static_cast<Game::Action>(action));

        // Don't consider any action that does not change the game state
        if (transition.terminal) continue;

        const float value = transition.reward + Agent::expectimax_afterstate_value(transition.after_state, depth, deadline, params, state_cache);

        // Select the highest value
        if (value >= max_value) {
            max_transition = transition;
            max_value      = value;
        }
    }

    max_transition.value = max_value;
    return max_transition;
}

void Agent::train_agent(const int epoch, const int num_games, const int start_phase, const int end_phase,
                         const float learning_rate, Agent::PhaseParams &params, std::ostream &log) {

    // Start the clock, Carol
    auto start_time = std::chrono::high_resolution_clock::now();

    // Variables for accumulating statistics
    float sum_loss = 0, sum_weight = 0;

    constexpr float lambda     = 0.5;
    const int       trace_size = 3;

    // Parallelize over independent games, reduce statistics between threads to avoid race conditions
    #pragma omp parallel for schedule(dynamic, 1) num_threads(Agent::CPU_THREADS) reduction(+:sum_loss,sum_weight)
    for (int game = 0; game < num_games; game++) {

        // Place a random tile to start
        Game::State state = Agent::random_phase_state((game % (start_phase + 1)));

        std::deque<Agent::Trace> traces;
        Game::State prev_after_state = Game::EMPTY_STATE;

        for (int step = 0;; step++) {

            if ((Agent::phase(state) >= end_phase) || Game::terminal(state)) break;

            // Expectimax search for best local action
            const Game::Transition transition = Agent::expectimax_search_max_transition(state, 1, Agent::Deadline(Agent::Deadline::NO_DEADLINE), params);

            // Test if action ended the game
            if (transition.terminal) break;

            // Place a random tile
            const Game::State new_state = Game::place_random_tile(transition.after_state, Game::rand_state(),
                                                                  Game::rand_tile());

            // TC(lambda)
            if (prev_after_state != Game::EMPTY_STATE) {
                const float error = transition.value - Agent::evaluate_state(prev_after_state, params);
                traces.push_back(Agent::Trace{prev_after_state, error});

                if (traces.size() >= trace_size) {
                    float lambda_error = 0, lam = 1;
                    for (auto &trace : traces) {
                        lambda_error += trace.error * lam;
                        lam *= lambda;
                    }

                    sum_loss += lambda_error;
                    sum_weight++;

                    Agent::update_state_TC(traces.front().after_state, lambda_error, learning_rate, params);
                    traces.pop_front();
                }
            }

            // If placing a tile puts the game into a terminal state then end the game
            if (state == new_state) break;

            // Update to the next state
            state = new_state;
            prev_after_state = transition.after_state;
        }
    }

    // Stop the clock, Carol
    auto end_time   = std::chrono::high_resolution_clock::now();
    auto delta_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Log data to log file
    log  << epoch << ','
         << num_games << ','
         << start_phase << ','
         << end_phase << ','
         << std::setprecision(4) << delta_time << ','
         << std::setprecision(4) << (sum_loss / sum_weight) << ','
         << std::setprecision(4) << learning_rate << std::endl;

    std::cout << std::endl << "Epoch " << epoch << " training   completed in "
              << std::setprecision(4) << (delta_time / 1000.f) << " seconds..." << std::endl;
}

void Agent::evaluate_agent(const int epoch, const int num_games, const int start_phase, const int end_phase,
                          const int initial_depth, const int max_depth, const int max_duration, const Agent::PhaseParams &params, std::ostream &log) {

    // Start the clock, Carol
    auto start_time = std::chrono::high_resolution_clock::now();

    // Variables for accumulating statistics
    float avg_phase = 0, maximum_phase = 0, avg_score = 0, maximum_score = 0, avg_tile  = 0, maximum_tile  = 0,
          tile_128  = 0, tile_256  = 0, tile_512  = 0, tile_1024  = 0,
          tile_2048 = 0, tile_4096 = 0, tile_8192 = 0, tile_16384 = 0, tile_32768 = 0;

    // Parallelize over independent games, reduce statistics between threads to avoid race conditions
    #pragma omp parallel for schedule(dynamic, 1) num_threads(Agent::CPU_THREADS) \
                             reduction(max : maximum_phase,maximum_score,maximum_tile) \
                             reduction( +  : avg_phase,avg_score,avg_tile,tile_128,tile_256,tile_512,tile_1024,\
                                             tile_2048,tile_4096,tile_8192,tile_16384,tile_32768)
    for (int game = 0; game < num_games; game++) {
        
        // Play a single game
        float score = 0;
        Game::State state = Agent::random_phase_state(start_phase);

        for (int step = 0;; step++) {

            if ((Agent::phase(state) >= end_phase) || Game::terminal(state)) break;

            // Expectimax search 1-ply for best local action
            const Game::Transition transition = Agent::expectimax_iterative_search_max_transition(state, initial_depth, max_depth, Agent::Deadline(max_duration), params);

            // Game score is sum of transition rewards
            score += transition.reward;

            // Test if action ended the game
            if (transition.terminal) break;

            // Place a random tile
            const Game::State new_state = Game::place_random_tile(transition.after_state, Game::rand_state(), Game::rand_tile());

            // If placing a tile puts the game into a terminal state then end the game
            if (state == new_state) break;

            // Update to the next state
            state = new_state;
        }
        
        // End of current game

        // Accumulate occurrences for tiles of at least the maximum final tile
        const Game::Tile max_tile = Game::maximum_tile(state);
        if (max_tile >= Game::Tiles::TILE_128)   tile_128++;
        if (max_tile >= Game::Tiles::TILE_256)   tile_256++;
        if (max_tile >= Game::Tiles::TILE_512)   tile_512++;
        if (max_tile >= Game::Tiles::TILE_1024)  tile_1024++;
        if (max_tile >= Game::Tiles::TILE_2048)  tile_2048++;
        if (max_tile >= Game::Tiles::TILE_4096)  tile_4096++;
        if (max_tile >= Game::Tiles::TILE_8192)  tile_8192++;
        if (max_tile >= Game::Tiles::TILE_16384) tile_16384++;
        if (max_tile >= Game::Tiles::TILE_32768) tile_32768++;

        const float state_phase = (float) Agent::phase(state);
        avg_phase += state_phase;
        if (state_phase > maximum_phase) maximum_phase = state_phase;

        // Accumulate sums for avgs
        avg_score += score;
        avg_tile  += (float) max_tile;
        
        // Accumulate maximums
        if ((float) max_tile >  maximum_tile) maximum_tile  = max_tile;
        if (score > maximum_score)            maximum_score = score;
        
        // Advance to next game
    }
    
    // Stop the clock, Carol
    auto end_time   = std::chrono::high_resolution_clock::now();
    auto delta_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Log data to log file
    log  << epoch << ','
         << num_games << ','
         << start_phase << ','
         << end_phase << ','
         << initial_depth << ','
         << max_duration  << ','
         << std::setprecision(4) << delta_time << ','

         << (int) maximum_phase << ','
         << std::setprecision(4) << (avg_phase  / (float) num_games) << ','
         << (int) maximum_score << ','
         << std::setprecision(4) << (avg_score  / (float) num_games) << ','
         << (int) maximum_tile << ','
         << std::setprecision(4) << (avg_tile   / (float) num_games) << ','

         << std::setprecision(4) << (tile_128   / (float) num_games) << ','
         << std::setprecision(4) << (tile_256   / (float) num_games) << ','
         << std::setprecision(4) << (tile_512   / (float) num_games) << ','
         << std::setprecision(4) << (tile_1024  / (float) num_games) << ','
         << std::setprecision(4) << (tile_2048  / (float) num_games) << ','
         << std::setprecision(4) << (tile_4096  / (float) num_games) << ','
         << std::setprecision(4) << (tile_8192  / (float) num_games) << ','
         << std::setprecision(4) << (tile_16384 / (float) num_games) << ','
         << std::setprecision(4) << (tile_32768 / (float) num_games) << std::endl;

    std::cout << "Epoch " << epoch << " evaluation completed in "
              << std::setprecision(4) << (delta_time / 1000.f) << " seconds..." << std::endl;
}

int Agent::phase(const Game::State &state) {
    if (Game::has_tile(state, Game::Tiles::TILE_32768)) {
        return 16;
    }

    return (Game::has_tile(state, Game::Tiles::TILE_16384) ? 8 : 0) +
           (Game::has_tile(state, Game::Tiles::TILE_8192)  ? 4 : 0) +
           (Game::has_tile(state, Game::Tiles::TILE_4096)  ? 2 : 0) +
           (Game::has_tile(state, Game::Tiles::TILE_2048)  ? 1 : 0);
}

Game::State Agent::random_phase_state(const int phase) {
    if (phase == 0) {
        return Game::place_random_tile(0, Game::rand_state(), Game::rand_tile());
    }

    if (phase >= 16) {
        return Game::place_random_tile(0, Game::rand_state(), Game::Tiles::TILE_32768);
    }

    Game::State state = 0;
    if (phase >= 8) {
        state = Game::place_random_tile(state, Game::rand_state(), Game::Tiles::TILE_16384);
    }
    if ((phase >= 4 && phase <= 7) || (phase >= 12)) {
        state = Game::place_random_tile(state, Game::rand_state(), Game::Tiles::TILE_8192);
    }
    if (phase == 2 || phase == 3 || phase == 6 || phase == 7 || phase == 10 || phase == 11 || phase == 14 || phase == 15) {
        state = Game::place_random_tile(state, Game::rand_state(), Game::Tiles::TILE_4096);
    }
    if (phase == 1 || phase == 3 || phase == 5 || phase == 7 || phase == 9 || phase == 11 || phase == 13 || phase == 15) {
        state = Game::place_random_tile(state, Game::rand_state(), Game::Tiles::TILE_2048);
    }

//    0     2 or 4
//    1     2048
//    2     4096
//    3     4096  2048
//    4     8192
//    5     8192  2048
//    6     8192  4096
//    7     8192  4096 2048
//    8     16384
//    9     16384 2048
//    10    16384 4096
//    11    16384 4096 2048
//    12    16384 8192
//    13    16384 8192 2048
//    14    16384 8192 4096
//    15    16384 8192 4096 2048

    return state;
}

std::ofstream Agent::log_evaluation_csv(const std::string &path) {
    std::ofstream log(path, std::ios_base::app);
    std::cout << "Opening [ " << path << " ] ";
    if (!log.is_open()) {
        std::cout << "failed." << std::endl;
        return log;
    }
    std::cout << "success." << std::endl;

    if (log.is_open() && (log.tellp() == 0)) {
        log << "epoch,num_games,start_phase,end_phase,depth,max_duration,time,max_phase,avg_phase,max_score,avg_score,max_tile,avg_tile,"
            << "tile_128,tile_256,tile_512,tile_1024,tile_2048,tile_4096,"
            << "tile_8192,tile_16384,tile_32768" << std::endl;
    }
    return log;
}

std::ofstream Agent::log_training_csv(const std::string &path) {
    std::ofstream log(path, std::ios_base::app);
    std::cout << "Opening [ " << path << " ] ";
    if (!log.is_open()) {
        std::cout << "failed." << std::endl;
        return log;
    }
    std::cout << "success." << std::endl;

    if (log.is_open() && (log.tellp() == 0)) {
        log << "epoch,num_games,start_phase,end_phase,time,loss,learning_rate" << std::endl;
    }
    return log;
}

void Agent::save(const std::string &path, const Agent::Params &params) {
    std::ofstream file(path, std::ios::binary);
    std::cout << "Saving [ " << path << " ] ";
    if (!file.is_open()) {
        std::cout << "failed." << std::endl;
        return;
    }

    params.table_0.save(file);
    params.table_1.save(file);
    params.table_2.save(file);
    params.table_3.save(file);
    std::cout << "success." << std::endl;
}

void Agent::save(const std::string &path, const Agent::PhaseParams &phase_params) {
    std::ofstream file(path, std::ios::binary);
    std::cout << "Saving [ " << path << " ] ";
    if (!file.is_open()) {
        std::cout << "failed." << std::endl;
        return;
    }

    for (const auto &params : phase_params) {
        params.table_0.save(file);
        params.table_1.save(file);
        params.table_2.save(file);
        params.table_3.save(file);
    }
    std::cout << "success." << std::endl;
}

void Agent::load(const std::string &path, Agent::Params &params) {
    std::ifstream file(path, std::ios::binary);
    std::cout << "Saving [ " << path << " ] ";
    if (!file.is_open()) {
        std::cout << "failed." << std::endl;
        return;
    }

    params.table_0.load(file);
    params.table_1.load(file);
    params.table_2.load(file);
    params.table_3.load(file);
    std::cout << "success." << std::endl;
}

void Agent::load(const std::string &path, Agent::PhaseParams &phase_params) {
    std::ifstream file(path, std::ios::binary);
    std::cout << "Saving [ " << path << " ] ";
    if (!file.is_open()) {
        std::cout << "failed." << std::endl;
        return;
    }

    for (auto &params : phase_params) {
        params.table_0.load(file);
        params.table_1.load(file);
        params.table_2.load(file);
        params.table_3.load(file);
    }
    std::cout << "success." << std::endl;
}