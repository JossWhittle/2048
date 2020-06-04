#include <iomanip>
#include <iostream>
#include <chrono>
#include <stack>
#include <tuple>

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

float Agent::evaluate_state(const Game::State &state, const Agent::Params &params) {
    // Unpack Agent parameters as non-modifiable references
    const Agent::NTupleTable_0 &table_0 = params.table_0;
    const Agent::NTupleTable_1 &table_1 = params.table_1;
    const Agent::NTupleTable_2 &table_2 = params.table_2;
    const Agent::NTupleTable_3 &table_3 = params.table_3;

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
    const float &t_0_00 = table_0(Agent::retina_0(s_00)), &t_0_10 = table_0(Agent::retina_0(s_10)),
                &t_0_20 = table_0(Agent::retina_0(s_20)), &t_0_30 = table_0(Agent::retina_0(s_30)),
                &t_0_01 = table_0(Agent::retina_0(s_01)), &t_0_11 = table_0(Agent::retina_0(s_11)),
                &t_0_21 = table_0(Agent::retina_0(s_21)), &t_0_31 = table_0(Agent::retina_0(s_31)),
                
                &t_1_00 = table_1(Agent::retina_1(s_00)), &t_1_10 = table_1(Agent::retina_1(s_10)),
                &t_1_20 = table_1(Agent::retina_1(s_20)), &t_1_30 = table_1(Agent::retina_1(s_30)),
                &t_1_01 = table_1(Agent::retina_1(s_01)), &t_1_11 = table_1(Agent::retina_1(s_11)),
                &t_1_21 = table_1(Agent::retina_1(s_21)), &t_1_31 = table_1(Agent::retina_1(s_31)),
                
                &t_2_00 = table_2(Agent::retina_2(s_00)), &t_2_10 = table_2(Agent::retina_2(s_10)),
                &t_2_20 = table_2(Agent::retina_2(s_20)), &t_2_30 = table_2(Agent::retina_2(s_30)),
                &t_2_01 = table_2(Agent::retina_2(s_01)), &t_2_11 = table_2(Agent::retina_2(s_11)),
                &t_2_21 = table_2(Agent::retina_2(s_21)), &t_2_31 = table_2(Agent::retina_2(s_31)),
                
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

float Agent::update_state_TD0(const Game::State &state, const float &expected_value, const float &learning_rate, Agent::Params &params) {
    // Unpack Agent parameters as modifiable references
    Agent::NTupleTable_0 &table_0 = params.table_0;
    Agent::NTupleTable_1 &table_1 = params.table_1;
    Agent::NTupleTable_2 &table_2 = params.table_2;
    Agent::NTupleTable_3 &table_3 = params.table_3;

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
    float &t_0_00 = table_0(Agent::retina_0(s_00)), &t_0_10 = table_0(Agent::retina_0(s_10)),
          &t_0_20 = table_0(Agent::retina_0(s_20)), &t_0_30 = table_0(Agent::retina_0(s_30)),
          &t_0_01 = table_0(Agent::retina_0(s_01)), &t_0_11 = table_0(Agent::retina_0(s_11)),
          &t_0_21 = table_0(Agent::retina_0(s_21)), &t_0_31 = table_0(Agent::retina_0(s_31)),

          &t_1_00 = table_1(Agent::retina_1(s_00)), &t_1_10 = table_1(Agent::retina_1(s_10)),
          &t_1_20 = table_1(Agent::retina_1(s_20)), &t_1_30 = table_1(Agent::retina_1(s_30)),
          &t_1_01 = table_1(Agent::retina_1(s_01)), &t_1_11 = table_1(Agent::retina_1(s_11)),
          &t_1_21 = table_1(Agent::retina_1(s_21)), &t_1_31 = table_1(Agent::retina_1(s_31)),

          &t_2_00 = table_2(Agent::retina_2(s_00)), &t_2_10 = table_2(Agent::retina_2(s_10)),
          &t_2_20 = table_2(Agent::retina_2(s_20)), &t_2_30 = table_2(Agent::retina_2(s_30)),
          &t_2_01 = table_2(Agent::retina_2(s_01)), &t_2_11 = table_2(Agent::retina_2(s_11)),
          &t_2_21 = table_2(Agent::retina_2(s_21)), &t_2_31 = table_2(Agent::retina_2(s_31)),

          &t_3_00 = table_3(Agent::retina_3(s_00)), &t_3_10 = table_3(Agent::retina_3(s_10)),
          &t_3_20 = table_3(Agent::retina_3(s_20)), &t_3_30 = table_3(Agent::retina_3(s_30)),
          &t_3_01 = table_3(Agent::retina_3(s_01)), &t_3_11 = table_3(Agent::retina_3(s_11)),
          &t_3_21 = table_3(Agent::retina_3(s_21)), &t_3_31 = table_3(Agent::retina_3(s_31));

    // Response is sum over all retinas
    const float observed_value = (t_0_00 + t_0_10 + t_0_20 + t_0_30) + (t_0_01 + t_0_11 + t_0_21 + t_0_31)
                               + (t_1_00 + t_1_10 + t_1_20 + t_1_30) + (t_1_01 + t_1_11 + t_1_21 + t_1_31)
                               + (t_2_00 + t_2_10 + t_2_20 + t_2_30) + (t_2_01 + t_2_11 + t_2_21 + t_2_31)
                               + (t_3_00 + t_3_10 + t_3_20 + t_3_30) + (t_3_01 + t_3_11 + t_3_21 + t_3_31);
    
    // TD(0) Temporal Difference Learning
    const float error = expected_value - observed_value;
    const float delta = error * learning_rate;

    // Update all NTupleTables at all the accessed addresses
    t_0_00 += delta; t_0_10 += delta; t_0_20 += delta; t_0_30 += delta;
    t_0_01 += delta; t_0_11 += delta; t_0_21 += delta; t_0_31 += delta;
    t_1_00 += delta; t_1_10 += delta; t_1_20 += delta; t_1_30 += delta;
    t_1_01 += delta; t_1_11 += delta; t_1_21 += delta; t_1_31 += delta;
    t_2_00 += delta; t_2_10 += delta; t_2_20 += delta; t_2_30 += delta;
    t_2_01 += delta; t_2_11 += delta; t_2_21 += delta; t_2_31 += delta;
    t_3_00 += delta; t_3_10 += delta; t_3_20 += delta; t_3_30 += delta;
    t_3_01 += delta; t_3_11 += delta; t_3_21 += delta; t_3_31 += delta;

    // Return L1 loss
    return std::abs(error);
}

float Agent::expectimax_estimate_chance_value(const Game::State &state, const int depth, const Agent::Params &params) {

    float sum_chance_value = 0, sum_weight = 0;

    // Consider each tile on the board
    for (int tile_index = 0; tile_index < Game::BOARD_SIZE; tile_index++) {

        // If this tile is not empty then skip
        if (Game::get_tile(state, tile_index) > 0)
            continue;

        // If this tile is empty, generate states with a 2 and 4 tile in that location
        const Game::State state_2 = Game::set_tile(state, tile_index, Game::Tiles::TILE_2),
                          state_4 = Game::set_tile(state, tile_index, Game::Tiles::TILE_4);

        // Sum weighted average over expectations of the chance states.
        // 2 is placed 90% of the time, 4 is placed 10% of the time.
        const float value_2 = Agent::expectimax_search_max_action_value(state_2, depth, params) * 0.9f;
        if (value_2 > 0) {
            sum_weight       += 0.9f;
            sum_chance_value += value_2;
        }
        const float value_4 = Agent::expectimax_search_max_action_value(state_4, depth, params) * 0.1f;
        if (value_4 > 0) {
            sum_weight       += 0.1f;
            sum_chance_value += value_4;
        }
    }

    // Weighted average over potential chance states
    return (sum_chance_value > 0) ? (sum_chance_value / sum_weight) : 0;
}

float Agent::expectimax_search_max_action_value(const Game::State &state, const int depth, const Agent::Params &params) {

    float max_value = 0;

    for (int action = Game::Action::UP; action < Game::NUM_ACTIONS; action++) {
        // Try out each action and compare the values of the transitions to after_states
        const Game::Transition transition = Game::transition(state, static_cast<Game::Action>(action));

        // Don't consider any action that does not change the game state
        if (transition.terminal) continue;

        // Value of this action is the sum of the local reward and the value of the after_state
        float value = transition.reward;
        if (depth > 1) {
            value += Agent::expectimax_estimate_chance_value(transition.after_state, (depth - 1), params);
        }
        else {
            value += Agent::evaluate_state(transition.after_state, params);
        }

        // Select the highest value
        if (value >= max_value) max_value = value;
    }

    return max_value;
}

Game::Transition Agent::expectimax_search_max_transition(const Game::State &state, const int depth, const Agent::Params &params) {

    Game::Transition max_transition{};
    float max_value = 0;

    for (int action = Game::Action::UP; action < Game::NUM_ACTIONS; action++) {
        // Try out each action and compare the values of the transitions to after_states
        const Game::Transition transition = Game::transition(state, static_cast<Game::Action>(action));

        // Don't consider any action that does not change the game state
        if (transition.terminal) continue;

        // Value of this action is the sum of the local reward and the value of the after_state
        float value = transition.reward;
        if (depth > 1) {
            value += Agent::expectimax_estimate_chance_value(transition.after_state, (depth - 1), params);
        }
        else {
            value += Agent::evaluate_state(transition.after_state, params);
        }

        // Select the highest value
        if (value >= max_value) {
            max_transition = transition;
            max_value      = value;
        }
    }

    return max_transition;
}

float Agent::train_agent(const int epoch, const int num_games, const float learning_rate, Agent::Params &params, std::ostream &log) {

    // Start the clock, Carol
    auto start_time = std::chrono::high_resolution_clock::now();

    // Variables for accumulating statistics
    float sum_loss = 0, sum_weight = 0;

    std::stack<Agent::Trace> trace;

    // Parallelize over independent games, reduce statistics between threads to avoid race conditions
    #pragma omp parallel for schedule(dynamic, 1) num_threads(Agent::CPU_THREADS) \
                             reduction(+:sum_loss,sum_weight) private(trace)
    for (int game = 0; game < num_games; game++) {

        // Place a random tile to start
        Game::State state = Game::place_random_tile(0, Game::rand_state(), (1 + (game % (Game::Tiles::TILE_32768))));

        for (int step = 0;; step++) {

            // Expectimax search for best local action
            const Game::Transition transition = Agent::expectimax_search_max_transition(state, 1, params);

            // Test if action ended the game
            if (transition.terminal) break;

            // Place a random tile
            const Game::State new_state = Game::place_random_tile(transition.after_state, Game::rand_state(), Game::rand_tile());

            // If placing a tile puts the game into a terminal state then end the game
            if ((state == new_state) || Game::terminal(new_state)) break;

            // If the game continues, then the expected value for this new state should be updated based on the best future reward
            trace.push({ transition.after_state, new_state });
//            const float target_value = Agent::expectimax_search_max_action_value(new_state, 1, params);
//            sum_loss += Agent::update_state_TD0(transition.after_state, target_value, learning_rate, params);
//            sum_weight++;

            // Update to the next state
            state = new_state;
        }

        while (!trace.empty()) {
            const auto &transition = trace.top();
            const float target_value = Agent::expectimax_search_max_action_value(transition.new_state, 1, params);
            sum_loss += Agent::update_state_TD0(transition.after_state, target_value, learning_rate, params);
            sum_weight++;
            trace.pop();
        }
    }

    // Stop the clock, Carol
    auto end_time   = std::chrono::high_resolution_clock::now();
    auto delta_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Log data to log file
    log  << epoch << ','
         << num_games << ','
         << std::setprecision(4) << delta_time << ','
         << std::setprecision(4) << (sum_loss / sum_weight) << ','
         << std::setprecision(4) << learning_rate << std::endl;

    std::cout << std::endl << "Epoch " << epoch << " training   completed in "
              << std::setprecision(4) << (delta_time / 1000.f) << " seconds..." << std::endl;

    return (sum_loss / sum_weight);
}

void Agent::evaluate_agent(const int epoch, const int num_games, const int depth, const Agent::Params &params, std::ostream &log) {

    // Start the clock, Carol
    auto start_time = std::chrono::high_resolution_clock::now();

    // Variables for accumulating statistics
    float avg_score = 0, maximum_score = 0, avg_tile  = 0, maximum_tile  = 0, 
          tile_128  = 0, tile_256  = 0, tile_512  = 0, tile_1024  = 0,
          tile_2048 = 0, tile_4096 = 0, tile_8192 = 0, tile_16384 = 0, tile_32768 = 0;

    // Parallelize over independent games, reduce statistics between threads to avoid race conditions
    #pragma omp parallel for schedule(dynamic, 1) num_threads(Agent::CPU_THREADS) \
                             reduction(max : maximum_tile) \
                             reduction( +  : avg_score,avg_tile,tile_128,tile_256,tile_512,tile_1024,\
                                             tile_2048,tile_4096,tile_8192,tile_16384,tile_32768)
    for (int game = 0; game < num_games; game++) {
        
        // Play a single game
        float score = 0;
        Game::State state = Game::place_random_tile(0, Game::rand_state(), Game::rand_tile());

        for (int step = 0;; step++) {

            // Expectimax search 1-ply for best local action
            const Game::Transition transition = Agent::expectimax_search_max_transition(state, depth, params);

            // Game score is sum of transition rewards
            score += transition.reward;

            // Test if action ended the game
            if (transition.terminal) break;

            // Place a random tile
            const Game::State new_state = Game::place_random_tile(transition.after_state, Game::rand_state(), Game::rand_tile());

            // If placing a tile puts the game into a terminal state then end the game
            if ((state == new_state) || Game::terminal(new_state)) break;

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
         << depth << ','
         << std::setprecision(4) << delta_time << ','
         << (int) maximum_score << ','
         << (int) (avg_score  / (float) num_games) << ','
         << (int) std::pow(2, maximum_tile) << ','
         << (int) std::pow(2, avg_tile  / (float) num_games) << ','
         << std::setprecision(4) << ((float) tile_128   / (float) num_games) << ','
         << std::setprecision(4) << ((float) tile_256   / (float) num_games) << ','
         << std::setprecision(4) << ((float) tile_512   / (float) num_games) << ','
         << std::setprecision(4) << ((float) tile_1024  / (float) num_games) << ','
         << std::setprecision(4) << ((float) tile_2048  / (float) num_games) << ','
         << std::setprecision(4) << ((float) tile_4096  / (float) num_games) << ','
         << std::setprecision(4) << ((float) tile_8192  / (float) num_games) << ','
         << std::setprecision(4) << ((float) tile_16384 / (float) num_games) << ','
         << std::setprecision(4) << ((float) tile_32768 / (float) num_games) << std::endl;

    std::cout << "Epoch " << epoch << " evaluation completed in "
              << std::setprecision(4) << (delta_time / 1000.f) << " seconds..." << std::endl;
}

std::ofstream Agent::log_evaluation_csv(const std::string &path) {
    std::ofstream log(path, std::ios_base::app);
    if (log.is_open() && (log.tellp() == 0)) {
        log << "epoch,num_games,depth,time,max_score,avg_score,max_tile,avg_tile,"
            << "tile_128,tile_256,tile_512,tile_1024,tile_2048,tile_4096,"
            << "tile_8192,tile_16384,tile_32768" << std::endl;
    }
    return log;
}

std::ofstream Agent::log_training_csv(const std::string &path) {
    std::ofstream log(path, std::ios_base::app);
    if (log.is_open() && (log.tellp() == 0)) {
        log << "epoch,num_games,time,loss,learning_rate" << std::endl;
    }
    return log;
}

void Agent::save(const std::string &path, const Agent::Params &params) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return;
    params.table_0.save(file);
    params.table_1.save(file);
    params.table_2.save(file);
    params.table_3.save(file);
}

void Agent::load(const std::string &path, Agent::Params &params) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return;
    params.table_0.load(file);
    params.table_1.load(file);
    params.table_2.load(file);
    params.table_3.load(file);
}