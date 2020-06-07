#include <sstream>
#include <string>

#include "game.h"
#include "agent.h"

#define STRING(exp) ([&]() { std::stringstream sstr; sstr << exp; return sstr.str(); }())

int main() {

    constexpr int   TRAIN_GAMES   = 50000;
    constexpr int   TRAIN_EPOCHS  = 50;
    constexpr float LEARNING_RATE = (1e-2); // / 1.5;

    std::ofstream log_train = Agent::log_training_csv("./logs/log_train.csv");
    std::ofstream log_eval  = Agent::log_evaluation_csv("./logs/log_eval.csv");

    Agent::PhaseParams params;

    // Start from the last phase and work backwards towards 2048
    for (int phase = 0, epoch = 0; phase <= Agent::NUM_PHASES; phase++) {

        const int end_phase = std::min((phase + 2), Agent::END_PHASE);

        for (int phase_epoch = 0; phase_epoch < TRAIN_EPOCHS; phase_epoch++, epoch++) {

            std::cout << std::endl << "Phase " << phase << " Phase Epoch " << phase_epoch << " Epoch " << epoch << std::endl;

            // Perform one epoch of training
            Agent::train_agent(epoch, TRAIN_GAMES, phase, end_phase, LEARNING_RATE, params, log_train);

            // Save the current params for all phases greater than and including the current phase
//            for (int i = 0; i <= phase; i++) {
//                Agent::save(STRING( "./logs/params phase_" << phase << "_" << i << ".bin" ), params[i]);
//            }

            // Play evaluation games and log stats
            Agent::evaluate_agent(epoch, 1000, 0, end_phase, 1, params, log_eval);
            Agent::evaluate_agent(epoch, 100,  0, end_phase, 2, params, log_eval);
        }
    }

    return 0;
}