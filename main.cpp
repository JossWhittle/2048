#include <sstream>
#include <string>

#include "game.h"
#include "agent.h"

#define STRING(exp) ([&]() { std::stringstream sstr; sstr << exp; return sstr.str(); }())

int main() {

    constexpr int   TRAIN_GAMES   = 25000;
    constexpr int   TRAIN_EPOCHS  = 50;
    constexpr float LEARNING_RATE = (1e-2); // / 1.5;

    std::ofstream log_train = Agent::log_training_csv("./logs/log_train.csv");
    std::ofstream log_eval  = Agent::log_evaluation_csv("./logs/log_eval.csv");

    Agent::PhaseParams params;

    // Start from the last phase and work backwards towards 2048
    for (int phase = 0, epoch = 0; phase <= Agent::NUM_PHASES; phase++) {
        for (int phase_epoch = 0; phase_epoch < TRAIN_EPOCHS; phase_epoch++, epoch++) {

            std::cout << std::endl << "Phase " << phase << " Phase Epoch " << phase_epoch << " Epoch " << epoch << std::endl;

            // Perform one epoch of training
            Agent::train_agent(epoch, TRAIN_GAMES, phase, Agent::END_PHASE, LEARNING_RATE, params, log_train);

            // Play evaluation games and log stats
            Agent::evaluate_agent(epoch, 1000, 0, Agent::END_PHASE, 1, params, log_eval);
            Agent::evaluate_agent(epoch, 100,  0, Agent::END_PHASE, 2, params, log_eval);
        }

        Agent::save("./logs/params.bin", params);
    }

    return 0;
}