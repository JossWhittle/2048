#include <sstream>
#include <string>

#include "game.h"
#include "agent.h"

int main() {

    constexpr int   TRAIN_GAMES   = 30000;
    constexpr int   TRAIN_EPOCHS  = 10;
    constexpr float LEARNING_RATE = 1e-2;

    std::ofstream log_train = Agent::log_training_csv("./logs/log_train.csv");
    std::ofstream log_eval  = Agent::log_evaluation_csv("./logs/log_eval.csv");

    Agent::PhaseParams params;

    for (int phase = 0, epoch = 0; phase < Agent::NUM_PHASES; phase++) {

        if (phase > 0) {
            // Promote weights
            params[phase].table_0.promote(params[phase - 1].table_0);
            params[phase].table_1.promote(params[phase - 1].table_1);
            params[phase].table_2.promote(params[phase - 1].table_2);
            params[phase].table_3.promote(params[phase - 1].table_3);
        }

        for (int phase_epoch = 0; phase_epoch < TRAIN_EPOCHS; phase_epoch++, epoch++) {

            std::cout << std::endl << "Epoch " << epoch << ", Phase " << phase << ", Phase Epoch " << phase_epoch << std::endl;

            // Perform one epoch of training
            Agent::train_agent(epoch, TRAIN_GAMES, phase, (phase + 1), LEARNING_RATE, params, log_train);

            Agent::save([&]() {
                std::stringstream sstr;
                sstr << "./logs/params_" << phase << ".bin";
                return sstr.str();
            }(), params[phase]);

            Agent::evaluate_agent(epoch, 1000, 0, (phase + 1), 1, params, log_eval);
            Agent::evaluate_agent(epoch,  100, 0, (phase + 1), 2, params, log_eval);
        }

        if (phase > 0) {
            // Perform one epoch of training on both the current phase and previous phase to fine tune the transition now the current phase is trained
            epoch++;
            Agent::train_agent(epoch, TRAIN_GAMES, (phase - 1), (phase + 1), LEARNING_RATE, params, log_train);
        }
    }

    return 0;
}