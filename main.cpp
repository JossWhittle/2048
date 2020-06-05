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
            Agent::load([&]() {
                std::stringstream sstr;
                sstr << "./logs/params_" << (phase - 1) << ".bin";
                return sstr.str();
            }(), params[phase]);
        }

        for (int phase_epoch = 0; phase_epoch < TRAIN_EPOCHS; phase_epoch++, epoch++) {

            // Perform one epoch of training
            Agent::train_agent(epoch, TRAIN_GAMES, phase, LEARNING_RATE, params, log_train);

            Agent::save([&]() {
                std::stringstream sstr;
                sstr << "./logs/params_" << phase << ".bin";
                return sstr.str();
            }(), params[phase]);

            Agent::evaluate_agent(epoch, 1000, phase, 1, params, log_eval);
            Agent::evaluate_agent(epoch,  100, phase, 2, params, log_eval);
            //Agent::evaluate_agent(epoch,  16, phase, 3, params, log_eval);
        }
    }

    return 0;
}