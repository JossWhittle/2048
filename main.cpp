#include "game.h"
#include "agent.h"

int main() {

    constexpr int   TRAIN_GAMES   = 30000;
    constexpr int   TRAIN_EPOCHS  = 1000;
    constexpr float LEARNING_RATE = 1e-2;

    Agent::Params params;
    //Agent::load("./logs/params.bin", params);

    std::ofstream log_train = Agent::log_training_csv("./logs/log_train.csv");
    std::ofstream log_eval  = Agent::log_evaluation_csv("./logs/log_eval.csv");

    for (int epoch = 0; epoch < TRAIN_EPOCHS; epoch++) {

        // Perform one epoch of training
        Agent::train_agent(epoch, TRAIN_GAMES, LEARNING_RATE, params, log_train);
        Agent::save("./logs/params.bin", params);

        Agent::evaluate_agent(epoch, 1000, 1, params, log_eval);
        Agent::evaluate_agent(epoch,  100, 2, params, log_eval);

        if (epoch % 10 == 0) {
            Agent::evaluate_agent(epoch,  10, 3, params, log_eval);
        }
    }

    return 0;
}