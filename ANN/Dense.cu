//
// Created by steve on 8/19/2021.
//

#include "../Propulsion.cuh"

Propulsion::ArtificialNeuralNetwork::Dense::Dense(unsigned int nInputs, unsigned int nNeurons)
{
    // Create a shared ptr to a Tensor with n inputs by n Neurons.
    this->weights = std::make_shared<Propulsion::Tensor<double>>(nInputs, nNeurons);

    // Set attribute biases to number of nuerons.
    this->biases = std::make_shared<Propulsion::Tensor<double>>(nNeurons);

    // Initialize weights to a random number betweeen -1.0 and 1.0. Multiply by .1 to minimize weights.
    this->weights->populateWithRandomRealDistribution(-1.0, 1.0);
    this->weights->scalarProduct(.10);
}