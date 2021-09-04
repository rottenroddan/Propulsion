//
// Created by steve on 9/3/2021.
//

#include "../Propulsion.cuh"

Propulsion::ArtificialNeuralNetwork::ActivationFunctions::ActivationFunctions()
{
    this->outputLayer = std::make_shared<Tensor<double>>(1,1);
}

std::shared_ptr<Propulsion::Tensor<double>> Propulsion::ArtificialNeuralNetwork::ActivationFunctions::getOutputLayer()
{
    return this->outputLayer;
}

void Propulsion::ArtificialNeuralNetwork::ActivationFunctions::printOutputLayer()
{
    this->outputLayer->print();
}