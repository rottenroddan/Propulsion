//
// Created by steve on 8/19/2021.
//

#include "../Propulsion.cuh"

std::shared_ptr<Propulsion::Tensor<double>> Propulsion::ArtificialNeuralNetwork::Layer::getBiases() {
    return this->biases;
}

std::shared_ptr<Propulsion::Tensor<double>> Propulsion::ArtificialNeuralNetwork::Layer::getOutputLayer() {
    return this->outputLayer;
}

std::shared_ptr<Propulsion::Tensor<double>> Propulsion::ArtificialNeuralNetwork::Layer::getWeights() {
    return this->weights;
}

void Propulsion::ArtificialNeuralNetwork::Layer::printBiases() {
    this->biases->print();
}

void Propulsion::ArtificialNeuralNetwork::Layer::printOutputLayer() {
    this->outputLayer->print();
}

void Propulsion::ArtificialNeuralNetwork::Layer::printWeights() {
    this->weights->print();
}