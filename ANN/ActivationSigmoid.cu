//
// Created by steve on 11/22/2020.
//
#include "Propulsion.cuh"

void Propulsion::ArtificialNeuralNetwork::ActivationSigmoid::forward(LayerDense &input)
{
    // Store a shared pointer of inputMatrix
    auto inputMatrixPtr = input.getOutputLayer();

    outputLayer = std::make_shared<Matrix<double>>(inputMatrixPtr->getRowSize(), inputMatrixPtr->getColSize());

    for(unsigned i = 0; i < outputLayer->getTotalSize(); i++)
    {
        outputLayer->at(i) = 1 / ( 1 + std::exp(inputMatrixPtr->at(i)));
    }
}

void Propulsion::ArtificialNeuralNetwork::ActivationSigmoid::printOutputLayer()
{
    this->outputLayer->print();
}