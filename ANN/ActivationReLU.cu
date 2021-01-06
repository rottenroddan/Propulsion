//
// Created by steve on 11/21/2020.
//
#include "Propulsion.cuh"
#define RELU_LOWER_CONST 0

void Propulsion::ArtificialNeuralNetwork::ActivationReLU::forward(LayerDense &input)
{
    const double RELU_ZERO = 0.0;

    // Just store a shared pointer of inputMatrix
    auto inputMatrixPtr = input.getOutputLayer();

    // Output layer is now initialized to the same size of the input ptr.
    outputLayer = std::make_shared<Matrix<double>>(inputMatrixPtr->getRowSize(), inputMatrixPtr->getColSize());

    // For each i in Input, we find the Max of 0 or that value.
    for(unsigned i = 0; i < outputLayer->getTotalSize(); i++)
    {
        outputLayer->at(i) = (inputMatrixPtr->at(i) > RELU_ZERO ? inputMatrixPtr->at(i) : RELU_ZERO);
    }
}

void Propulsion::ArtificialNeuralNetwork::ActivationReLU::printOutputLayer()
{
    this->outputLayer->print();
}
