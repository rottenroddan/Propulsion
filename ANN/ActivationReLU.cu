//
// Created by steve on 11/21/2020.
//
#include "../Propulsion.cuh"
#define RELU_LOWER_CONST 0

/*
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
}*/

void Propulsion::ArtificialNeuralNetwork::ReLU::forward(Layer &input)
{
    // Copy the contents of output layer from input into this output layer,
    // that way it can be ReLU activated in the next steps.
    *this->outputLayer = *input.getOutputLayer();

    for(unsigned long long i = 0; i < this->outputLayer->getTotalMatrices(); i++)
    {
        for(unsigned long long j = 0; j < this->outputLayer->matrixAt(i)->getTotalSize(); j++)
        {
            // IF value is > 0, don't change. Else it is zero.
            this->outputLayer->matrixAt(i)->at(j) = this->outputLayer->matrixAt(i)->at(j) > 0 ?
                    this->outputLayer->matrixAt(i)->at(j) : 0;
        }
    }
}
