//
// Created by steve on 11/20/2020.
// Should be a small file describing the LayerDense class for ANN.
//
#include "Propulsion.cuh"
#include "../Propulsion.cuh"


Propulsion::ArtificialNeuralNetwork::LayerDense::LayerDense(unsigned nInputs, unsigned nNeurons)
{
    // Create the weights layer with N->Inputs(each sample) and N->Neurons(total neurons in this layer.
    // Also note the weights are set up this way so we don't have to take transpose later.
    weights = std::make_shared<Matrix<double>>(nInputs, nNeurons);

    // Create the bias layer with 1 row and N->Neurons columns. Since each neuron has some bias.
    biases = std::make_shared<Matrix<double>>(1,nNeurons,Matrix<double>::MatrixInitVal::zero);

    Propulsion::Matrix<double>::randomRealDistribution( weights, -1, 1);

    // multiply by 0.10 to minimize the random values.
    //weights->multiply(0.10);

}

void Propulsion::ArtificialNeuralNetwork::LayerDense::forward(Propulsion::ArtificialNeuralNetwork::LayerDense &inputs)
{
    //auto dotProduct = (*inputs.outputLayer) * (*weights);
    this->outputLayer = std::make_shared<Matrix<double>>(((*inputs.outputLayer) * (*weights)).addRowVector(*biases) );
}

void Propulsion::ArtificialNeuralNetwork::LayerDense::forward(Propulsion::Matrix<double> &inputs)
{
    //auto dotProduct = inputs * (*weights);
    this->outputLayer = std::make_shared<Matrix<double>>((inputs * (*weights)).addRowVector(*biases) );
}




void Propulsion::ArtificialNeuralNetwork::LayerDense::printWeights()
{
    this->weights->print();
}

void Propulsion::ArtificialNeuralNetwork::LayerDense::printBiases()
{
    this->biases->print();
}

void Propulsion::ArtificialNeuralNetwork::LayerDense::printOutputLayer()
{
    this->outputLayer->print();
}

std::shared_ptr<Propulsion::Matrix < double>> Propulsion::ArtificialNeuralNetwork::LayerDense::getWeights() {
    return this->weights;
}

std::shared_ptr<Propulsion::Matrix < double>> Propulsion::ArtificialNeuralNetwork::LayerDense::getBiases() {
    return this->biases;
}

std::shared_ptr<Propulsion::Matrix < double>> Propulsion::ArtificialNeuralNetwork::LayerDense::getOutputLayer() {
    return this->outputLayer;
}
