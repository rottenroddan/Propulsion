//
// Created by steve on 11/22/2020.
//
#include "../Propulsion.cuh"

void Propulsion::ArtificialNeuralNetwork::ActivationSoftmax::forward(LayerDense &input)
{
    double sArr[] = { 1,2,3,
                    1,2,3,
                    2,4,6,
                    4,8,12};
    Propulsion::Matrix<double> sInp(sArr,4,3);

    // Store a shared ptr of the output layer.
    //auto inputMatrixPtr = input.getOutputLayer();
    auto inputMatrixPtr = &sInp;

    // Output layer is now init. to the same size of the input ptr.
    outputLayer = std::make_shared<Matrix<double>>(inputMatrixPtr->getRowSize(), inputMatrixPtr->getColSize());

    double max = inputMatrixPtr->getMax();

    // Unnormalized values
    auto expValues = Propulsion::Matrix<double>::subtractBroadScalar(*inputMatrixPtr,max);

    // exponetial now;
    for(unsigned i = 0; i < expValues.getTotalSize(); i++)
    {
        expValues.at(i) = std::exp(expValues.at(i));
    }

    auto rowSumOfExp = Matrix<double>::sumRows(expValues);
    Matrix<double> probabilities(expValues.getRowSize(),expValues.getColSize());

    for(unsigned i = 0; i < rowSumOfExp.getTotalSize(); i++)
    {
        for(unsigned j = 0; j < probabilities.getColSize(); j++)
        {
            probabilities.at(i,j) = expValues.at(i,j) / rowSumOfExp.at(i);
        }
    }

    probabilities.print();

}

void Propulsion::ArtificialNeuralNetwork::ActivationSoftmax::printOutputLayer()
{
    outputLayer->print();
}

