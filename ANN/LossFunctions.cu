//
// Created by steve on 1/4/2021.
//

#include "../Propulsion.cuh"

/*
double Propulsion::ArtificialNeuralNetwork::Loss::regularization(Matrix<double> &output, Matrix<double> &y)
{
    return 0.0;
}

bool Propulsion::ArtificialNeuralNetwork::LossCategoricalCrossentropy::calculate(std::shared_ptr<Matrix<double>> y_pred, std::shared_ptr<Matrix<double>> y_true)
{
    Matrix<double> confidences(1,y_true->getColSize());

    // Check if categorical labels. Meaning each classification can only
    // have one correct answer. Like is this one animal a dog/cat/etc.
    if(y_true->getRowSize() == 1)
    {
        // Find from the predicted matrix the index of the truth matrix...
        for(unsigned i = 0; i < y_true->getColSize(); i++)
        {
            unsigned yIndexTruth = (unsigned)y_true->at(i);

            // If statement to check if the truth index is a correct index for the y_pred
            if(yIndexTruth < y_pred->getColSize())
            {
                confidences.at(i) = y_pred->at(i, yIndexTruth);
            }
            else
            {
                std::cout << "Error at LossCategoricalCrossentropy: y_pred index predictions is less than the desired y_true index. Trying to access: " << yIndexTruth << ". y_pred col size: " << y_pred->getColSize() << std::endl;
                return false;
            }
        }
    }


    else if(y_true->getRowSize() == y_pred->getRowSize() && y_true->getColSize() == y_pred->getColSize())
    {
        std::cout << "Loss Debugging" << std::endl;
        y_true->print();
        y_pred->print();

        //confidences = Propulsion::Matrix<double>::sumRows((*y_pred) * (*y_true));
        confidences = (*y_pred) * (*y_true);
    }
    else
    {
        std::cout << "Error at LossCategoricalCrossentropy: y_true is of size: (" << y_true->getRowSize() << ", " << y_true->getColSize() << ") -> Last Neuron column shape doesn't match the output size: " << y_pred->getColSize() << std::endl;
    }

    confidences.print();
    return true;
}*/