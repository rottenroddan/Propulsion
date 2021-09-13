//
// Created by steve on 9/8/2021.
//

#include "../Propulsion.cuh"

template<typename type>
bool Propulsion::Tensor<type>::reshape(std::deque<unsigned long long> reshapeDims)
{
    // Desired shape is empty. Edge case.
    if(reshapeDims.size() == 0) {
        return false;
    }

    // Bool for checking Matrix shape size.
    bool goodMatrixShape = false;

    // Get desired Matrices and current Matrices stored.
    unsigned long long desiredMatrices = getTotalMatricesFromDims(reshapeDims);
    unsigned long long currentMatrices = this->getTotalMatrices();

    // Set to one in case reshape dims doesn't have the size for it.
    unsigned long long desiredRows = 1;
    unsigned long long desiredCols = 1;

    // Check if size is 1, or 2 to N.
    if(reshapeDims.size() == 1) {
        desiredCols = reshapeDims[0];
    } else {
        desiredRows = reshapeDims[reshapeDims.size()-2];
        desiredCols = reshapeDims[reshapeDims.size()-1];
    }

    // If the reshape is possible for the Matrix.
    if(this->tensor[0]->getRowSize() * this->tensor[0]->getColSize() == desiredRows * desiredCols)
    {
        goodMatrixShape = true;
    }
    else
    {
        return false;
    }

    // Debug purposes.
    std::cout << "Desired: " << desiredMatrices << std::endl;
    std::cout << "Current: " << currentMatrices << std::endl;

    // Check that they're the same size, that way the tensor
    // data is guaranteed to not be wasted.
    if(desiredMatrices == currentMatrices && goodMatrixShape)
    {
        // Change all Matrices to desired size.
        for(unsigned long long i = 0; i < this->tensor.size(); i++)
        {
            this->tensor[i]->reshape(desiredRows, desiredCols);
        }

        // Set dims to the desired dims.
        this->dims = reshapeDims;

        return true;
    }

    return false;
}