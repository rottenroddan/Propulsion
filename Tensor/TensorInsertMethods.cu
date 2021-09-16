//
// Created by steve on 9/7/2021.
//

#include "../Propulsion.cuh"

template<typename type>
void Propulsion::Tensor<type>::push_back(const Matrix<type>& m)
{
    // Check if empty, that way we can create a new dims object.
    if(this->isEmpty())
    {
        std::deque<unsigned long long> newDims = {1, m.getRowSize(), m.getColSize()};

        this->tensor.push_back(std::make_shared<Matrix<type>>(m));
        this->reshape(newDims);
    }
    else
    {
        auto rowColDim = getRowColDimension();
        unsigned long long rows = rowColDim[0];
        unsigned long long cols = rowColDim[1];

        if(rows == m.getRowSize() && cols == m.getColSize())
        {
            // Since a new Matrix is being pushed back, the dims must now conform
            // to standard size to match any size.
            std::deque<unsigned long long> newDims = {this->getTotalMatrices()+1, rows,cols};
            this->tensor.push_back(std::make_shared<Matrix<type>>(m));
            this->reshape(newDims);
        }
        else
        {
            /// TODO: throw exception.
        }
    }
}