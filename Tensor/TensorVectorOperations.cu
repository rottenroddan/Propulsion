//
// Created by steve on 8/23/2021.
//
#pragma once

template<typename type>
void Propulsion::Tensor<type>::addRowVector(Tensor<type> &rowVector)
{
    // Get the total amount of Matrices as we want to this value to be one.
    unsigned long long dimsAboveSecond = rowVector.getTotalMatrices();

    // If its narrowed down to a 1xnxm vector. Let the Matrix class handle the vector portion of the Tensor.
    if(dimsAboveSecond == 1)
    {
        // For every Matrix in this tensor, call addRowVector for the given Matrix.
        for(unsigned i = 0; i < this->tensor.size(); i++)
        {
            try {
                this->tensor[i] = std::make_shared<Matrix<type>>(
                        this->tensor[i]->addRowVector(*rowVector.tensor[0]));
            }
            catch (typename Propulsion::Matrix<type>::MatrixException& mE) {
                // Generate TensorException
                std::string err;
                std::string matrixWhat = mE.what();
                // Using getDimsExceptionString helper function.
                err += "Tensor Size Mismatch in addRowVector, this: " + getDimsExceptionString(this->dims) + " vs " + getDimsExceptionString(rowVector.dims);
                throw Tensor<type>::TensorException(err.c_str(), __FILE__, __LINE__,
                                                    "addRowVector", (std::string("The passed \"Row Vector\" doesn't have the same col size as this Tensor. Matrix Exception: ") + matrixWhat).c_str());
            }
        }
    }
    else
    {
        std::string err;
        err += "Tensor Size Mismatch in addRowVector, this: " + getDimsExceptionString(this->dims) + " vs " + getDimsExceptionString(rowVector.dims);
        throw Tensor<type>::TensorException(err.c_str(), __FILE__, __LINE__,
                                            "addRowVector", "The passed \"Row Vector\" has to have ");
    }
}

template<typename type>
void Propulsion::Tensor<type>::addRowVector(Matrix <type> &rowVector)
{
    for(unsigned i = 0; i < this->getTotalMatrices(); i++)
    {
        try {
            this->tensor[i] = std::make_shared<Matrix<type>>(
                    this->tensor[i]->addRowVector(*rowVector.M)
                    );
        }
        catch (typename Propulsion::Matrix<type>::MatrixException& mE) {
            // Generate TensorException
            std::string err;
            std::string matrixWhat = mE.what();
            // Using getDimsExceptionString helper function.
            err += "Tensor Size Mismatch in addRowVector, this: " + getDimsExceptionString(this->dims) + " vs " + getDimsExceptionString(rowVector.dims);
            throw Tensor<type>::TensorException(err.c_str(), __FILE__, __LINE__,
                                                "addRowVector", (std::string("The passed \"Row Vector\" doesn't have the same col size as this Tensor. Matrix Exception: ") + matrixWhat).c_str());
        }
    }
}

template<typename type>
Propulsion::Tensor<type>& Propulsion::Tensor<type>::addRowVector(Tensor <type> &A, Tensor <type> &rowVector)
{
    Tensor<type> retTensor = A;
    A.addRowVector(rowVector);
    return A;
}