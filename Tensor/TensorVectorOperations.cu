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
    // Get the total amount of Matrices as we want to this value to be one.
    unsigned long long dimsAboveSecond = rowVector.getTotalMatrices();

    // If its narrowed down to a 1xnxm vector. Let the Matrix class handle the vector portion of the Tensor.
    if(dimsAboveSecond == 1) {
        for (unsigned i = 0; i < this->getTotalMatrices(); i++) {
            try {
                this->tensor[i] = std::make_shared<Matrix < type>>
                (
                        this->tensor[i]->addRowVector(*rowVector.M)
                );
            }
            catch (typename Propulsion::Matrix<type>::MatrixException &mE) {
                // Generate TensorException
                std::string err;
                std::string matrixWhat = mE.what();
                // Using getDimsExceptionString helper function.
                err += "Tensor Size Mismatch in addRowVector, this: " + getDimsExceptionString(this->dims) + " vs " +
                       getDimsExceptionString(rowVector.dims);
                throw Tensor<type>::TensorException(err.c_str(), __FILE__, __LINE__,
                                                    "addRowVector", (std::string(
                                "The passed \"Row Vector\" doesn't have the same col size as this Tensor. Matrix Exception: ") +
                                                                     matrixWhat).c_str());
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
Propulsion::Tensor<type> Propulsion::Tensor<type>::addRowVector(Tensor <type> &A, Tensor <type> &rowVector)
{
    Tensor<type> retTensor = A;
    A.addRowVector(rowVector);
    return A;
}

template<typename type>
void Propulsion::Tensor<type>::subtractRowVector(Tensor<type> &rowVector)
{
    // Get the total amount of Matrices as we want to this value to be one.
    unsigned long long dimsAboveSecond = rowVector.getTotalMatrices();

    // If its narrowed down to a 1xnxm vector. Let the Matrix class handle the vector portion of the Tensor.
    if(dimsAboveSecond == 1)
    {
        // For every Matrix in this tensor, call subtractRowVector for the given Matrix.
        for(unsigned i = 0; i < this->tensor.size(); i++)
        {
            try {
                this->tensor[i] = std::make_shared<Matrix<type>>(
                        this->tensor[i]->subtractRowVector(*rowVector.tensor[0]));
            }
            catch (typename Propulsion::Matrix<type>::MatrixException& mE) {
                // Generate TensorException
                std::string err;
                std::string matrixWhat = mE.what();
                // Using getDimsExceptionString helper function.
                err += "Tensor Size Mismatch in subtractRowVector, this: " + getDimsExceptionString(this->dims) + " vs " + getDimsExceptionString(rowVector.dims);
                throw Tensor<type>::TensorException(err.c_str(), __FILE__, __LINE__,
                                                    "subtractRowVector", (std::string("The passed \"Row Vector\" doesn't have the same col size as this Tensor. Matrix Exception: ") + matrixWhat).c_str());
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
void Propulsion::Tensor<type>::subtractRowVector(Matrix <type> &rowVector)
{
    for (unsigned i = 0; i < this->getTotalMatrices(); i++) {
        try {
            this->tensor[i] = std::make_shared<Matrix < type>>
            (
                    this->tensor[i]->subtractRowVector(*rowVector.M)
            );
        }
        catch (typename Propulsion::Matrix<type>::MatrixException &mE) {
            // Generate TensorException
            std::string err;
            std::string matrixWhat = mE.what();
            // Using getDimsExceptionString helper function.
            err += "Tensor Size Mismatch in subtractRowVector, this: " + getDimsExceptionString(this->dims) + " vs " +
                   getDimsExceptionString(rowVector.dims);
            throw Tensor<type>::TensorException(err.c_str(), __FILE__, __LINE__,
                                                "subtractRowVector", (std::string(
                            "The passed \"Row Vector\" doesn't have the same col size as this Tensor. Matrix Exception: ") +
                                                                 matrixWhat).c_str());
        }
    }
}

template<typename type>
Propulsion::Tensor<type> Propulsion::Tensor<type>::subtractRowVector(Tensor <type> &A, Tensor <type> &rowVector)
{
    Tensor<type> retTensor = A;
    A.subtractRowVector(rowVector);
    return A;
}

template<typename type>
void Propulsion::Tensor<type>::addColVector(Tensor<type> &colVector)
{
    // Get the total amount of Matrices as we want to this value to be one.
    unsigned long long dimsAboveSecond = colVector.getTotalMatrices();

    // If its narrowed down to a 1xnxm vector. Let the Matrix class handle the vector portion of the Tensor.
    if(dimsAboveSecond == 1)
    {
        // For every Matrix in this tensor, call addRowVector for the given Matrix.
        for(unsigned i = 0; i < this->tensor.size(); i++)
        {
            try {
                this->tensor[i] = std::make_shared<Matrix<type>>(
                        this->tensor[i]->addColVector(*colVector.tensor[0]));
            }
            catch (typename Propulsion::Matrix<type>::MatrixException& mE) {
                // Generate TensorException
                std::string err;
                std::string matrixWhat = mE.what();
                // Using getDimsExceptionString helper function.
                err += "Tensor Size Mismatch in addColVector, this: " + getDimsExceptionString(this->dims) + " vs " + getDimsExceptionString(colVector.dims);
                throw Tensor<type>::TensorException(err.c_str(), __FILE__, __LINE__,
                                                    "addColVector", (std::string("The passed \"Col Vector\" doesn't have the same row size as this Tensor. Matrix Exception: ") + matrixWhat).c_str());
            }
        }
    }
    else
    {
        std::string err;
        err += "Tensor Size Mismatch in addColVector, this: " + getDimsExceptionString(this->dims) + " vs " + getDimsExceptionString(colVector.dims);
        throw Tensor<type>::TensorException(err.c_str(), __FILE__, __LINE__,
                                            "addColVector", "The passed \"Col Vector\" has to have only one matrix.");
    }
}

template<typename type>
void Propulsion::Tensor<type>::addColVector(Matrix <type> &colVector)
{
    for(unsigned i = 0; i < this->getTotalMatrices(); i++)
    {
        try {
            this->tensor[i] = std::make_shared<Matrix<type>>(
                    this->tensor[i]->addColVector(*colVector.M)
            );
        }
        catch (typename Propulsion::Matrix<type>::MatrixException& mE) {
            // Generate TensorException
            std::string err;
            std::string matrixWhat = mE.what();
            // Using getDimsExceptionString helper function.
            err += "Tensor Size Mismatch in addColVector, this: " + getDimsExceptionString(this->dims) + " vs " + getDimsExceptionString(colVector.dims);
            throw Tensor<type>::TensorException(err.c_str(), __FILE__, __LINE__,
                                                "addColVector", (std::string("The passed \"Col Vector\" doesn't have the same row size as this Tensor. Matrix Exception: ") + matrixWhat).c_str());
        }
    }
}

template<typename type>
Propulsion::Tensor<type> Propulsion::Tensor<type>::addColVector(Tensor <type> &A, Tensor <type> &colVector)
{
    Tensor<type> retTensor = A;
    A.addColVector(colVector);
    return A;
}

template<typename type>
void Propulsion::Tensor<type>::subtractColVector(Tensor<type> &colVector)
{
    // Get the total amount of Matrices as we want to this value to be one.
    unsigned long long dimsAboveSecond = colVector.getTotalMatrices();

    // If its narrowed down to a 1xnxm vector. Let the Matrix class handle the vector portion of the Tensor.
    if(dimsAboveSecond == 1)
    {
        // For every Matrix in this tensor, call addRowVector for the given Matrix.
        for(unsigned i = 0; i < this->tensor.size(); i++)
        {
            try {
                this->tensor[i] = std::make_shared<Matrix<type>>(
                        this->tensor[i]->subtractColVector(*colVector.tensor[0]));
            }
            catch (typename Propulsion::Matrix<type>::MatrixException& mE) {
                // Generate TensorException
                std::string err;
                std::string matrixWhat = mE.what();
                // Using getDimsExceptionString helper function.
                err += "Tensor Size Mismatch in subtractColVector, this: " + getDimsExceptionString(this->dims) + " vs " + getDimsExceptionString(colVector.dims);
                throw Tensor<type>::TensorException(err.c_str(), __FILE__, __LINE__,
                                                    "subtractColVector", (std::string("The passed \"Col Vector\" doesn't have the same row size as this Tensor. Matrix Exception: ") + matrixWhat).c_str());
            }
        }
    }
    else
    {
        std::string err;
        err += "Tensor Size Mismatch in subtractColVector, this: " + getDimsExceptionString(this->dims) + " vs " + getDimsExceptionString(colVector.dims);
        throw Tensor<type>::TensorException(err.c_str(), __FILE__, __LINE__,
                                            "subtractColVector", "The passed \"Col Vector\" has to have only one matrix.");
    }
}

template<typename type>
void Propulsion::Tensor<type>::subtractColVector(Matrix <type> &colVector)
{
    for(unsigned i = 0; i < this->getTotalMatrices(); i++)
    {
        try {
            this->tensor[i] = std::make_shared<Matrix<type>>(
                    this->tensor[i]->subtractColVector(*colVector.M)
            );
        }
        catch (typename Propulsion::Matrix<type>::MatrixException& mE) {
            // Generate TensorException
            std::string err;
            std::string matrixWhat = mE.what();
            // Using getDimsExceptionString helper function.
            err += "Tensor Size Mismatch in subtractColVector, this: " + getDimsExceptionString(this->dims) + " vs " + getDimsExceptionString(colVector.dims);
            throw Tensor<type>::TensorException(err.c_str(), __FILE__, __LINE__,
                                                "subtractColVector", (std::string("The passed \"Col Vector\" doesn't have the same row size as this Tensor. Matrix Exception: ") + matrixWhat).c_str());
        }
    }
}

template<typename type>
Propulsion::Tensor<type> Propulsion::Tensor<type>::subtractColVector(Tensor <type> &A, Tensor <type> &colVector)
{
    Tensor<type> retTensor = A;
    A.subtractColVector(colVector);
    return A;
}