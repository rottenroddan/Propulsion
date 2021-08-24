//
// Created by steve on 8/24/2021.
//
#pragma once

template<typename type>
Propulsion::Tensor<type>& Propulsion::Tensor<type>::operator=(const Tensor <type> &rhs)
{
    if(this == &rhs) {return *this;}
    else {
        // Copy dims of rhs.
        this->dims = rhs.dims;
        this->tensor.clear();   // delete this tensor as its contents are no longer needed.

        // If Matrix size is 2 or greater.
        if(this->dims.size() >= 2) {
            unsigned long long rowIdx = this->dims.size() - 2;
            unsigned long long colIdx = this->dims.size() - 1;

            for (unsigned i = 0; i < rhs.tensor.size(); i++) {
                this->tensor.push_back(std::unique_ptr<Propulsion::Matrix<type>>(
                        new Propulsion::Matrix<type>(this->dims[rowIdx], this->dims[colIdx])));
                this->tensor[i]->operator=(*rhs.tensor[i]);
            }
        }
            // else its size of 1.
        else if(this->dims.size() == 1)
        {
            unsigned long long colIdx = this->dims.size() - 1;
            this->tensor.push_back(std::unique_ptr<Propulsion::Matrix<type>>(
                    new Propulsion::Matrix<type>(this->dims[colIdx])));
            this->tensor[0]->operator=(*rhs.tensor[0]);
        }
    }
    return *this;
}


