//
// Created by steve on 3/15/2021.
//

#include <deque>
#include <iostream>
#include <string>

std::string getDimsExceptionString(const std::deque<unsigned long long>& dims)
{
    std::string err = "( ";
    if(dims.size() != 0)
    {
        for(unsigned long long i = 0;i<dims.size()- 1; i++)
        {
            err += std::to_string(dims[i]) + ", ";
        }
        err += std::to_string(dims[dims.size() - 1]);

    }
    err += ")";
    return err;
}

unsigned long long getTotalMatricesFromDims(std::deque<unsigned long long>& dims)
{
    if(dims.size() == 0)
    {
        return 0;
    }
    else if(dims.size() <= 2)
    {
        return 1;
    }
    else
    {
        unsigned long long prod = 1;
        for(unsigned i = 0; i < dims.size()-2; i++)
        {
            prod *= dims[i];
        }
        return prod;
    }
}