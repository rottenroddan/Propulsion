//
// Created by steve on 3/15/2021.
//

#include <string>
#include <deque>
#include <iostream>

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