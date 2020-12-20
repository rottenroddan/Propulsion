//
// Created by steve on 11/1/2020.
//

#include "Propulsion.cuh"

template<typename type>
Propulsion::Matrix<type>* Propulsion::Matrix<type>::backwardSubstitution(Matrix<type> A, Matrix<type> y)
{
    if(A.getRowSize() != y.getRowSize())
        return nullptr;

    if(A.isUpperTriangular())
    {
        Matrix<type> temp = y;
        for(long int i = A.getRowSize() - 1; i >= 0; i--)
        {
            // if singular, return.
            if(A.at(i,i) == 0)
            {
                std::cout << "Matrix provided into backward substitution is singular." << std::endl;
                return nullptr;
            }

            for(unsigned j = A.getColSize() - 1; j >= i; j--)
            {
                temp(i,j) = temp(i) - A(i,j)*temp(j);
            }
            temp(i) = temp(i) / A(i,i);

        }
    }
    else
    {
        std::cout << "Matrix provided into backward substitution is not an upper triangle." << std::endl;
        return nullptr;
    }
}
