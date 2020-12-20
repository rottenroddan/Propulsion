//
// Created by steve on 7/16/2020.
//


#pragma once
#include "Propulsion.cuh"


template <typename type> __global__ void deviceAdd2DMatricesWithStride(type **dev_a, type **dev_b, type **dev_c, unsigned ROWS, unsigned COLS)
{
    printf("[%d,%d,%d] : %d \n",blockDim.x,blockIdx.x,threadIdx.x, dev_a[0]);
}

template<typename type>
void Propulsion::cudaAdd2DMatrices(type **a, type **b, type **c, unsigned int rSize, unsigned int cSize)
{

}





template<typename type>
void Propulsion::hostAdd2DMatrices(type **a, type **b, type **c, unsigned int rSize, unsigned int cSize)
{
    // Create starting point for timer
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    for(unsigned i = 0; i < rSize; i++)
    {
        for(unsigned j = 0; j < cSize; j++)
        {
            c[i][j] = a[i][j] + b[i][j];
        }
    }

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    float milliseconds = (float)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() / 1000;

    std::cout << std::left << std::setw(TIME_FORMAT) << " HOST:  2D Matrix Addition " <<
              std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
              " ms." << std::setw(TIME_WIDTH) << (cSize*sizeof(type))/milliseconds/1e6 << " GB/s" << std::endl;
    return;
}