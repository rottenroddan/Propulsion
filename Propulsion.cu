//
// Created by steve on 7/7/2020.
//
#pragma once
#include "Propulsion.cuh"

__global__ void deviceHelloWorld()
{
    if(threadIdx.x == 1023)
        printf("Hello World! From thread [%d,%d]!\n", blockIdx.x , threadIdx.x);
}

void Propulsion::helloWorld() {

    int b,t;
    b = 10240;
    t = 1024;

    deviceHelloWorld<<<b,t>>>();
    cudaDeviceSynchronize();
    return;
}

template<typename type>__global__ void deviceAddVector(type *d_a, type *d_b, type *d_c, long long col_size, long long row_size)
{
    /*int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // if the
    if( (row * col) < col_size * row_size) {
        d_c[col*col_size + row] = d_a[col*col_size + row] + d_b[col*col_size + row];
    }*/
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < row_size && j < col_size)
    {
        d_c[j*row_size + i] =  d_a[j*row_size + i] +  d_b[j*row_size + i];
    }

}




template<typename type> __global__ void deviceAdd1DMatrices(type *dev_a, type *dev_b, type *dev_c, unsigned C)
{
    unsigned tID = blockDim.x * blockIdx.x + threadIdx.x;
    if (tID < C)
    {
        dev_c[tID] = dev_a[tID] + dev_b[tID];
    }
}

template<typename type>
void Propulsion::cudaAdd1DArrays(type *a, type *b, type *c, unsigned C, bool printTime) {
    /*if(C > CUDA_NO_STRIDE_MAX_THREADS)
    {
        std::cout << "--------cudaAdd1DMatrice requires the array size less than 65536---------" << std::endl;
        return;
    }*/
    // Start Timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Create device array pointers.
    type *dev_a, *dev_b, *dev_c;

    // Create memory for dev_a/b/c... It is created via R(rows) * C(columns) * type(size of type like int, float, etc).
    cudaMalloc((void**) &dev_a, C * sizeof(type));
    cudaMalloc((void**) &dev_b, C * sizeof(type));
    cudaMalloc((void**) &dev_c, C * sizeof(type));

    // Copy data from host to device.
    cudaMemcpy(dev_a, a, C * sizeof(type), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, C * sizeof(type), cudaMemcpyHostToDevice);

    // Generate block. Based off total size of the array.
    // Block = The ceiling of array_size / 1024.
    // 1024 is the max thread amount. Therefore, we can split the workload up into blocks of threads.
    int block = C/(MAX_THREADS) + 1;

    cudaEventRecord(start);
    // Start kernel if < 65536
    deviceAdd1DMatrices<<<block,MAX_THREADS>>>(dev_a, dev_b, dev_c, C);
    cudaThreadSynchronize(); // Synchronize CUDA with HOST. Wait for Device.

    cudaEventRecord(stop);

    // Copy the data from device c to c.
    cudaMemcpy(c, dev_c, C*sizeof(type), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);


    // Free Memory in Device.
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);


    if(printTime){
        std::cout << std::left << std::setw(TIME_FORMAT) << " CUDA:  1D Array Addition: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (C * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return;
}

template <typename type> __global__ void deviceAdd1DMatricesWithStride(type *dev_a, type *dev_b, type *dev_c, unsigned C)
{
    /*for(int i = blockIdx.x * blockDim.x + threadIdx.x;
        i < C;
        i += blockDim.x * gridDim.x)
    {
        dev_c[i] = dev_a[i] + dev_b[i];
    }*/

    int tID = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = tID; i < C; i += stride)
    {
        dev_c[i] = dev_a[i] + dev_b[i];
    }
}

template<typename type>
void Propulsion::cudaAdd1DArraysWithStride(type *a, type *b, type *c, unsigned C, bool printTime)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Create device array pointers.
    type *dev_a, *dev_b, *dev_c;

    // Create memory for dev_a/b/c... It is created via R(rows) * C(columns) * type(size of type like int, float, etc).
    cudaMalloc((void**) &dev_a, C * sizeof(type));
    cudaMalloc((void**) &dev_b, C * sizeof(type));
    cudaMalloc((void**) &dev_c, C * sizeof(type));

    // Copy data from host to device.
    cudaMemcpy(dev_a, a, C * sizeof(type), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, C * sizeof(type), cudaMemcpyHostToDevice);

    int blockSize = MAX_THREADS;
    int numBlocks = (C + blockSize - 1) / blockSize;

    cudaEventRecord(start);
    //deviceAdd1DMatriceWithStride<<<8*numSMs,1024>>>(dev_a, dev_b, dev_c, C);
    deviceAdd1DMatricesWithStride<<<numBlocks,blockSize>>>(dev_a,dev_b,dev_c, C);

    cudaEventRecord(stop);

    // Copy the data from device c to c.
    cudaMemcpy(c, dev_c, C*sizeof(type), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Free Memory in Device.
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    if(printTime){
        std::cout << std::left << std::setw(TIME_FORMAT) << " CUDA:  1D Array Addition with Stride: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (C * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return;

}

template <typename type> __global__ void deviceSubtract1DMatrices(type *dev_a, type *dev_b, type *dev_c, unsigned C)
{
    unsigned tID = blockDim.x * blockIdx.x + threadIdx.x;
    if(tID < C)
    {
        dev_c[tID] = dev_a[tID] - dev_b[tID];
    }
}

template<typename type>
void Propulsion::cudaSubtract1DArrays(type *a, type *b, type *c, unsigned C, bool printTime)
{
    // Start Timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Create device array pointers.
    type *dev_a, *dev_b, *dev_c;

    // Create memory for dev_a/b/c... It is created via R(rows) * C(columns) * type(size of type like int, float, etc).
    cudaMalloc((void**) &dev_a, C * sizeof(type));
    cudaMalloc((void**) &dev_b, C * sizeof(type));
    cudaMalloc((void**) &dev_c, C * sizeof(type));

    // Copy data from host to device.
    cudaMemcpy(dev_a, a, C * sizeof(type), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, C * sizeof(type), cudaMemcpyHostToDevice);

    // Generate block. Based off total size of the array.
    // Block = The ceiling of array_size / 1024.
    // 1024 is the max thread amount. Therefore, we can split the workload up into blocks of threads.
    int block = C/MAX_THREADS + 1;

    cudaEventRecord(start);
    // Start kernel.
    deviceSubtract1DMatrices<<<block,MAX_THREADS>>>(dev_a, dev_b, dev_c, C);
    cudaThreadSynchronize(); // Synchronize CUDA with HOST. Wait for Device.

    cudaEventRecord(stop);

    // Copy the data from device c to c.
    cudaMemcpy(c, dev_c, C*sizeof(type), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);


    // Free Memory in Device.
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    if(printTime){
        std::cout << std::left << std::setw(TIME_FORMAT) << " CUDA:  1D Array Difference: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (C * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return;
}

template <typename type> __global__ void deviceSubtract1DMatricesWithStride(type *dev_a, type *dev_b, type *dev_c, unsigned C)
{
    int tID = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = tID; i < C; i += stride)
    {
        dev_c[i] = dev_a[i] - dev_b[i];
    }
}

template<typename type>
void Propulsion::cudaSubtract1DArraysWithStride(type *a, type *b, type *c, unsigned C, bool printTime)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Create device array pointers.
    type *dev_a, *dev_b, *dev_c;

    // Create memory for dev_a/b/c... It is created via R(rows) * C(columns) * type(size of type like int, float, etc).
    cudaMalloc((void**) &dev_a, C * sizeof(type));
    cudaMalloc((void**) &dev_b, C * sizeof(type));
    cudaMalloc((void**) &dev_c, C * sizeof(type));

    // Copy data from host to device.
    cudaMemcpy(dev_a, a, C * sizeof(type), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, C * sizeof(type), cudaMemcpyHostToDevice);

    int blockSize = MAX_THREADS;
    int numBlocks = (C + blockSize - 1) / blockSize;

    cudaEventRecord(start);
    deviceSubtract1DMatricesWithStride<<<numBlocks,blockSize>>>(dev_a,dev_b,dev_c, C);

    cudaEventRecord(stop);

    // Copy the data from device c to c.
    cudaMemcpy(c, dev_c, C*sizeof(type), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Free Memory in Device.
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    if(printTime){
        std::cout << std::left << std::setw(TIME_FORMAT) << " CUDA:  1D Array Diff. with Stride: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (C * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return;
}

template <typename type> __global__ void deviceMultiply1DMatricesStride(type *dev_a, type *dev_b, type *dev_c, unsigned C)
{
    int tID = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = tID; i < C; i += stride)
    {
        dev_c[i] = dev_a[i] * dev_b[i];
    }
}

template <typename type>
__global__ void deviceMultiplyMatrices(type *dev_a, type *dev_b, type *dev_c, unsigned ROWS, unsigned AcolsBRows,unsigned COLS)
{
    unsigned tRow = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned tCol = blockIdx.y * blockDim.y + threadIdx.y;
    int tempSum = 0;

    if(tRow < ROWS && tCol < COLS)
    {
        for (unsigned i = 0; i < AcolsBRows; i++)
        {
            tempSum += dev_a[tRow * AcolsBRows + i] * dev_b[i * AcolsBRows + tCol];
        }
        dev_c[tRow * COLS + tCol] = tempSum;
    }
}

template<typename type>
void Propulsion::cudaMultiply1DArrays(type *a, type *b, type *c, unsigned ROWS, unsigned AcolsBRows,unsigned COLS, bool printTime)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Create device array pointers.
    type *dev_a, *dev_b, *dev_c;

    // Create memory for dev_a/b/c... It is created via R(rows) * C(columns) * type(size of type like int, float, etc).
    cudaMalloc((void**) &dev_a, ROWS * sizeof(type) * AcolsBRows);
    cudaMalloc((void**) &dev_b, AcolsBRows * sizeof(type) * COLS);
    cudaMalloc((void**) &dev_c, ROWS * sizeof(type) * COLS);

    // Copy data from host to device.
    cudaMemcpy(dev_a, a, ROWS * sizeof(type) * AcolsBRows, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, AcolsBRows * sizeof(type) * COLS, cudaMemcpyHostToDevice);

    // block of threads.
    dim3 block_dim(32,32);

    // total blocks of threads in x direction
    unsigned blocksX = std::ceil( ( (double)COLS) / ( (double)block_dim.x) );
    // total blocks of threads in y direction
    unsigned blocksY = std::ceil( ( (double)ROWS) / ( (double)block_dim.y) );
    dim3 grid_dim(blocksX, blocksY);

    // start timer.
    cudaEventRecord(start);
    deviceMultiplyMatrices<<<grid_dim,block_dim>>>(dev_a,dev_b,dev_c, ROWS, AcolsBRows, COLS);

    cudaEventRecord(stop);

    // Copy the data from device c to c.
    cudaMemcpy(c, dev_c, ROWS*sizeof(type)*COLS, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Free Memory in Device.
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    if(printTime){
        std::cout << std::left << std::setw(TIME_FORMAT) << " CUDA:  Matrix Multiplication: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << std::endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return;
}


template <typename type> __global__ void deviceDivide1DMatricesStride(type *dev_a, type *dev_b, type *dev_c, unsigned C)
{
    int tID = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = tID; i < C; i += stride)
    {
        dev_c[i] = dev_a[i] / dev_b[i];
    }
}


template<typename type>
void Propulsion::cudaDivide1DArrays(type *a, type *b, type *c, unsigned C, bool printTime)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Create device array pointers.
    type *dev_a, *dev_b, *dev_c;

    // Create memory for dev_a/b/c... It is created via R(rows) * C(columns) * type(size of type like int, float, etc).
    cudaMalloc((void**) &dev_a, C * sizeof(type));
    cudaMalloc((void**) &dev_b, C * sizeof(type));
    cudaMalloc((void**) &dev_c, C * sizeof(type));

    // Copy data from host to device.
    cudaMemcpy(dev_a, a, C * sizeof(type), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, C * sizeof(type), cudaMemcpyHostToDevice);

    int blockSize = MAX_THREADS;
    int numBlocks = (C + blockSize - 1) / blockSize;

    // Start cuda timer
    cudaEventRecord(start);

    // Start Division Kernel
    deviceDivide1DMatricesStride<<<numBlocks,blockSize>>>(dev_a,dev_b,dev_c, C);

    // Record Cuda end time.
    cudaEventRecord(stop);

    // Copy the data from device c to c.
    cudaMemcpy(c, dev_c, C*sizeof(type), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Free Memory in Device.
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    if(printTime){
        std::cout << std::left << std::setw(TIME_FORMAT) << " CUDA:  1D Array Division with Stride: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (C * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return;
}

template <typename type> __global__ void deviceMultiplyArrayByScalar(type *dev_a, type s, unsigned C)
{
    int tID = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = tID; i < C; i += stride)
    {
        dev_a[tID] *= s;
    }
}

template<typename type>
void Propulsion::cudaMultiply1DArrayByScalar(type *a, type s, unsigned C, bool printTime) {
    // Create cuda event objects.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // For device array pointer
    type *dev_a;

    // Allocate memory for dev_a based on the size of a
    cudaMalloc((void**) &dev_a, C * sizeof(type));
    // Copy contents of a over to dev_a
    cudaMemcpy(dev_a, a, C * sizeof(type), cudaMemcpyHostToDevice);

    // Create a Kernel that will consist of 1024 Threads along with a block size of (C * 1024 -1) / 1024
    int blockSize = MAX_THREADS;
    int numBlocks = (C + blockSize - 1) / blockSize;

    // Start Timers
    cudaEventRecord(start);
    deviceMultiplyArrayByScalar<<<numBlocks,blockSize>>>(dev_a, s, C);
    cudaEventRecord(stop);

    cudaMemcpy(a,dev_a,C * sizeof(type), cudaMemcpyDeviceToHost);

    // Synchronize cuda event object.
    cudaEventSynchronize(stop);


    // Get time for the operation.
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Free Memory in Device.
    cudaFree(dev_a);

    if(printTime){
        std::cout << std::left << std::setw(TIME_FORMAT) << " CUDA:  1D Array Multiply Scalar: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (C * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return;
}

template <typename type> __global__ void deviceStencilSum1DMatrix(type *d_a, type *d_c, unsigned C, int spacing, int radius)
{
    int tID = threadIdx.x * spacing;
    type tempSum = 0;


    for(int i = tID; (i < (tID + spacing) && i < (signed)C); i++)
    {
        //printf("tID = %d : i = %d : r = %d\n",threadIdx.x, i, radius);
        for(int j = ((i - radius) < 0 ? 0 : (i - radius)) ; j <= i + radius && j < C; j++)
        {
            tempSum += d_a[j];
        }

        d_c[i] = tempSum; // c array at i'th pos equal to sum of the stencil.
        tempSum = 0;    // reset sum
        __syncthreads();
    }
}

template <typename type> __global__ void deviceStencilSum1D(type *d_a, type *d_c, unsigned C, int radius)
{
    int tID = (blockDim.x * blockIdx.x + threadIdx.x) * (radius * 2 + 1);
    type tempSum = 0;

    for(int i = tID; (i < tID + (radius * 2 + 1)) && (i < C); i++ )
    {
        for(int j = (i - radius) < 0 ? 0 : (i - radius); (j <= (radius + i)) && (j < C); j++)
        {
            tempSum += d_a[j];
        }
        d_c[i] = tempSum;
        tempSum = 0;
        __syncthreads();
    }
}

template<typename type>
void Propulsion::cudaStencilSum1DArrays(type *a, type *c, unsigned C, unsigned radius, bool printTime)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int diameter = radius * 2 +1;

    // Create device array pointers.
    type *dev_a, *dev_c;

    // Create memory for dev_a/b/c... It is created via R(rows) * C(columns) * type(size of type like int, float, etc).
    cudaMalloc((void**) &dev_a, C * sizeof(type));
    cudaMalloc((void**) &dev_c, C * sizeof(type));

    // Copy data from host to device.
    cudaMemcpy(dev_a, a, C * sizeof(type), cudaMemcpyHostToDevice);

    // Max Elements Per Thread. E.g.
    //              =>|<=
    //  [ 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8]
    //            3 + 4 + 5  #radius = 1
    //            =  12
    int numBlocks = C / MAX_THREADS + 1;


    // Start cuda timer
    cudaEventRecord(start);

    // Start Division Kernel
    //deviceStencilSum1DMatrix<<<numBlocks, threads>>>(dev_a,dev_c, C, spaceBetweenThreads, radius);
    deviceStencilSum1D<<<numBlocks, MAX_THREADS>>>(dev_a, dev_c, C, radius);

    // Record Cuda end time.
    cudaEventRecord(stop);

    // Copy the data from device c to c.
    cudaMemcpy(c, dev_c, C*sizeof(type), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Free Memory in Device.
    cudaFree(dev_a);
    cudaFree(dev_c);

    if(printTime){
        std::cout << std::left << std::setw(TIME_FORMAT)
                  << " CUDA:  1D Array Stencil(r=" + std::to_string(radius) + ") : " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (diameter * C * sizeof(type)) / milliseconds / 1e6 << " GB/s"
                  << std::endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return;
}

void Propulsion::hostAdd1DArraysInt16AVX256(short *a, short *b, short *c, unsigned C, bool printTime)
{
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    unsigned boundedRange = 0;

    if(C % (short)(AVX256BYTES/(sizeof(short))) == 0)
    {
        boundedRange = C;
    }
    else
    {
        boundedRange = C - (C % (unsigned)(AVX256BYTES/sizeof(short)));
        for(unsigned i = boundedRange; i < C; i++)
        {
            c[i] = a[i] + b[i];
        }
    }

    __m256i _x;
    __m256i _y;
    __m256i _z;

    for(unsigned i = 0; i < boundedRange; i += (unsigned)AVX256BYTES/sizeof(short))
    {
        _x = _mm256_load_si256((__m256i *)&a[i]);
        _y = _mm256_load_si256((__m256i *)&b[i]);
        _z = _mm256_add_epi16(_x,_y);
        _mm256_store_si256((__m256i *)&c[i],_z);
    }

    if(printTime){
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

        std::cout << std::left << std::setw(TIME_FORMAT) << " HOST:  1D Array AVX-256 Addition(Int16): " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (C * sizeof(short)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }
}


void Propulsion::hostAdd1DArraysInt32AVX256(int *a, int *b, int *c, unsigned C, bool printTime) {
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    unsigned boundedRange = 0;

    if(C % (unsigned)(AVX256BYTES/(sizeof(int))) == 0)
    {
        boundedRange = C;
    }
    // else the bounds lie outside of avx256
    else
    {
        boundedRange = C - (C % (unsigned)(AVX256BYTES/(sizeof(int))));
        for(unsigned i = boundedRange; i < C; i++)
        {
            c[i] = a[i] + b[i];
        }
    }

    __m256i _x;
    __m256i _y;
    __m256i _z;

    for(unsigned i = 0; i < boundedRange; i += (unsigned)AVX256BYTES/(sizeof(int)))
    {
        // load the 8 ints into their respective registers.
        _x = _mm256_load_si256((__m256i *)&a[i]);
        _y = _mm256_load_si256((__m256i *)&b[i]);

        // add the two registers together.
        _z = _mm256_add_epi32(_x, _y);

        // store the register contents of z into c array.
        _mm256_store_si256((__m256i *)&c[i],_z);
    }

    if(printTime){
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

        std::cout << std::left << std::setw(TIME_FORMAT) << " HOST:  1D Array AVX-256 Addition(Int32): " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (C * sizeof(int)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }
    return;
}

void Propulsion::hostAdd1DArraysUInt32AVX256(unsigned *a, unsigned *b, unsigned *c, unsigned C, bool printTime)
{
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    unsigned boundedRange = 0;

    if(C % (unsigned)(AVX256BYTES/sizeof(unsigned)) == 0)
    {
        boundedRange = 0;
    }
    // else the bounds lie outside of an even AVX256 ins.
    else
    {
        boundedRange = C - (C % (unsigned)(AVX256BYTES/(sizeof(unsigned ))));
        std::cout << "Bounded Range boi! " << C - (C % (unsigned)(AVX256BYTES/(sizeof(unsigned)))) << std::endl;
        for(unsigned i = boundedRange; i < C; i++)
        {
            c[i] = a[i] + b[i];
        }
    }

    __m256i _x;
    __m256i _y;
    __m256i _z;

    for(unsigned i = 0; i < boundedRange; i += AVX256BYTES/sizeof(unsigned))
    {
        _x = _mm256_load_si256((const __m256i*)&a[i]);
        _y = _mm256_load_si256((const __m256i*)&b[i]);
        _z = _mm256_add_epi32(_x,_y);
        _mm256_store_si256((__m256i *)&c[i], _z);
    }

    if(printTime){
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

        std::cout << std::left << std::setw(TIME_FORMAT) << " HOST:  1D Array AVX-256 Addition(UInt32): " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (C * sizeof(unsigned)) / milliseconds / 1e6 << " GB/s"
                  << std::endl;
    }
}


void Propulsion::hostAdd1DArraysDouble64AVX256(double *a, double *b, double *c, unsigned C, bool printTime)
{
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    unsigned boundedRange = 0;

    if(C % (unsigned)(AVX256BYTES/sizeof(double)) == 0)
    {
        boundedRange = C;
    }
    // else the bounds lie outside of avx256
    else
    {
        boundedRange = C - (C % (unsigned)(AVX256BYTES/(sizeof(double))));
        std::cout << "Bounded Range boi! " << C - (C % (unsigned)(AVX256BYTES/(sizeof(double)))) << std::endl;
        for(unsigned i = boundedRange; i < C; i++)
        {
            c[i] = a[i] + b[i];
        }
    }

    __m256d _x;
    __m256d _y;
    __m256d _z;

    for(unsigned i = 0; i < boundedRange; i += AVX256BYTES/(sizeof(double)))
    {
        _x = _mm256_load_pd(&a[i]);
        _y = _mm256_load_pd(&b[i]);

        _z = _mm256_add_pd(_x,_y);

        _mm256_store_pd(&c[i],_z);
    }

    if(printTime){
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

        std::cout << std::left << std::setw(TIME_FORMAT) << " HOST:  1D Array AVX-256 Addition(Double): " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (C * sizeof(double)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }
}

void Propulsion::hostAdd1DArraysFloat32AVX256(float *a, float *b, float *c, unsigned C, bool printTime)
{
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    unsigned boundedRange = 0;

    if(C % (unsigned)(AVX256BYTES/(sizeof(float))) == 0)
    {
        boundedRange = C;
    }
    // else outside of boundedRange
    else
    {
        boundedRange = C - (C % (unsigned)(AVX256BYTES/(sizeof(float))));
        for(unsigned i = boundedRange; i < C; i++)
        {
            c[i] = a[i] + b[i];
        }
    }

    __m256 _x;
    __m256 _y;
    __m256 _z;

    for(unsigned i = 0; i < boundedRange; i += AVX256BYTES/(sizeof(float)))
    {
        _x = _mm256_load_ps(&a[i]);
        _y = _mm256_load_ps(&b[i]);
        _z = _mm256_add_ps(_x,_y);
        _mm256_store_ps(&c[i],_z);
    }
    if(printTime){
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

        std::cout << std::left << std::setw(TIME_FORMAT) << " HOST:  1D Array AVX-256 Addition(Float): " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (C * sizeof(float)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }
}

/*
 * constexpr is a function that is described in c++17 and beyond. Since this is c++14, and VS/CUDA supports this
 * feature, we are using it that way I don't have to convert the arrays below to void* arrays, and pass them to
 * static cast them later(I want to save a headache or two, and the fact of the matter is that I'm spending already
 * more time on this project then I intended[Which is totally fine!]).
 */
#pragma warning(disable:4984)
template<typename type>
void Propulsion::hostAdd1DArraysAVX256(type *a, type *b, type *c, unsigned C, bool printTime)
{
    if(a == nullptr || b == nullptr || c == nullptr)
    {
        return;
    }


    if constexpr(std::is_same_v<type, int>)
    {
        Propulsion::hostAdd1DArraysInt32AVX256(a,b,c,C,printTime);
    }
    else if constexpr(std::is_same_v<type, unsigned>)
    {
        Propulsion::hostAdd1DArraysUInt32AVX256(a,b,c,C,printTime);
    }
    else if constexpr(std::is_same_v<type, short>)
    {
        Propulsion::hostAdd1DArraysInt16AVX256(a,b,c,C,printTime);
    }
    else if constexpr(std::is_same_v<type, double>)
    {
        Propulsion::hostAdd1DArraysDouble64AVX256(a,b,c,C,printTime);
    }
    else if constexpr(std::is_same_v<type, float>)
    {
        Propulsion::hostAdd1DArraysFloat32AVX256(a,b,c,C,printTime);
    }
    #pragma warning(default:4984)
}


template<typename type>
void Propulsion::hostAdd1DArrays(type *a, type *b, type *c, unsigned C, bool printTime) {
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    for(unsigned i = 0; i < C; i++)
    {
        c[i] = a[i] + b[i];
    }

    if(printTime){
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

        std::cout << std::left << std::setw(TIME_FORMAT) << " HOST:  1D Array Addition: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (C * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }
    return;
}

void Propulsion::hostSubtract1DArraysInt16AVX256(short *a, short *b, short *c, unsigned C, bool printTime)
{
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    unsigned boundedRange = 0;

    if(C % (short)(AVX256BYTES/(sizeof(short))) == 0)
    {
        boundedRange = C;
    }
    else
    {
        boundedRange = C - (C % (unsigned)(AVX256BYTES/sizeof(short)));
        for(unsigned i = boundedRange; i < C; i++)
        {
            c[i] = a[i] + b[i];
        }
    }

    __m256i _x;
    __m256i _y;
    __m256i _z;

    for(unsigned i = 0; i < boundedRange; i += (unsigned)AVX256BYTES/sizeof(short))
    {
        _x = _mm256_load_si256((__m256i *)&a[i]);
        _y = _mm256_load_si256((__m256i *)&b[i]);
        _z = _mm256_sub_epi16(_x,_y);
        _mm256_store_si256((__m256i *)&c[i],_z);
    }

    if(printTime){
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

        std::cout << std::left << std::setw(TIME_FORMAT) << " HOST:  1D Array AVX-256 Difference(Int16): " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (C * sizeof(short)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }
}


void Propulsion::hostSubtract1DArraysInt32AVX256(int *a, int *b, int *c, unsigned C, bool printTime) {
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    /*
    for(unsigned i = 0; i < C; i++)
    {
        c[i] = a[i] + b[i];
    }
    */
    unsigned boundedRange = 0;

    if(C % (unsigned)(AVX256BYTES/(sizeof(int))) == 0)
    {
        boundedRange = C;
    }
        // else the bounds lie outside of avx256
    else
    {
        boundedRange = C - (C % (unsigned)(AVX256BYTES/(sizeof(int))));
        for(unsigned i = boundedRange; i < C; i++)
        {
            c[i] = a[i] + b[i];
        }
    }

    __m256i _x;
    __m256i _y;
    __m256i _z;

    for(unsigned i = 0; i < boundedRange; i += (unsigned)AVX256BYTES/(sizeof(int)))
    {
        // load the 8 ints into their respective registers.
        _x = _mm256_load_si256((__m256i *)&a[i]);
        _y = _mm256_load_si256((__m256i *)&b[i]);

        // add the two registers together.
        _z = _mm256_sub_epi32(_x, _y);

        // store the register contents of z into c array.
        _mm256_store_si256((__m256i *)&c[i],_z);
    }

    if(printTime){
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

        std::cout << std::left << std::setw(TIME_FORMAT) << " HOST:  1D Array AVX-256 Difference(Int32): " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (C * sizeof(int)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }
    return;
}

void Propulsion::hostSubtract1DArraysUInt32AVX256(unsigned *a, unsigned *b, unsigned *c, unsigned C, bool printTime)
{
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    unsigned boundedRange = 0;

    if(C % (unsigned)(AVX256BYTES/sizeof(unsigned)) == 0)
    {
        boundedRange = 0;
    }
        // else the bounds lie outside of an even AVX256 ins.
    else
    {
        boundedRange = C - (C % (unsigned)(AVX256BYTES/(sizeof(unsigned ))));
        std::cout << "Bounded Range boi! " << C - (C % (unsigned)(AVX256BYTES/(sizeof(unsigned)))) << std::endl;
        for(unsigned i = boundedRange; i < C; i++)
        {
            c[i] = a[i] + b[i];
        }
    }

    __m256i _x;
    __m256i _y;
    __m256i _z;

    for(unsigned i = 0; i < boundedRange; i += AVX256BYTES/sizeof(unsigned))
    {
        _x = _mm256_load_si256((const __m256i*)&a[i]);
        _y = _mm256_load_si256((const __m256i*)&b[i]);
        _z = _mm256_sub_epi32(_x,_y);
        _mm256_store_si256((__m256i *)&c[i], _z);
    }

    if(printTime){
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

        std::cout << std::left << std::setw(TIME_FORMAT) << " HOST:  1D Array AVX-256 Difference(UInt32): " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (C * sizeof(unsigned)) / milliseconds / 1e6 << " GB/s"
                  << std::endl;
    }
}


void Propulsion::hostSubtract1DArraysDouble64AVX256(double *a, double *b, double *c, unsigned C, bool printTime)
{
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    unsigned boundedRange = 0;

    if(C % (unsigned)(AVX256BYTES/sizeof(double)) == 0)
    {
        boundedRange = C;
    }
        // else the bounds lie outside of avx256
    else
    {
        boundedRange = C - (C % (unsigned)(AVX256BYTES/(sizeof(double))));
        std::cout << "Bounded Range boi! " << C - (C % (unsigned)(AVX256BYTES/(sizeof(double)))) << std::endl;
        for(unsigned i = boundedRange; i < C; i++)
        {
            c[i] = a[i] + b[i];
        }
    }

    __m256d _x;
    __m256d _y;
    __m256d _z;

    for(unsigned i = 0; i < boundedRange; i += AVX256BYTES/(sizeof(double)))
    {
        _x = _mm256_load_pd(&a[i]);
        _y = _mm256_load_pd(&b[i]);

        _z = _mm256_sub_pd(_x,_y);

        _mm256_store_pd(&c[i],_z);
    }

    if(printTime){
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

        std::cout << std::left << std::setw(TIME_FORMAT) << " HOST:  1D Array AVX-256 Difference(Double): " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (C * sizeof(double)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }
}

void Propulsion::hostSubtract1DArraysFloat32AVX256(float *a, float *b, float *c, unsigned C, bool printTime)
{
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    unsigned boundedRange = 0;

    if(C % (unsigned)(AVX256BYTES/(sizeof(float))) == 0)
    {
        boundedRange = C;
    }
        // else outside of boundedRange
    else
    {
        boundedRange = C - (C % (unsigned)(AVX256BYTES/(sizeof(float))));
        std::cout << "Bounded Range boi! " << C - (C % (unsigned)(AVX256BYTES/(sizeof(float)))) << std::endl;
        for(unsigned i = boundedRange; i < C; i++)
        {
            c[i] = a[i] + b[i];
        }
    }

    __m256 _x;
    __m256 _y;
    __m256 _z;

    for(unsigned i = 0; i < boundedRange; i += AVX256BYTES/(sizeof(float)))
    {
        _x = _mm256_load_ps(&a[i]);
        _y = _mm256_load_ps(&b[i]);
        _z = _mm256_sub_ps(_x,_y);
        _mm256_store_ps(&c[i],_z);
    }

    if(printTime){
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

        std::cout << std::left << std::setw(TIME_FORMAT) << " HOST:  1D Array AVX-256 Difference(Float): " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (C * sizeof(float)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }
}

/*
 * constexpr is a function that is described in c++17 and beyond. Since this is c++14, and VS/CUDA supports this
 * feature, we are using it that way I don't have to convert the arrays below to void* arrays, and pass them to
 * static cast them later(I want to save a headache or two, and the fact of the matter is that I'm spending already
 * more time on this project then I intended[Which is totally fine!]).
 */
#pragma warning(disable:4984)
template<typename type>
void Propulsion::hostSubtract1DArraysAVX256(type *a, type *b, type *c, unsigned C, bool printTime)
{
    if(a == nullptr || b == nullptr || c == nullptr)
    {
        return;
    }


    if constexpr(std::is_same_v<type, int>)
    {
        Propulsion::hostSubtract1DArraysInt32AVX256(a,b,c,C,printTime);
    }
    else if constexpr(std::is_same_v<type, unsigned>)
    {
        Propulsion::hostSubtract1DArraysUInt32AVX256(a,b,c,C,printTime);
    }
    else if constexpr(std::is_same_v<type, short>)
    {
        Propulsion::hostSubtract1DArraysInt16AVX256(a,b,c,C,printTime);
    }
    else if constexpr(std::is_same_v<type, double>)
    {
        Propulsion::hostSubtract1DArraysDouble64AVX256(a,b,c,C,printTime);
    }
    else if constexpr(std::is_same_v<type, float>)
    {
        Propulsion::hostSubtract1DArraysFloat32AVX256(a,b,c,C,printTime);
    }
#pragma warning(default:4984)
}

template<typename type>
void Propulsion::hostSubtract1DArrays(type *a, type *b, type *c, unsigned C, bool printTime) {
    // Create starting point for timer.
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for(unsigned i = 0; i < C; i++)
    {
        c[i] = a[i] - b[i]; // ai minus bi = ci
    }

    if(printTime){
        // Create ending point for timer.
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

        std::cout << std::left << std::setw(TIME_FORMAT) << " HOST:  1D Array Difference: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (C * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }
    return;
}

template<typename type>
void Propulsion::hostMultiply1DArrays(type *a, type *b, type *c, unsigned C, bool printTime) {
    // Create starting point for timer
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for(unsigned i = 0; i < C; i++)
    {
        c[i] = a[i] * b[i]; // ai * bi = ci
    }
    if(printTime){
        // Create ending point for timer.
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

        std::cout << std::left << std::setw(TIME_FORMAT) << " HOST:  1D Array Multiply: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (C * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }
    return;
}

template<typename type>
void Propulsion::hostDivide1DArrays(type *a, type *b, type *c, unsigned C, bool printTime)
{
    // Create starting point for timer
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for(unsigned i = 0; i < C; i++)
    {
        c[i] = a[i] / b[i]; // ai * bi = ci
    }
    if(printTime){
        // Create ending point for timer.
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

        std::cout << std::left << std::setw(TIME_FORMAT) << " HOST:  1D Array Division: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (C * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }
    return;
}

template<typename type>
void Propulsion::hostMultiply1DArrayByScalar(type *a, type s, unsigned C, bool printTime)
{
    // Create starting point for timer
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    // a_i = a_i * s
    for(unsigned i = 0; i < C; i++)
    {
        a[i] *= s;
    }
    if(printTime){
        // Create ending point for timer.
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

        std::cout << std::left << std::setw(TIME_FORMAT) << " HOST:  1D Array Scalar: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (C * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }
    return;
}

template<typename type>
void Propulsion::hostStencilSum1DArrays(type * a, type * c, unsigned C, unsigned int radius, bool printTime)
{
    // Create starting point for timer
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    type temp;
    int j;
    for(int i = 0; i < C; i++)
    {
        temp = 0;

        for(j = (i - (signed)radius < 0 ? 0 : (i - radius)); j <= i + (signed)radius && j < (signed)C; j++)
        {
            temp += a[j];
        }
        c[i] = temp;
    }

    if(printTime){
        // Create ending point for timer.
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

        std::cout << std::left << std::setw(TIME_FORMAT)
                  << " HOST:  1D Array Stencil(r=" + std::to_string(radius) + ") :" <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << ((C * sizeof(type) * (radius * 2 + 1))) / milliseconds / 1e6
                  << " GB/s" << std::endl;
    }
    return;
}

