//
// Created by steve on 7/7/2020.
//
#pragma once
#include "Propulsion.cuh"


/*
 * Nice Wrapper function provided by:
 * https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


__global__ void deviceHelloWorld()
{
    if(threadIdx.x == 0)
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


template<typename type> __global__ void deviceAdd1DMatrices(type *dev_a, type *dev_b, type *dev_c, unsigned cols)
{
    unsigned tID = blockDim.x * blockIdx.x + threadIdx.x;
    if (tID < cols)
    {
        dev_c[tID] = dev_a[tID] + dev_b[tID];
    }
}


template<typename type>
void Propulsion::cudaAdd1DArrays(type *a, type *b, type *c, unsigned cols, bool printTime) {
    // Start Timer
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    // Create device array pointers.
    type *dev_a, *dev_b, *dev_c;

    // Create memory for dev_a/b/c... It is created via R(rows) * C(columns) * type(size of type like int, float, etc).
    gpuErrchk(cudaMalloc((void**) &dev_a, cols * sizeof(type)));
    gpuErrchk(cudaMalloc((void**) &dev_b, cols * sizeof(type)));
    gpuErrchk(cudaMalloc((void**) &dev_c, cols * sizeof(type)));

    // Copy data from host to device.
    gpuErrchk(cudaMemcpy(dev_a, a, cols * sizeof(type), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_b, b, cols * sizeof(type), cudaMemcpyHostToDevice));

    // Generate block. Based off total size of the array.
    // Block = The ceiling of array_size / 1024.
    // 1024 is the max thread amount. Therefore, we can split the workload up into blocks of threads.
    int block = cols / (MAX_THREADS) + 1;

    gpuErrchk(cudaEventRecord(start));
    // Start kernel if < 65536
    deviceAdd1DMatrices<<<block,MAX_THREADS>>>(dev_a, dev_b, dev_c, cols);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize()); // Synchronize CUDA with HOST. Wait for Device.

    gpuErrchk(cudaEventRecord(stop));

    // Copy the data from device c to c.
    gpuErrchk(cudaMemcpy(c, dev_c, cols * sizeof(type), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaEventSynchronize(stop));
    float milliseconds = 0;
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));


    // Free Memory in Device.
    gpuErrchk(cudaFree(dev_a));
    gpuErrchk(cudaFree(dev_b));
    gpuErrchk(cudaFree(dev_c));


    if(printTime){
        std::cout << std::left << std::setw(TIME_FORMAT) << " CUDA:  1D Array Addition: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (cols * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }

    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
    return;
}


template <typename type> __global__ void deviceAdd1DMatricesWithStride(type *dev_a, type *dev_b, type *dev_c, unsigned cols)
{
    /*for(int i = blockIdx.x * blockDim.x + threadIdx.x;
        i < C;
        i += blockDim.x * gridDim.x)
    {
        dev_c[i] = dev_a[i] + dev_b[i];
    }*/

    int tID = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = tID; i < cols; i += stride)
    {
        dev_c[i] = dev_a[i] + dev_b[i];
    }
}


template<typename type>
void Propulsion::cudaAdd1DArraysWithStride(type *a, type *b, type *c, unsigned cols, bool printTime)
{
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    // Create device array pointers.
    type *dev_a, *dev_b, *dev_c;

    // Create memory for dev_a/b/c... It is created via R(rows) * C(columns) * type(size of type like int, float, etc).
    gpuErrchk(cudaMalloc((void**) &dev_a, cols * sizeof(type)));
    gpuErrchk(cudaMalloc((void**) &dev_b, cols * sizeof(type)));
    gpuErrchk(cudaMalloc((void**) &dev_c, cols * sizeof(type)));

    // Copy data from host to device.
    gpuErrchk(cudaMemcpy(dev_a, a, cols * sizeof(type), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_b, b, cols * sizeof(type), cudaMemcpyHostToDevice));

    int blockSize = MAX_THREADS;
    int numBlocks = (cols + blockSize - 1) / blockSize;

    gpuErrchk(cudaEventRecord(start));
    //deviceAdd1DMatriceWithStride<<<8*numSMs,1024>>>(dev_a, dev_b, dev_c, C);
    deviceAdd1DMatricesWithStride<<<numBlocks,blockSize>>>(dev_a, dev_b, dev_c, cols);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());


    gpuErrchk(cudaEventRecord(stop));

    // Copy the data from device c to c.
    gpuErrchk(cudaMemcpy(c, dev_c, cols * sizeof(type), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaEventSynchronize(stop));
    float milliseconds = 0;
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));

    // Free Memory in Device.
    gpuErrchk(cudaFree(dev_a));
    gpuErrchk(cudaFree(dev_b));
    gpuErrchk(cudaFree(dev_c));

    if(printTime){
        std::cout << std::left << std::setw(TIME_FORMAT) << " CUDA:  1D Array Addition with Stride: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (cols * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }

    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
    return;

}


template <typename type> __global__ void deviceSubtract1DMatrices(type *dev_a, type *dev_b, type *dev_c, unsigned cols)
{
    unsigned tID = blockDim.x * blockIdx.x + threadIdx.x;
    if(tID < cols)
    {
        dev_c[tID] = dev_a[tID] - dev_b[tID];
    }
}


template<typename type>
void Propulsion::cudaSubtract1DArrays(type *a, type *b, type *c, unsigned cols, bool printTime)
{
    // Start Timer
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    // Create device array pointers.
    type *dev_a, *dev_b, *dev_c;

    // Create memory for dev_a/b/c... It is created via R(rows) * C(columns) * type(size of type like int, float, etc).
    gpuErrchk(cudaMalloc((void**) &dev_a, cols * sizeof(type)));
    gpuErrchk(cudaMalloc((void**) &dev_b, cols * sizeof(type)));
    gpuErrchk(cudaMalloc((void**) &dev_c, cols * sizeof(type)));

    // Copy data from host to device.
    gpuErrchk(cudaMemcpy(dev_a, a, cols * sizeof(type), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_b, b, cols * sizeof(type), cudaMemcpyHostToDevice));

    // Generate block. Based off total size of the array.
    // Block = The ceiling of array_size / 1024.
    // 1024 is the max thread amount. Therefore, we can split the workload up into blocks of threads.
    int block = cols / MAX_THREADS + 1;

    gpuErrchk(cudaEventRecord(start));
    // Start kernel.
    deviceSubtract1DMatrices<<<block,MAX_THREADS>>>(dev_a, dev_b, dev_c, cols);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize()); // Synchronize CUDA with HOST. Wait for Device.

    gpuErrchk(cudaEventRecord(stop));

    // Copy the data from device c to c.
    gpuErrchk(cudaMemcpy(c, dev_c, cols * sizeof(type), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaEventSynchronize(stop));
    float milliseconds = 0;
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));


    // Free Memory in Device.
    gpuErrchk(cudaFree(dev_a));
    gpuErrchk(cudaFree(dev_b));
    gpuErrchk(cudaFree(dev_c));

    if(printTime){
        std::cout << std::left << std::setw(TIME_FORMAT) << " CUDA:  1D Array Difference: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (cols * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }

    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
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
void Propulsion::cudaSubtract1DArraysWithStride(type *a, type *b, type *c, unsigned cols, bool printTime)
{
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    // Create device array pointers.
    type *dev_a, *dev_b, *dev_c;

    // Create memory for dev_a/b/c... It is created via R(rows) * C(columns) * type(size of type like int, float, etc).
    gpuErrchk(cudaMalloc((void**) &dev_a, cols * sizeof(type)));
    gpuErrchk(cudaMalloc((void**) &dev_b, cols * sizeof(type)));
    gpuErrchk(cudaMalloc((void**) &dev_c, cols * sizeof(type)));

    // Copy data from host to device.
    gpuErrchk(cudaMemcpy(dev_a, a, cols * sizeof(type), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_b, b, cols * sizeof(type), cudaMemcpyHostToDevice));

    int blockSize = MAX_THREADS;
    int numBlocks = (cols + blockSize - 1) / blockSize;

    gpuErrchk(cudaEventRecord(start));
    deviceSubtract1DMatricesWithStride<<<numBlocks,blockSize>>>(dev_a, dev_b, dev_c, cols);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaEventRecord(stop));

    // Copy the data from device c to c.
    gpuErrchk(cudaMemcpy(c, dev_c, cols * sizeof(type), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaEventSynchronize(stop));
    float milliseconds = 0;
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));

    // Free Memory in Device.
    gpuErrchk(cudaFree(dev_a));
    gpuErrchk(cudaFree(dev_b));
    gpuErrchk(cudaFree(dev_c));

    if(printTime){
        std::cout << std::left << std::setw(TIME_FORMAT) << " CUDA:  1D Array Diff. with Stride: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (cols * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }

    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
    return;
}


template <typename type> __global__ void deviceSchursProductStride(type *dev_a, type *dev_b, type *dev_c, unsigned cols)
{
    int tID = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = tID; i < cols; i += stride)
    {
        dev_c[i] = dev_a[i] * dev_b[i];
    }
}


template<typename type> void Propulsion::hostDotProduct(type *A, type*B, type *C, unsigned aRows, unsigned aColsBRows, unsigned bCols, bool printTime)
{
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    type sum = 0;
    unsigned n = 0;

    for (unsigned r = 0; r < aRows; r++) {
        for (unsigned c = 0; c < bCols; c++) {
            for (unsigned i = 0; i < aColsBRows; i++) {
                //sum += at(r, i) * b.M[i * b.cols + c];
                sum += A[r*aColsBRows + i] * B[i * bCols + c];
            }
            C[n] = sum;
            sum = 0;
            n++;
        }
    }
    if(printTime){
        // Create ending point for timer.
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        float milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

        std::cout << std::left << std::setw(TIME_FORMAT) << " HOST:  Matrix Dot Product: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << ((double)aRows * (double)aColsBRows * (double)bCols * (double)sizeof(type)) / (double)milliseconds / 1e6 << " GB/s" << std::endl;
    }
    return;
}


template <typename type>
__global__ void deviceDotProduct(type *dev_a, type *dev_b, type *dev_c, unsigned rows, unsigned AcolsBRows, unsigned cols)
{
    unsigned tRow = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned tCol = blockIdx.y * blockDim.y + threadIdx.y;
    type tempSum = 0;

    if(tRow < rows && tCol < cols)
    {
        for (unsigned i = 0; i < AcolsBRows; i++)
        {
            tempSum += dev_a[tRow * AcolsBRows + i] * dev_b[i * cols + tCol];
        }
        dev_c[tRow * cols + tCol] = tempSum;
    }
}


template<typename type>
void Propulsion::cudaDotProduct(type *a, type *b, type *c, unsigned aRows, unsigned aColsBRows, unsigned bCols, bool printTime)
{
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    // Create device array pointers.
    type *dev_a, *dev_b, *dev_c;

    // Create memory for dev_a/b/c... It is created via R(rows) * C(columns) * type(size of type like int, float, etc).
    gpuErrchk(cudaMalloc((void**) &dev_a, aRows * sizeof(type) * aColsBRows));
    gpuErrchk(cudaMalloc((void**) &dev_b, aColsBRows * sizeof(type) * bCols));
    gpuErrchk(cudaMalloc((void**) &dev_c, aRows * sizeof(type) * bCols));

    // Copy data from host to device.
    gpuErrchk(cudaMemcpy(dev_a, a, aRows * sizeof(type) * aColsBRows, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_b, b, aColsBRows * sizeof(type) * bCols, cudaMemcpyHostToDevice));

    // block of threads.
    dim3 block_dim(32,32);

    // total blocks of threads in x direction
    unsigned blocksX = std::ceil(( (double)bCols) / ( (double)block_dim.x) );
    // total blocks of threads in y direction
    unsigned blocksY = std::ceil(( (double)aRows) / ( (double)block_dim.y) );
    dim3 grid_dim(blocksX, blocksY);

    // start timer.
    gpuErrchk(cudaEventRecord(start));
    deviceDotProduct<<<grid_dim, block_dim>>>(dev_a, dev_b, dev_c, aRows, aColsBRows, bCols);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaEventRecord(stop));

    // Copy the data from device c to c.
    gpuErrchk(cudaMemcpy(c, dev_c, aRows * sizeof(type) * bCols, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaEventSynchronize(stop));
    float milliseconds = 0;
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));

    // Free Memory in Device.
    gpuErrchk(cudaFree(dev_a));
    gpuErrchk(cudaFree(dev_b));
    gpuErrchk(cudaFree(dev_c));

    if(printTime){
        std::cout << std::left << std::setw(TIME_FORMAT) << " CUDA:  Matrix Multiplication: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << ((double)aRows * (double)aColsBRows * (double)bCols * (double)sizeof(type)) / (double)milliseconds / 1e6 << " GB/s" << std::endl;
    }

    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
    return;
}


template<typename type>
void Propulsion::hostSchurProduct(type *a, type *b, type *c, unsigned int cols, bool printTime) {
    // Create starting point for timer
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for(unsigned i = 0; i < cols; i++)
    {
        c[i] = a[i] * b[i]; // ai * bi = ci
    }
    if(printTime){
        // Create ending point for timer.
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

        std::cout << std::left << std::setw(TIME_FORMAT) << " HOST:  Schur Product: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (cols * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }
    return;
}


template <typename type> __global__ void deviceSchurProduct(type *dev_a, type *dev_b, type *dev_c, unsigned cols)
{
    int tID = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = tID; i < cols; i += stride)
    {
        dev_c[i] = dev_a[i] * dev_b[i];
    }
}


template<typename type>
void Propulsion::cudaSchurProduct(type *a, type *b, type *c, unsigned cols, bool printTime)
{
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    // Create device array pointers.
    type *dev_a, *dev_b, *dev_c;

    // Create memory for dev_a/b/c... It is created via R(rows) * C(columns) * type(size of type like int, float, etc).
    gpuErrchk(cudaMalloc((void**) &dev_a, cols * sizeof(type)));
    gpuErrchk(cudaMalloc((void**) &dev_b, cols * sizeof(type)));
    gpuErrchk(cudaMalloc((void**) &dev_c, cols * sizeof(type)));

    // Copy data from host to device.
    gpuErrchk(cudaMemcpy(dev_a, a, cols * sizeof(type), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_b, b, cols * sizeof(type), cudaMemcpyHostToDevice));

    int blockSize = MAX_THREADS;
    int numBlocks = (cols + blockSize - 1) / blockSize;

    // Start cuda timer
    gpuErrchk(cudaEventRecord(start));

    // Start Division Kernel
    deviceSchurProduct<<<numBlocks,blockSize>>>(dev_a, dev_b, dev_c, cols);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Record Cuda end time.
    gpuErrchk(cudaEventRecord(stop));

    // Copy the data from device c to c.
    gpuErrchk(cudaMemcpy(c, dev_c, cols * sizeof(type), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaEventSynchronize(stop));
    float milliseconds = 0;
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));

    // Free Memory in Device.
    gpuErrchk(cudaFree(dev_a));
    gpuErrchk(cudaFree(dev_b));
    gpuErrchk(cudaFree(dev_c));

    if(printTime){
        std::cout << std::left << std::setw(TIME_FORMAT) << " CUDA:  1D Array Division with Stride: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (cols * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }

    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
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
void Propulsion::cudaDivide1DArrays(type *a, type *b, type *c, unsigned cols, bool printTime)
{
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    // Create device array pointers.
    type *dev_a, *dev_b, *dev_c;

    // Create memory for dev_a/b/c... It is created via R(rows) * C(columns) * type(size of type like int, float, etc).
    gpuErrchk(cudaMalloc((void**) &dev_a, cols * sizeof(type)));
    gpuErrchk(cudaMalloc((void**) &dev_b, cols * sizeof(type)));
    gpuErrchk(cudaMalloc((void**) &dev_c, cols * sizeof(type)));

    // Copy data from host to device.
    gpuErrchk(cudaMemcpy(dev_a, a, cols * sizeof(type), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_b, b, cols * sizeof(type), cudaMemcpyHostToDevice));

    int blockSize = MAX_THREADS;
    int numBlocks = (cols + blockSize - 1) / blockSize;

    // Start cuda timer
    gpuErrchk(cudaEventRecord(start));

    // Start Division Kernel
    deviceDivide1DMatricesStride<<<numBlocks,blockSize>>>(dev_a, dev_b, dev_c, cols);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Record Cuda end time.
    gpuErrchk(cudaEventRecord(stop));

    // Copy the data from device c to c.
    gpuErrchk(cudaMemcpy(c, dev_c, cols * sizeof(type), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaEventSynchronize(stop));
    float milliseconds = 0;
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));

    // Free Memory in Device.
    gpuErrchk(cudaFree(dev_a));
    gpuErrchk(cudaFree(dev_b));
    gpuErrchk(cudaFree(dev_c));

    if(printTime){
        std::cout << std::left << std::setw(TIME_FORMAT) << " CUDA:  1D Array Division with Stride: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (cols * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }

    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
    return;
}


template <typename type> __global__ void deviceMultiplyArrayByScalar(type *dev_a, type s, unsigned cols)
{
    int tID = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = tID; i < cols; i += stride)
    {
        dev_a[tID] *= s;
    }
}


template<typename type>
void Propulsion::cudaMultiply1DArrayByScalar(type *a, type s, unsigned cols, bool printTime) {
    // Create cuda event objects.
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    // For device array pointer
    type *dev_a;

    // Allocate memory for dev_a based on the size of a
    gpuErrchk(cudaMalloc((void**) &dev_a, cols * sizeof(type)));
    // Copy contents of a over to dev_a
    gpuErrchk(cudaMemcpy(dev_a, a, cols * sizeof(type), cudaMemcpyHostToDevice));

    // Create a Kernel that will consist of 1024 Threads along with a block size of (C * 1024 -1) / 1024
    int blockSize = MAX_THREADS;
    int numBlocks = (cols + blockSize - 1) / blockSize;

    // Start Timers
    gpuErrchk(cudaEventRecord(start));
    deviceMultiplyArrayByScalar<<<numBlocks,blockSize>>>(dev_a, s, cols);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaEventRecord(stop));

    gpuErrchk(cudaMemcpy(a, dev_a, cols * sizeof(type), cudaMemcpyDeviceToHost));

    // Synchronize cuda event object.
    gpuErrchk(cudaEventSynchronize(stop));


    // Get time for the operation.
    float milliseconds = 0;
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));

    // Free Memory in Device.
    gpuErrchk(cudaFree(dev_a));

    if(printTime){
        std::cout << std::left << std::setw(TIME_FORMAT) << " CUDA:  1D Array Multiply Scalar: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (cols * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }

    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
    return;
}


template <typename type> __global__ void deviceStencilSum1DMatrix(type *d_a, type *d_c, unsigned cols, int spacing, int radius)
{
    int tID = threadIdx.x * spacing;
    type tempSum = 0;


    for(int i = tID; (i < (tID + spacing) && i < (signed)cols); i++)
    {
        //printf("tID = %d : i = %d : r = %d\n",threadIdx.x, i, radius);
        for(int j = ((i - radius) < 0 ? 0 : (i - radius)) ; j <= i + radius && j < cols; j++)
        {
            tempSum += d_a[j];
        }

        d_c[i] = tempSum; // c array at i'th pos equal to sum of the stencil.
        tempSum = 0;    // reset sum
        __syncthreads();
    }
}


template <typename type> __global__ void deviceStencilSum1D(type *d_a, type *d_c, unsigned cols, int radius)
{
    int tID = (blockDim.x * blockIdx.x + threadIdx.x) * (radius * 2 + 1);
    type tempSum = 0;

    for(int i = tID; (i < tID + (radius * 2 + 1)) && (i < cols); i++ )
    {
        for(int j = (i - radius) < 0 ? 0 : (i - radius); (j <= (radius + i)) && (j < cols); j++)
        {
            tempSum += d_a[j];
        }
        d_c[i] = tempSum;
        tempSum = 0;
        __syncthreads();
    }
}


template<typename type>
void Propulsion::cudaStencilSum1DArrays(type *a, type *c, unsigned cols, unsigned radius, bool printTime)
{
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    int diameter = radius * 2 +1;

    // Create device array pointers.
    type *dev_a, *dev_c;

    // Create memory for dev_a/b/c... It is created via R(rows) * C(columns) * type(size of type like int, float, etc).
    gpuErrchk(cudaMalloc((void**) &dev_a, cols * sizeof(type)));
    gpuErrchk(cudaMalloc((void**) &dev_c, cols * sizeof(type)));

    // Copy data from host to device.
    gpuErrchk(cudaMemcpy(dev_a, a, cols * sizeof(type), cudaMemcpyHostToDevice));

    // Max Elements Per Thread. E.g.
    //              =>|<=
    //  [ 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8]
    //            3 + 4 + 5  #radius = 1
    //            =  12
    int numBlocks = cols / MAX_THREADS + 1;


    // Start cuda timer
    gpuErrchk(cudaEventRecord(start));

    // Start Division Kernel
    //deviceStencilSum1DMatrix<<<numBlocks, threads>>>(dev_a,dev_c, C, spaceBetweenThreads, radius);
    deviceStencilSum1D<<<numBlocks, MAX_THREADS>>>(dev_a, dev_c, cols, radius);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Record Cuda end time.
    gpuErrchk(cudaEventRecord(stop));

    // Copy the data from device c to c.
    gpuErrchk(cudaMemcpy(c, dev_c, cols * sizeof(type), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaEventSynchronize(stop));
    float milliseconds = 0;
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));

    // Free Memory in Device.
    gpuErrchk(cudaFree(dev_a));
    gpuErrchk(cudaFree(dev_c));

    if(printTime){
        std::cout << std::left << std::setw(TIME_FORMAT)
                  << " CUDA:  1D Array Stencil(r=" + std::to_string(radius) + ") : " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (diameter * cols * sizeof(type)) / milliseconds / 1e6 << " GB/s"
                  << std::endl;
    }

    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
    return;
}


void Propulsion::hostAdd1DArraysInt16AVX256(short *a, short *b, short *c, unsigned cols, bool printTime)
{
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    unsigned boundedRange = 0;

    if(cols % (short)(AVX256BYTES / (sizeof(short))) == 0)
    {
        boundedRange = cols;
    }
    else
    {
        boundedRange = cols - (cols % (unsigned)(AVX256BYTES / sizeof(short)));
        for(unsigned i = boundedRange; i < cols; i++)
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
                  " ms." << std::setw(TIME_WIDTH) << (cols * sizeof(short)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }
}


void Propulsion::hostAdd1DArraysInt32AVX256(int *a, int *b, int *c, unsigned cols, bool printTime) {
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    unsigned boundedRange = 0;

    if(cols % (unsigned)(AVX256BYTES / (sizeof(int))) == 0)
    {
        boundedRange = cols;
    }
    // else the bounds lie outside of avx256
    else
    {
        boundedRange = cols - (cols % (unsigned)(AVX256BYTES / (sizeof(int))));
        for(unsigned i = boundedRange; i < cols; i++)
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
                  " ms." << std::setw(TIME_WIDTH) << (cols * sizeof(int)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }
    return;
}


void Propulsion::hostAdd1DArraysUInt32AVX256(unsigned *a, unsigned *b, unsigned *c, unsigned cols, bool printTime)
{
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    unsigned boundedRange = 0;

    if(cols % (unsigned)(AVX256BYTES / sizeof(unsigned)) == 0)
    {
        boundedRange = 0;
    }
    // else the bounds lie outside of an even AVX256 ins.
    else
    {
        boundedRange = cols - (cols % (unsigned)(AVX256BYTES / (sizeof(unsigned ))));
        std::cout << "Bounded Range boi! " << cols - (cols % (unsigned)(AVX256BYTES / (sizeof(unsigned)))) << std::endl;
        for(unsigned i = boundedRange; i < cols; i++)
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
                  " ms." << std::setw(TIME_WIDTH) << (cols * sizeof(unsigned)) / milliseconds / 1e6 << " GB/s"
                  << std::endl;
    }
}


void Propulsion::hostAdd1DArraysDouble64AVX256(double *a, double *b, double *c, unsigned cols, bool printTime)
{
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    unsigned boundedRange = 0;

    if(cols % (unsigned)(AVX256BYTES / sizeof(double)) == 0)
    {
        boundedRange = cols;
    }
    // else the bounds lie outside of avx256
    else
    {
        boundedRange = cols - (cols % (unsigned)(AVX256BYTES / (sizeof(double))));
        std::cout << "Bounded Range boi! " << cols - (cols % (unsigned)(AVX256BYTES / (sizeof(double)))) << std::endl;
        for(unsigned i = boundedRange; i < cols; i++)
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
                  " ms." << std::setw(TIME_WIDTH) << (cols * sizeof(double)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }
}


void Propulsion::hostAdd1DArraysFloat32AVX256(float *a, float *b, float *c, unsigned cols, bool printTime)
{
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    unsigned boundedRange = 0;

    if(cols % (unsigned)(AVX256BYTES / (sizeof(float))) == 0)
    {
        boundedRange = cols;
    }
    // else outside of boundedRange
    else
    {
        boundedRange = cols - (cols % (unsigned)(AVX256BYTES / (sizeof(float))));
        for(unsigned i = boundedRange; i < cols; i++)
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
                  " ms." << std::setw(TIME_WIDTH) << (cols * sizeof(float)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }
}


/*
 * constexpr is a function that is described in c++17 and beyond. Since this is c++14, and VS/CUDA supports this
 * feature, we are using it that way I don't have to convert the arrays below to void* arrays, and pass them to
 * static cast them later(I want to save a headache or two). Presuming that this is using VS/nvcc compiler to
 * compile the code anyways.
 */
#pragma warning(disable:4984)
template<typename type>
void Propulsion::hostAdd1DArraysAVX256(type *a, type *b, type *c, unsigned cols, bool printTime)
{
    if(a == nullptr || b == nullptr || c == nullptr)
    {
        return;
    }


    if constexpr(std::is_same_v<type, int>)
    {
        Propulsion::hostAdd1DArraysInt32AVX256(a, b, c, cols, printTime);
    }
    else if constexpr(std::is_same_v<type, unsigned>)
    {
        Propulsion::hostAdd1DArraysUInt32AVX256(a, b, c, cols, printTime);
    }
    else if constexpr(std::is_same_v<type, short>)
    {
        Propulsion::hostAdd1DArraysInt16AVX256(a, b, c, cols, printTime);
    }
    else if constexpr(std::is_same_v<type, double>)
    {
        Propulsion::hostAdd1DArraysDouble64AVX256(a, b, c, cols, printTime);
    }
    else if constexpr(std::is_same_v<type, float>)
    {
        Propulsion::hostAdd1DArraysFloat32AVX256(a, b, c, cols, printTime);
    }
    #pragma warning(default:4984)
}


template<typename type>
void Propulsion::hostAdd1DArrays(type *a, type *b, type *c, unsigned cols, bool printTime) {
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    for(unsigned i = 0; i < cols; i++)
    {
        c[i] = a[i] + b[i];
    }

    if(printTime){
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

        std::cout << std::left << std::setw(TIME_FORMAT) << " HOST:  1D Array Addition: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (cols * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }
    return;
}


void Propulsion::hostSubtract1DArraysInt16AVX256(short *a, short *b, short *c, unsigned cols, bool printTime)
{
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    unsigned boundedRange = 0;

    if(cols % (short)(AVX256BYTES / (sizeof(short))) == 0)
    {
        boundedRange = cols;
    }
    else
    {
        boundedRange = cols - (cols % (unsigned)(AVX256BYTES / sizeof(short)));
        for(unsigned i = boundedRange; i < cols; i++)
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
                  " ms." << std::setw(TIME_WIDTH) << (cols * sizeof(short)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }
}


void Propulsion::hostSubtract1DArraysInt32AVX256(int *a, int *b, int *c, unsigned cols, bool printTime) {
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    /*
    for(unsigned i = 0; i < C; i++)
    {
        c[i] = a[i] + b[i];
    }
    */
    unsigned boundedRange = 0;

    if(cols % (unsigned)(AVX256BYTES / (sizeof(int))) == 0)
    {
        boundedRange = cols;
    }
        // else the bounds lie outside of avx256
    else
    {
        boundedRange = cols - (cols % (unsigned)(AVX256BYTES / (sizeof(int))));
        for(unsigned i = boundedRange; i < cols; i++)
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
                  " ms." << std::setw(TIME_WIDTH) << (cols * sizeof(int)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }
    return;
}


void Propulsion::hostSubtract1DArraysUInt32AVX256(unsigned *a, unsigned *b, unsigned *c, unsigned cols, bool printTime)
{
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    unsigned boundedRange = 0;

    if(cols % (unsigned)(AVX256BYTES / sizeof(unsigned)) == 0)
    {
        boundedRange = 0;
    }
        // else the bounds lie outside of an even AVX256 ins.
    else
    {
        boundedRange = cols - (cols % (unsigned)(AVX256BYTES / (sizeof(unsigned ))));
        std::cout << "Bounded Range boi! " << cols - (cols % (unsigned)(AVX256BYTES / (sizeof(unsigned)))) << std::endl;
        for(unsigned i = boundedRange; i < cols; i++)
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
                  " ms." << std::setw(TIME_WIDTH) << (cols * sizeof(unsigned)) / milliseconds / 1e6 << " GB/s"
                  << std::endl;
    }
}


void Propulsion::hostSubtract1DArraysDouble64AVX256(double *a, double *b, double *c, unsigned cols, bool printTime)
{
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    unsigned boundedRange = 0;

    if(cols % (unsigned)(AVX256BYTES / sizeof(double)) == 0)
    {
        boundedRange = cols;
    }
        // else the bounds lie outside of avx256
    else
    {
        boundedRange = cols - (cols % (unsigned)(AVX256BYTES / (sizeof(double))));
        for(unsigned i = boundedRange; i < cols; i++)
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
                  " ms." << std::setw(TIME_WIDTH) << (cols * sizeof(double)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }
}


void Propulsion::hostSubtract1DArraysFloat32AVX256(float *a, float *b, float *c, unsigned cols, bool printTime)
{
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    unsigned boundedRange = 0;

    if(cols % (unsigned)(AVX256BYTES / (sizeof(float))) == 0)
    {
        boundedRange = cols;
    }
        // else outside of boundedRange
    else
    {
        boundedRange = cols - (cols % (unsigned)(AVX256BYTES / (sizeof(float))));
        for(unsigned i = boundedRange; i < cols; i++)
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
                  " ms." << std::setw(TIME_WIDTH) << (cols * sizeof(float)) / milliseconds / 1e6 << " GB/s" << std::endl;
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
void Propulsion::hostSubtract1DArraysAVX256(type *a, type *b, type *c, unsigned cols, bool printTime)
{
    if(a == nullptr || b == nullptr || c == nullptr)
    {
        return;
    }


    if constexpr(std::is_same_v<type, int>)
    {
        Propulsion::hostSubtract1DArraysInt32AVX256(a, b, c, cols, printTime);
    }
    else if constexpr(std::is_same_v<type, unsigned>)
    {
        Propulsion::hostSubtract1DArraysUInt32AVX256(a, b, c, cols, printTime);
    }
    else if constexpr(std::is_same_v<type, short>)
    {
        Propulsion::hostSubtract1DArraysInt16AVX256(a, b, c, cols, printTime);
    }
    else if constexpr(std::is_same_v<type, double>)
    {
        Propulsion::hostSubtract1DArraysDouble64AVX256(a, b, c, cols, printTime);
    }
    else if constexpr(std::is_same_v<type, float>)
    {
        Propulsion::hostSubtract1DArraysFloat32AVX256(a, b, c, cols, printTime);
    }
#pragma warning(default:4984)
}


template<typename type>
void Propulsion::hostSubtract1DArrays(type *a, type *b, type *c, unsigned cols, bool printTime) {
    // Create starting point for timer.
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for(unsigned i = 0; i < cols; i++)
    {
        c[i] = a[i] - b[i]; // ai minus bi = ci
    }

    if(printTime){
        // Create ending point for timer.
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

        std::cout << std::left << std::setw(TIME_FORMAT) << " HOST:  1D Array Difference: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (cols * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }
    return;
}


template<typename type>
void Propulsion::hostMultiply1DArrays(type *a, type *b, type *c, unsigned cols, bool printTime) {
    // Create starting point for timer
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for(unsigned i = 0; i < cols; i++)
    {
        c[i] = a[i] * b[i]; // ai * bi = ci
    }
    if(printTime){
        // Create ending point for timer.
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

        std::cout << std::left << std::setw(TIME_FORMAT) << " HOST:  1D Array Multiply: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (cols * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }
    return;
}


template<typename type>
void Propulsion::hostDivide1DArrays(type *a, type *b, type *c, unsigned cols, bool printTime)
{
    // Create starting point for timer
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for(unsigned i = 0; i < cols; i++)
    {
        c[i] = a[i] / b[i]; // ai * bi = ci
    }
    if(printTime){
        // Create ending point for timer.
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

        std::cout << std::left << std::setw(TIME_FORMAT) << " HOST:  1D Array Division: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (cols * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }
    return;
}


template<typename type>
void Propulsion::hostMultiply1DArrayByScalar(type *a, type s, unsigned cols, bool printTime)
{
    // Create starting point for timer
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    // a_i = a_i * s
    for(unsigned i = 0; i < cols; i++)
    {
        a[i] *= s;
    }
    if(printTime){
        // Create ending point for timer.
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

        std::cout << std::left << std::setw(TIME_FORMAT) << " HOST:  1D Array Scalar: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (cols * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }
    return;
}


template<typename type>
void Propulsion::hostStencilSum1DArrays(type * a, type * c, unsigned cols, unsigned int radius, bool printTime)
{
    // Create starting point for timer
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    type temp;
    int j;
    for(int i = 0; i < cols; i++)
    {
        temp = 0;

        for(j = (i - (signed)radius < 0 ? 0 : (i - radius)); j <= i + (signed)radius && j < (signed)cols; j++)
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
                  " ms." << std::setw(TIME_WIDTH) << ((cols * sizeof(type) * (radius * 2 + 1))) / milliseconds / 1e6
                  << " GB/s" << std::endl;
    }
    return;
}


template <typename type>
__global__ void deviceCopyArray(type *dev_a, type *dev_b, unsigned totalSize)
{
    unsigned tID = blockIdx.x * blockDim.x + threadIdx.x;
    if(tID < totalSize)
    {
        dev_b[tID] = dev_a[tID];
    }
}


template<typename type>
void Propulsion::cudaCopyArray(type *a, type *b, unsigned int totalSize, bool printTime)
{
    // Start Timer
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    // Create device array pointers.
    type *dev_a, *dev_b;

    // Create memory for dev_a/b/c... It is created via R(rows) * C(columns) * type(size of type like int, float, etc).
    gpuErrchk(cudaMalloc((void**) &dev_a, totalSize * sizeof(type)));
    gpuErrchk(cudaMalloc((void**) &dev_b, totalSize * sizeof(type)));

    gpuErrchk(cudaMemcpy(dev_a, a, totalSize*sizeof(type), cudaMemcpyHostToDevice));

    // block of threads.
    dim3 block_dim(MAX_THREADS,1);

    // total blocks of threads in x direction
    unsigned blocks = std::ceil( ( (double)totalSize) / ( (double)block_dim.x) );

    dim3 grid_dim(blocks, 1);

    // start timer.
    gpuErrchk(cudaEventRecord(start));
    deviceCopyArray<<<grid_dim,block_dim>>>(dev_a,dev_b,totalSize);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaEventRecord(stop));

    // Copy the data from device c to c.
    gpuErrchk(cudaMemcpy(b, dev_b, totalSize*sizeof(type), cudaMemcpyDeviceToHost));

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Free Memory in Device.
    cudaFree(dev_a);
    cudaFree(dev_b);

    if(printTime){
        std::cout << std::left << std::setw(TIME_FORMAT) << " CUDA:  Copy Array: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << std::endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return;
}
