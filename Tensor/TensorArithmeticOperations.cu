//
// Created by steve on 8/23/2021.
//

#pragma once

template<typename type>
void Propulsion::Tensor<type>::add(Propulsion::Tensor<type> &B, bool printTime)
{
    std::chrono::high_resolution_clock::time_point start;

    if(printTime)
        start = std::chrono::high_resolution_clock::now();

    // Check if dimensions are the same that way we can add them together.
    if(this->checkAllDimensionsMatch(B))
    {
        for(unsigned long long i = 0; i < this->tensor.size(); i++)
        {
            this->tensor[i]->add(*B.tensor[i].get());
        }
    }
    else
    {
        // Generate TensorException
        std::string err = "";
        // Using getDimsExceptionString helper function.
        err += "Tensor Size Mismatch in Add, this: " + getDimsExceptionString(this->dims) + " vs " + getDimsExceptionString(B.dims);
        throw Tensor<type>::TensorException(err.c_str(), __FILE__, __LINE__,
                                            "add", "Both Tensors must match all dimensions with one another, as its element wise addition.");
    }

    if(printTime){
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

        std::cout << std::left << std::setw(TIME_FORMAT) << " HOST: Tensor add: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (this->getTotalSize() * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }
}

template<typename type>
void Propulsion::Tensor<type>::cudaAdd(Propulsion::Tensor<type> &B, bool printTime, bool printStats)
{

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    // Check if dimensions are the same that way we can add them together.
    if(this->checkAllDimensionsMatch(B))
    {

        // Declared for cudaMemGetInfo function.
        size_t free_bytes;
        size_t total_bytes;

        // Fill free and total bytes
        gpuErrchk(cudaMemGetInfo(&free_bytes, &total_bytes));

        // Get size of Matrices in Bytes.
        unsigned long long totalMatrixSizeBytes = this->tensor[0]->getTotalSize() * sizeof(type);
        unsigned long long totalKernelMemorySizeBytes = totalMatrixSizeBytes * 2; // 2 is for the allocated arrays for A,B.
        unsigned long long totalTensorsSizeBytes = totalMatrixSizeBytes * this->tensor.size() * 2;

        // Figure out how many passes we need to achieve the full tensor.
        unsigned long long passes = std::ceil((double)totalTensorsSizeBytes / (double)free_bytes);
        unsigned long long matrixOffset = std::floor( (double)this->tensor.size() / (double) passes);
        unsigned long long remainingMatrices = this->tensor.size() - matrixOffset * passes;

        // Calculate total bytes from each Matrix in Tensor.
        unsigned long long tensorTotalBytes = this->getTotalSize() * sizeof(type);

        // If prints stats are
        if(printStats) {
            std::cout << "[cudaAdd] Total Bytes Requested: " << totalTensorsSizeBytes << std::endl;
            std::cout << "[cudaAdd] Total Free Bytes:      " << free_bytes << std::endl;
            std::cout << "[cudaAdd] Total GPU Bytes:       " << total_bytes << std::endl;
            std::cout << "[cudaAdd] Total Passes:          " << passes << std::endl;
            std::cout << "[cudaAdd] Matrix Offset:         " << matrixOffset << std::endl;
            std::cout << "[cudaAdd] Kernel Byte Size:      " << totalKernelMemorySizeBytes << std::endl;
            std::cout << "[cudaAdd] Remaining Matrices:    " << remainingMatrices << std::endl;
        }

        /// Delete and move to Propulsion or some Const file.
        const int blockSize = 1024;

        // Declare number of streams and create Stream array.
        unsigned long long nStreams = this->tensor.size();
        cudaStream_t *streams = new cudaStream_t[nStreams];

        // Get stream size, in this case the size of the matrix of the tensor(....nxm) => nxm = streamSize.
        unsigned long long streamSize = this->tensor[0]->getTotalSize();
        unsigned long long streamBytes = streamSize * sizeof(type);

        for(unsigned long long i = 0; i < nStreams; i++)
        {
            gpuErrchk( cudaStreamCreateWithFlags(&(streams[i]), cudaStreamNonBlocking));
        }

        /*
         * We calculate how much memory we have available to use.
         * Check the size of the TotalTensors in bytes. Calculate how many
         * "passes" we need to perform. E.g. if the Total GPU is 11GB(as mine is),
         * and the total size of the addition is 24.5GB, then we need at least
         * 3 passes since 11GB->22GB->24.5GB. Since I am using streams to create
         * ~11GB for 2 arrays, and storing back in A.
         */
        for(unsigned long long i = 0; i < passes; i++)
        {
            // Offset for the next for loop to use!
            unsigned long long matrixStartOffset = matrixOffset * i;

            /*
             * Allocate total bytes for the device pointers.
             * Since using streams, we are allocating the entire tensor
             * to the GPU.
             */
            type *dev_a, *dev_b;
            gpuErrchk( cudaMalloc((void**) &dev_a, matrixOffset * totalMatrixSizeBytes));
            gpuErrchk( cudaMalloc((void**) &dev_b, matrixOffset * totalMatrixSizeBytes));
            //gpuErrchk( cudaMalloc((void**) &dev_c, matrixOffset * totalMatrixSizeBytes));

            /*
             * 1. Essentially I have broken down a stream into whole Matrix. I create a size array of the size of the whole
             * tensors(this and B) as dev_a/dev_b respectively, then dev_c as the outcome array.
             *
             * 2. streamSize is equal to the Matrices total size. E.g. A 3x4x4 Tensor -> streamSize = 16. Use this to pass
             * to the kernel, and most importantly the offset value. Offset value is passed along as an index from which
             * the kernel starts at in the device array.
             */
            for(unsigned long long j = 0; j < matrixOffset; j++)
            {

                unsigned long long offset = j * streamSize;
                gpuErrchk( cudaMemcpyAsync( &dev_a[offset], this->tensor[matrixStartOffset + j]->getArray(), streamBytes, cudaMemcpyHostToDevice, streams[matrixStartOffset + j]));
                gpuErrchk( cudaMemcpyAsync( &dev_b[offset], B.tensor[matrixStartOffset + j]->getArray(), streamBytes, cudaMemcpyHostToDevice, streams[matrixStartOffset + j]));

                // Call deviceAddMatrices with the current stream obj. Offset for the correct positions in the array.
                deviceAccumulateAddMatrices<<<std::ceil((double)streamSize/(double)blockSize),blockSize, 0, streams[matrixStartOffset + j]>>>(&dev_a[offset], &dev_b[offset], streamSize);

                // Copy Device array back to pointer.
                gpuErrchk( cudaMemcpyAsync( this->tensor[matrixStartOffset + j]->getArray(), &dev_a[offset], streamBytes, cudaMemcpyDeviceToHost, streams[matrixStartOffset + j]));

            }

            // Wait for device to finish.
            gpuErrchk( cudaDeviceSynchronize());

            // Free Memory in Device.
            gpuErrchk( cudaFree(dev_a));
            gpuErrchk( cudaFree(dev_b));
            //gpuErrchk( cudaFree(dev_c));
        }
        if(remainingMatrices != 0) {
            // Get start index of first untouched matrix.
            unsigned long long startIdx = passes * matrixOffset;

            type *dev_a, *dev_b;
            gpuErrchk( cudaMalloc((void**) &dev_a, remainingMatrices * totalMatrixSizeBytes));
            gpuErrchk( cudaMalloc((void**) &dev_b, remainingMatrices * totalMatrixSizeBytes));
            //gpuErrchk( cudaMalloc((void**) &dev_c, remainingMatrices * totalMatrixSizeBytes));

            // Start at first untouched index.
            for (unsigned long long r = 0; r < remainingMatrices; r++) {

                unsigned long long offset = r * streamSize;
                gpuErrchk( cudaMemcpyAsync( &dev_a[offset], this->tensor[startIdx + r]->getArray(), streamBytes, cudaMemcpyHostToDevice, streams[startIdx + r]));
                gpuErrchk( cudaMemcpyAsync( &dev_b[offset], B.tensor[startIdx + r]->getArray(), streamBytes, cudaMemcpyHostToDevice, streams[startIdx + r]));

                // Call deviceAddMatrices with the current stream obj. Offset for the correct positions in the array.
                deviceAccumulateAddMatrices<<<std::ceil((double)streamSize/(double)blockSize),blockSize, 0, streams[startIdx + r]>>>(&dev_a[offset], &dev_b[offset], streamSize);

                // Copy Device array back to pointer.
                gpuErrchk( cudaMemcpyAsync( this->tensor[startIdx + r]->getArray(), &dev_a[offset], streamBytes, cudaMemcpyDeviceToHost, streams[startIdx + r]));

            }

            // Wait for device.
            gpuErrchk( cudaDeviceSynchronize());

            // Free Memory in Device.
            gpuErrchk( cudaFree(dev_a));
            gpuErrchk( cudaFree(dev_b));
            //gpuErrchk( cudaFree(dev_c));
        }

        // Destroy All streams.
        for(unsigned long long i = 0; i < nStreams; i++) {
            gpuErrchk( cudaStreamDestroy(streams[i]));
        }

        // Print Total Time and Bandwidth if needed.
        if(printTime){
            std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
            float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

            std::cout << std::left << std::setw(TIME_FORMAT) << " CUDA: CUDA STREAM Tensor add: " <<
                      std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                      " ms." << std::setw(TIME_WIDTH) << (this->getTotalSize() * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
        }

    }
    else
    {
        // Generate TensorException
        std::string err = "";
        // Using getDimsExceptionString helper function.
        err += "Tensor Size Mismatch in Add, this: " + getDimsExceptionString(this->dims) + " vs " + getDimsExceptionString(B.dims);
        throw Tensor<type>::TensorException(err.c_str(), __FILE__, __LINE__,
                                            "add", "Both Tensors must match all dimensions with one another, as its element wise addition.");
    }
}

template<typename type>
void Propulsion::Tensor<type>::subtract(Propulsion::Tensor<type> &B, bool printTime)
{
    std::chrono::high_resolution_clock::time_point start;

    if(printTime)
        start = std::chrono::high_resolution_clock::now();

    // Check if dimensions are the same that way we can subtract them together.
    if(this->checkAllDimensionsMatch(&B))
    {
        for(unsigned long long i = 0; i < this->tensor.size(); i++)
        {
            this->tensor[i]->subtract(*B.tensor[i].get());
        }
    }
    else
    {
        // Generate TensorException
        std::string err = "";
        // Using getDimsExceptionString helper function.
        err += "Tensor Size Mismatch in Subtract, this: " + getDimsExceptionString(this->dims) + " vs " + getDimsExceptionString(B.dims);
        throw Tensor<type>::TensorException(err.c_str(), __FILE__, __LINE__,
                                            "subtract", "Both Tensors must match all dimensions with one another, as its element wise subtraction.");
    }


    if(printTime){
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

        std::cout << std::left << std::setw(TIME_FORMAT) << " HOST: Tensor subtraction: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (this->getTotalSize() * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }
}

template<typename type>
void Propulsion::Tensor<type>::cudaSubtract(Propulsion::Tensor<type> &B, bool printTime, bool printStats)
{
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    // Check if dimensions are the same that way we can subtract them together.
    if(this->checkAllDimensionsMatch(B))
    {
        // Declared for cudaMemGetInfo function.
        size_t free_bytes;
        size_t total_bytes;

        // Fill free and total bytes
        gpuErrchk(cudaMemGetInfo(&free_bytes, &total_bytes));

        // Get size of Matrices in Bytes.
        unsigned long long totalMatrixSizeBytes = this->tensor[0]->getTotalSize() * sizeof(type);
        unsigned long long totalKernelMemorySizeBytes = totalMatrixSizeBytes * 2; // 2 is for the allocated arrays for A,B.
        unsigned long long totalTensorsSizeBytes = totalMatrixSizeBytes * this->tensor.size() * 2;

        // Figure out how many passes we need to achieve the full tensor.
        unsigned long long passes = std::ceil((double)totalTensorsSizeBytes / (double)free_bytes);
        unsigned long long matrixOffset = std::floor( (double)this->tensor.size() / (double) passes);
        unsigned long long remainingMatrices = this->tensor.size() - matrixOffset * passes;

        // Calculate total bytes from each Matrix in Tensor.
        unsigned long long tensorTotalBytes = this->getTotalSize() * sizeof(type);

        // If prints stats are
        if(printStats) {
            std::cout << "[cudaSubtract] Total Bytes Requested: " << totalTensorsSizeBytes << std::endl;
            std::cout << "[cudaSubtract] Total Free Bytes:      " << free_bytes << std::endl;
            std::cout << "[cudaSubtract] Total GPU Bytes:       " << total_bytes << std::endl;
            std::cout << "[cudaSubtract] Total Passes:          " << passes << std::endl;
            std::cout << "[cudaSubtract] Matrix Offset:         " << matrixOffset << std::endl;
            std::cout << "[cudaSubtract] Kernel Byte Size:      " << totalKernelMemorySizeBytes << std::endl;
            std::cout << "[cudaSubtract] Remaining Matrices:    " << remainingMatrices << std::endl;
        }

        /// Delete and move to Propulsion or some Const file.
        const int blockSize = 1024;

        // Declare number of streams and create Stream array.
        unsigned long long nStreams = this->tensor.size();
        cudaStream_t *streams = new cudaStream_t[nStreams];

        // Get stream size, in this case the size of the matrix of the tensor(....nxm) => nxm = streamSize.
        unsigned long long streamSize = this->tensor[0]->getTotalSize();
        unsigned long long streamBytes = streamSize * sizeof(type);

        for(unsigned long long i = 0; i < nStreams; i++)
        {
            gpuErrchk( cudaStreamCreateWithFlags(&(streams[i]), cudaStreamNonBlocking));
        }

        /*
         * Essentially, we calculate how much memory we have available to use.
         * Check the size of the TotalTensors in bytes. Calculate how many
         * "passes" we need to perform. E.g. if the Total GPU is 11GB(as mine is),
         * and the total size of the addition is 24.5GB, then we need at least
         * 3 passes since 11GB->22GB->24.5GB. Since I am using streams to create
         * ~11GB for 2 arrays, and storing back in A.
         */
        for(unsigned long long i = 0; i < passes; i++)
        {
            // Offset for the next for loop to use!
            unsigned long long matrixStartOffset = matrixOffset * i;

            /*
             * Allocate total bytes for the device pointers.
             * Since using streams, we are allocating the entire tensor
             * to the GPU.
             */
            type *dev_a, *dev_b;
            gpuErrchk( cudaMalloc((void**) &dev_a, matrixOffset * totalMatrixSizeBytes));
            gpuErrchk( cudaMalloc((void**) &dev_b, matrixOffset * totalMatrixSizeBytes));

            /*
             * 1. Essentially I have broken down a stream into whole Matrix. I create a size array of the size of the whole
             * tensors(this and B) as dev_a/dev_b respectively, then dev_c as the outcome array.
             *
             * 2. streamSize is equal to the Matrices total size. E.g. A 3x4x4 Tensor -> streamSize = 16. Use this to pass
             * to the kernel, and most importantly the offset value. Offset value is passed along as an index from which
             * the kernel starts at in the device array.
             */
            for(unsigned long long j = 0; j < matrixOffset; j++)
            {
                unsigned long long offset = j * streamSize;
                gpuErrchk( cudaMemcpyAsync( &dev_a[offset], this->tensor[matrixStartOffset + j]->getArray(), streamBytes, cudaMemcpyHostToDevice, streams[matrixStartOffset + j]));
                gpuErrchk( cudaMemcpyAsync( &dev_b[offset], B.tensor[matrixStartOffset + j]->getArray(), streamBytes, cudaMemcpyHostToDevice, streams[matrixStartOffset + j]));

                // Call deviceSubtractMatrices with the current stream obj. Offset for the correct positions in the array.
                deviceAccumulateSubtractMatrices<<<std::ceil((double)streamSize/(double)blockSize),blockSize, 0, streams[matrixStartOffset + j]>>>(&dev_a[offset], &dev_b[offset], streamSize);

                // Copy Device array back to pointer.
                gpuErrchk( cudaMemcpyAsync( this->tensor[matrixStartOffset + j]->getArray(), &dev_a[offset], streamBytes, cudaMemcpyDeviceToHost, streams[matrixStartOffset + j]));
            }

            // Wait for device to finish.
            gpuErrchk( cudaDeviceSynchronize());

            // Free Memory in Device.
            gpuErrchk( cudaFree(dev_a));
            gpuErrchk( cudaFree(dev_b));
        }
        if(remainingMatrices != 0) {
            // Get start index of first untouched matrix.
            unsigned long long startIdx = passes * matrixOffset;

            type *dev_a, *dev_b;
            gpuErrchk( cudaMalloc((void**) &dev_a, remainingMatrices * totalMatrixSizeBytes));
            gpuErrchk( cudaMalloc((void**) &dev_b, remainingMatrices * totalMatrixSizeBytes));
            //gpuErrchk( cudaMalloc((void**) &dev_c, remainingMatrices * totalMatrixSizeBytes));

            // Start at first untouched index.
            for (unsigned long long r = 0; r < remainingMatrices; r++) {
                unsigned long long offset = r * streamSize;
                gpuErrchk( cudaMemcpyAsync( &dev_a[offset], this->tensor[startIdx + r]->getArray(), streamBytes, cudaMemcpyHostToDevice, streams[startIdx + r]));
                gpuErrchk( cudaMemcpyAsync( &dev_b[offset], B.tensor[startIdx + r]->getArray(), streamBytes, cudaMemcpyHostToDevice, streams[startIdx + r]));

                // Call deviceAddMatrices with the current stream obj. Offset for the correct positions in the array.
                deviceAccumulateSubtractMatrices<<<std::ceil((double)streamSize/(double)blockSize),blockSize, 0, streams[startIdx + r]>>>(&dev_a[offset], &dev_b[offset], streamSize);

                // Copy Device array back to pointer.
                gpuErrchk( cudaMemcpyAsync( this->tensor[startIdx + r]->getArray(), &dev_a[offset], streamBytes, cudaMemcpyDeviceToHost, streams[startIdx + r]));

                cudaMemGetInfo(&free_bytes, &total_bytes);
            }

            // Wait for device.
            gpuErrchk( cudaDeviceSynchronize());

            // Free Memory in Device.
            gpuErrchk( cudaFree(dev_a));
            gpuErrchk( cudaFree(dev_b));
            //gpuErrchk( cudaFree(dev_c));
        }

        // Destroy All streams.
        for(unsigned long long i = 0; i < nStreams; i++) {
            gpuErrchk( cudaStreamDestroy(streams[i]));
        }

        // Print Total Time and Bandwidth if needed.
        if(printTime){
            std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
            float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

            std::cout << std::left << std::setw(TIME_FORMAT) << " CUDA: CUDA STREAM Tensor subtract: " <<
                      std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                      " ms." << std::setw(TIME_WIDTH) << (this->getTotalSize() * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
        }
    }
    else
    {
        // Generate TensorException
        std::string err = "";
        // Using getDimsExceptionString helper function.
        err += "Tensor Size Mismatch in Subtract, this: " + getDimsExceptionString(this->dims) + " vs " + getDimsExceptionString(B.dims);
        throw Tensor<type>::TensorException(err.c_str(), __FILE__, __LINE__,
                                            "cudaSubtract", "Both Tensors must match all dimensions with one another, as its element wise addition.");
    }
}

template<typename type>
void Propulsion::Tensor<type>::schurProduct(Tensor <type> &B, bool printTimes)
{
    std::chrono::high_resolution_clock::time_point start;

    if(printTimes)
        start = std::chrono::high_resolution_clock::now();

    // Check if dimensions are the same that way we can add them together.
    if(this->checkAllDimensionsMatch(B))
    {
        for(unsigned long long i = 0; i < this->tensor.size(); i++)
        {
            this->tensor[i]->schurProduct(*B.tensor[i].get(), printTimes);
        }
    }
    else
    {
        // Generate TensorException
        std::string err = "";
        // Using getDimsExceptionString helper function.
        err += "Tensor Size Mismatch in SchurProduct, this: " + getDimsExceptionString(this->dims) + " vs " + getDimsExceptionString(B.dims);
        throw Tensor<type>::TensorException(err.c_str(), __FILE__, __LINE__,
                                            "schurProduct", "Both Tensors must match all dimensions with one another, as its element wise addition.");
    }

    if(printTimes){
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

        std::cout << std::left << std::setw(TIME_FORMAT) << " HOST: Tensor SchurProduct: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (this->getTotalSize() * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }
}

template<typename type>
void Propulsion::Tensor<type>::cudaSchurProduct(Propulsion::Tensor<type> &B, bool printTime, bool printStats)
{
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    // Check if dimensions are the same that way we can subtract them together.
    if(this->checkAllDimensionsMatch(B))
    {
        // Declared for cudaMemGetInfo function.
        size_t free_bytes;
        size_t total_bytes;

        // Fill free and total bytes
        gpuErrchk(cudaMemGetInfo(&free_bytes, &total_bytes));

        // Get size of Matrices in Bytes.
        unsigned long long totalMatrixSizeBytes = this->tensor[0]->getTotalSize() * sizeof(type);
        unsigned long long totalKernelMemorySizeBytes = totalMatrixSizeBytes * 2; // 2 is for the allocated arrays for A,B.
        unsigned long long totalTensorsSizeBytes = totalMatrixSizeBytes * this->tensor.size() * 2;

        // Figure out how many passes we need to achieve the full tensor.
        unsigned long long passes = std::ceil((double)totalTensorsSizeBytes / (double)free_bytes);
        unsigned long long matrixOffset = std::floor( (double)this->tensor.size() / (double) passes);
        unsigned long long remainingMatrices = this->tensor.size() - matrixOffset * passes;

        // Calculate total bytes from each Matrix in Tensor.
        unsigned long long tensorTotalBytes = this->getTotalSize() * sizeof(type);

        // If prints stats are
        if(printStats) {
            std::cout << "[cudaSchurProduct] Total Bytes Requested: " << totalTensorsSizeBytes << std::endl;
            std::cout << "[cudaSchurProduct] Total Free Bytes:      " << free_bytes << std::endl;
            std::cout << "[cudaSchurProduct] Total GPU Bytes:       " << total_bytes << std::endl;
            std::cout << "[cudaSchurProduct] Total Passes:          " << passes << std::endl;
            std::cout << "[cudaSchurProduct] Matrix Offset:         " << matrixOffset << std::endl;
            std::cout << "[cudaSchurProduct] Kernel Byte Size:      " << totalKernelMemorySizeBytes << std::endl;
            std::cout << "[cudaSchurProduct] Remaining Matrices:    " << remainingMatrices << std::endl;
        }

        /// Delete and move to Propulsion or some Const file.
        const int blockSize = 1024;

        // Declare number of streams and create Stream array.
        unsigned long long nStreams = this->tensor.size();
        cudaStream_t *streams = new cudaStream_t[nStreams];

        // Get stream size, in this case the size of the matrix of the tensor(....nxm) => nxm = streamSize.
        unsigned long long streamSize = this->tensor[0]->getTotalSize();
        unsigned long long streamBytes = streamSize * sizeof(type);

        for(unsigned long long i = 0; i < nStreams; i++)
        {
            gpuErrchk( cudaStreamCreateWithFlags(&(streams[i]), cudaStreamNonBlocking));
        }

        /*
         * We calculate how much memory we have available to use.
         * Check the size of the TotalTensors in bytes. Calculate how many
         * "passes" we need to perform. E.g. if the Total GPU is 11GB(as mine is),
         * and the total size of the addition is 24.5GB, then we need at least
         * 3 passes since 11GB->22GB->24.5GB. Since I am using streams to create
         * ~11GB for 2 arrays, and storing back in A.
         */
        for(unsigned long long i = 0; i < passes; i++)
        {
            // Offset for the next for loop to use!
            unsigned long long matrixStartOffset = matrixOffset * i;

            /*
             * Allocate total bytes for the device pointers.
             * Since using streams, we are allocating the entire tensor
             * to the GPU.
             */
            type *dev_a, *dev_b;
            gpuErrchk( cudaMalloc((void**) &dev_a, matrixOffset * totalMatrixSizeBytes));
            gpuErrchk( cudaMalloc((void**) &dev_b, matrixOffset * totalMatrixSizeBytes));

            /*
             * 1. Essentially I have broken down a stream into whole Matrix. I create a size array of the size of the whole
             * tensors(this and B) as dev_a/dev_b respectively, then dev_c as the outcome array.
             *
             * 2. streamSize is equal to the Matrices total size. E.g. A 3x4x4 Tensor -> streamSize = 16. Use this to pass
             * to the kernel, and most importantly the offset value. Offset value is passed along as an index from which
             * the kernel starts at in the device array.
             */
            for(unsigned long long j = 0; j < matrixOffset; j++)
            {
                unsigned long long offset = j * streamSize;
                gpuErrchk( cudaMemcpyAsync( &dev_a[offset], this->tensor[matrixStartOffset + j]->getArray(), streamBytes, cudaMemcpyHostToDevice, streams[matrixStartOffset + j]));
                gpuErrchk( cudaMemcpyAsync( &dev_b[offset], B.tensor[matrixStartOffset + j]->getArray(), streamBytes, cudaMemcpyHostToDevice, streams[matrixStartOffset + j]));

                // Call deviceSubtractMatrices with the current stream obj. Offset for the correct positions in the array.
                deviceAccumulateSchurProductMatrices<<<std::ceil((double)streamSize/(double)blockSize),blockSize, 0, streams[matrixStartOffset + j]>>>(&dev_a[offset], &dev_b[offset], streamSize);

                // Copy Device array back to pointer.
                gpuErrchk( cudaMemcpyAsync( this->tensor[matrixStartOffset + j]->getArray(), &dev_a[offset], streamBytes, cudaMemcpyDeviceToHost, streams[matrixStartOffset + j]));
            }

            // Wait for device to finish.
            gpuErrchk( cudaDeviceSynchronize());

            // Free Memory in Device.
            gpuErrchk( cudaFree(dev_a));
            gpuErrchk( cudaFree(dev_b));
        }
        if(remainingMatrices != 0) {
            // Get start index of first untouched matrix.
            unsigned long long startIdx = passes * matrixOffset;

            type *dev_a, *dev_b;
            gpuErrchk( cudaMalloc((void**) &dev_a, remainingMatrices * totalMatrixSizeBytes));
            gpuErrchk( cudaMalloc((void**) &dev_b, remainingMatrices * totalMatrixSizeBytes));
            //gpuErrchk( cudaMalloc((void**) &dev_c, remainingMatrices * totalMatrixSizeBytes));

            // Start at first untouched index.
            for (unsigned long long r = 0; r < remainingMatrices; r++) {
                unsigned long long offset = r * streamSize;
                gpuErrchk( cudaMemcpyAsync( &dev_a[offset], this->tensor[startIdx + r]->getArray(), streamBytes, cudaMemcpyHostToDevice, streams[startIdx + r]));
                gpuErrchk( cudaMemcpyAsync( &dev_b[offset], B.tensor[startIdx + r]->getArray(), streamBytes, cudaMemcpyHostToDevice, streams[startIdx + r]));

                // Call deviceAddMatrices with the current stream obj. Offset for the correct positions in the array.
                deviceAccumulateSchurProductMatrices<<<std::ceil((double)streamSize/(double)blockSize),blockSize, 0, streams[startIdx + r]>>>(&dev_a[offset], &dev_b[offset], streamSize);

                // Copy Device array back to pointer.
                gpuErrchk( cudaMemcpyAsync( this->tensor[startIdx + r]->getArray(), &dev_a[offset], streamBytes, cudaMemcpyDeviceToHost, streams[startIdx + r]));

                cudaMemGetInfo(&free_bytes, &total_bytes);
            }

            // Wait for device.
            gpuErrchk( cudaDeviceSynchronize());

            // Free Memory in Device.
            gpuErrchk( cudaFree(dev_a));
            gpuErrchk( cudaFree(dev_b));
            //gpuErrchk( cudaFree(dev_c));
        }

        // Destroy All streams.
        for(unsigned long long i = 0; i < nStreams; i++) {
            gpuErrchk( cudaStreamDestroy(streams[i]));
        }

        // Print Total Time and Bandwidth if needed.
        if(printTime){
            std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
            float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

            std::cout << std::left << std::setw(TIME_FORMAT) << " CUDA: CUDA STREAM Tensor cudaSchurProduct: " <<
                      std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                      " ms." << std::setw(TIME_WIDTH) << (this->getTotalSize() * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
        }

    }
    else
    {
        // Generate TensorException
        std::string err = "";
        // Using getDimsExceptionString helper function.
        err += "Tensor Size Mismatch in cudaSchurProduct, this: " + getDimsExceptionString(this->dims) + " vs " + getDimsExceptionString(B.dims);
        throw Tensor<type>::TensorException(err.c_str(), __FILE__, __LINE__,
                                            "cudaSchurProduct", "Both Tensors must match all dimensions with one another, as its element wise addition.");
    }
}

template<typename type>
void Propulsion::Tensor<type>::scalarProduct(type scalar, bool printTime, bool printStats)
{
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    // Declared for cudaMemGetInfo function.
    size_t free_bytes;
    size_t total_bytes;

    // Fill free and total bytes
    gpuErrchk(cudaMemGetInfo(&free_bytes, &total_bytes));

    // Get size of Matrices in Bytes.
    unsigned long long totalMatrixSizeBytes = this->tensor[0]->getTotalSize() * sizeof(type);
    unsigned long long totalKernelMemorySizeBytes = totalMatrixSizeBytes;
    unsigned long long totalTensorsSizeBytes = totalMatrixSizeBytes * this->tensor.size();

    // Figure out how many passes we need to achieve the full tensor.
    unsigned long long passes = std::ceil((double) totalTensorsSizeBytes / (double) free_bytes);
    unsigned long long matrixOffset = std::floor((double) this->tensor.size() / (double) passes);
    unsigned long long remainingMatrices = this->tensor.size() - matrixOffset * passes;

    // Calculate total bytes from each Matrix in Tensor.
    unsigned long long tensorTotalBytes = this->getTotalSize() * sizeof(type);

    // If prints stats are
    if (printStats) {
        std::cout << "[cudaScalar] Total Bytes Requested: " << totalTensorsSizeBytes << std::endl;
        std::cout << "[cudaScalar] Total Free Bytes:      " << free_bytes << std::endl;
        std::cout << "[cudaScalar] Total GPU Bytes:       " << total_bytes << std::endl;
        std::cout << "[cudaScalar] Total Passes:          " << passes << std::endl;
        std::cout << "[cudaScalar] Matrix Offset:         " << matrixOffset << std::endl;
        std::cout << "[cudaScalar] Kernel Byte Size:      " << totalKernelMemorySizeBytes << std::endl;
        std::cout << "[cudaScalar] Remaining Matrices:    " << remainingMatrices << std::endl;
    }

    /// Delete and move to Propulsion or some Const file.
    const int blockSize = 1024;

    // Declare number of streams and create Stream array.
    unsigned long long nStreams = this->tensor.size();
    cudaStream_t *streams = new cudaStream_t[nStreams];

    // Get stream size, in this case the size of the matrix of the tensor(....nxm) => nxm = streamSize.
    unsigned long long streamSize = this->tensor[0]->getTotalSize();
    unsigned long long streamBytes = streamSize * sizeof(type);

    for (unsigned long long i = 0; i < nStreams; i++) {
        gpuErrchk(cudaStreamCreateWithFlags(&(streams[i]), cudaStreamNonBlocking));
    }

    /*
     * Essentially, we calculate how much memory we have available to use.
     * Check the size of the TotalTensors in bytes. Calculate how many
     * "passes" we need to perform. E.g. if the Total GPU is 11GB(as mine is),
     * and the total size of the addition is 24.5GB, then we need at least
     * 3 passes since 11GB->22GB->24.5GB. Since I am using streams to create
     * ~11GB for 2 arrays, and storing back in A.
     */
    for (unsigned long long i = 0; i < passes; i++) {
        // Offset for the next for loop to use!
        unsigned long long matrixStartOffset = matrixOffset * i;

        /*
         * Allocate total bytes for the device pointers.
         * Since using streams, we are allocating the entire tensor
         * to the GPU.
         */
        type *dev_a;
        gpuErrchk(cudaMalloc((void **) &dev_a, matrixOffset * totalMatrixSizeBytes));


        /*
         * 1. Essentially I have broken down a stream into whole Matrix. I create a size array of the size of the whole
         * tensors(this and B) as dev_a/dev_b respectively, then dev_c as the outcome array.
         *
         * 2. streamSize is equal to the Matrices total size. E.g. A 3x4x4 Tensor -> streamSize = 16. Use this to pass
         * to the kernel, and most importantly the offset value. Offset value is passed along as an index from which
         * the kernel starts at in the device array.
         */
        for (unsigned long long j = 0; j < matrixOffset; j++) {
            unsigned long long offset = j * streamSize;
            gpuErrchk(cudaMemcpyAsync(&dev_a[offset], this->tensor[matrixStartOffset + j]->getArray(), streamBytes,
                                      cudaMemcpyHostToDevice, streams[matrixStartOffset + j]));

            // Call deviceSubtractMatrices with the current stream obj. Offset for the correct positions in the array.
            deviceMultiplyArrayByScalar<<<std::ceil((double)streamSize/(double)blockSize), blockSize, 0, streams[matrixStartOffset + j]>>>(
                    &dev_a[offset], scalar, streamSize);

            // Copy Device array back to pointer.
            gpuErrchk(cudaMemcpyAsync(this->tensor[matrixStartOffset + j]->getArray(), &dev_a[offset], streamBytes,
                                      cudaMemcpyDeviceToHost, streams[j]));
        }

        // Wait for device to finish.
        gpuErrchk(cudaDeviceSynchronize());

        // Free Memory in Device.
        gpuErrchk(cudaFree(dev_a));
    }
    if (remainingMatrices != 0) {
        // Get start index of first untouched matrix.
        unsigned long long startIdx = passes * matrixOffset;

        type *dev_a;
        gpuErrchk(cudaMalloc((void **) &dev_a, remainingMatrices * totalMatrixSizeBytes));


        // Start at first untouched index.
        for (unsigned long long r = 0; r < remainingMatrices; r++) {
            unsigned long long offset = r * streamSize;
            gpuErrchk(cudaMemcpyAsync(&dev_a[offset], this->tensor[startIdx + r]->getArray(), streamBytes,
                                      cudaMemcpyHostToDevice, streams[startIdx + r]));

            // Call deviceAddMatrices with the current stream obj. Offset for the correct positions in the array.
            deviceMultiplyArrayByScalar<<<std::ceil((double)streamSize/(double)blockSize), blockSize, 0, streams[startIdx + r]>>>(
                    &dev_a[offset], scalar, streamSize);

            // Copy Device array back to pointer.
            gpuErrchk(cudaMemcpyAsync(this->tensor[startIdx + r]->getArray(), &dev_a[offset], streamBytes,
                                      cudaMemcpyDeviceToHost, streams[startIdx + r]));
        }

        // Wait for device.
        gpuErrchk(cudaDeviceSynchronize());

        // Free Memory in Device.
        gpuErrchk(cudaFree(dev_a));
    }

    // Print Total Time and Bandwidth if needed.
    if(printTime){
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

        std::cout << std::left << std::setw(TIME_FORMAT) << " CUDA: CUDA STREAM Tensor cudaSchurProduct: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (this->getTotalSize() * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }
}

template<typename type>
Propulsion::Tensor<type> Propulsion::Tensor<type>::operator+(Tensor <type> &rhs)
{
    // Check if dimensions are the same that way we can add them together.
    if(!this->checkAllDimensionsMatch(rhs)) {
        // Generate TensorException
        std::string err = "";
        // Using getDimsExceptionString helper function.
        err += "Tensor Size Mismatch in Add, this: " + getDimsExceptionString(this->dims) + " vs " + getDimsExceptionString(rhs.dims);
        throw Tensor<type>::TensorException(err.c_str(), __FILE__, __LINE__,
                                            "operator+", "Both Tensors must match all dimensions with one another, as its element wise addition.");
        return *this;
    }

    // Copy this to return that way its non modifying.
    Tensor<type> ret = *this;

    // Use CUDA to add the Tensors.
    ret.cudaAdd(rhs, false, false);

    return ret;
}

template<typename type>
Propulsion::Tensor<type> Propulsion::Tensor<type>::operator-(Tensor <type> &rhs)
{
    // Check if dimensions are the same that way we can add them together.
    if(!this->checkAllDimensionsMatch(rhs)) {
        // Generate TensorException
        std::string err = "";
        // Using getDimsExceptionString helper function.
        err += "Tensor Size Mismatch in Subtract, this: " + getDimsExceptionString(this->dims) + " vs " + getDimsExceptionString(rhs.dims);
        throw Tensor<type>::TensorException(err.c_str(), __FILE__, __LINE__,
                                            "operator-", "Both Tensors must match all dimensions with one another, as its element wise addition.");
        return *this;
    }

    // Copy this to return that way its non modifying.
    Tensor<type> ret = *this;

    // Use CUDA to subtract the Tensors.
    ret.cudaSubtract(rhs, false, false);

    return ret;
}

template<typename type>
Propulsion::Tensor<type> Propulsion::Tensor<type>::operator*(Tensor <type> &rhs)
{
    // Copy this to return that way its non modifying.
    Tensor<type> ret = *this;

    // Use CUDA to dot product the Tensors.
    ret.cudaDotProduct(rhs, false, false);

    return ret;
}