//
// Created by steve on 8/23/2021.
//
#pragma once

template<typename type>
void Propulsion::Tensor<type>::dotProduct(Propulsion::Tensor<type> &B, bool printTime)
{
    if(!checkThirdDimensionsUpMatch(B))
    {
        // Generate TensorException
        std::string err = "";
        // Using getDimsExceptionString helper function.
        err += "Tensor Size Mismatch in cudaDotProduct, this: " + getDimsExceptionString(this->dims) + " vs " + getDimsExceptionString(B.dims);
        throw Tensor<type>::TensorException(err.c_str(), __FILE__, __LINE__,
                                            "cudaDotProduct", "Both Tensors must match all dimensions above 2nd. ");
    }

    std::chrono::high_resolution_clock::time_point start;

    if(printTime)
        start = std::chrono::high_resolution_clock::now();

    // Check if dimensions are the same that way we can multiply them together.
    if(this->getTotalDims() == 1 && B.getTotalDims() == 1)
    {
        this->tensor[0]->cudaDotProduct(*B.tensor[0]);
    }
    else if(this->getTotalDims() == B.getTotalDims())
    {
        unsigned long long rowIdx = this->dims.size() - 2;
        unsigned long long colIdx = this->dims.size() - 1;

        if(this->dims[colIdx] == B.dims[rowIdx])
        {
            for(unsigned i = 0; i < this->tensor.size(); i++)
            {
                this->tensor[i]->dot(*B.tensor[i]);
            }
        }
        else
        {
            // Generate TensorException
            std::string err = "";
            // Using getDimsExceptionString helper function.
            err += "Tensor Size Mismatch in Dot Prodcut, this: " + getDimsExceptionString(this->dims) + " vs " + getDimsExceptionString(B.dims);
            throw Tensor<type>::TensorException(err.c_str(), __FILE__, __LINE__,
                                                "dotProduct", "Both Tensors must match mxn * nxk standards for dot product for dimensions 1 and 2.");
        }
    }

    if(printTime){
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

        std::cout << std::left << std::setw(TIME_FORMAT) << " HOST: Tensor Dot Product: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (this->getTotalSize() * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }
}

template<typename type>
void Propulsion::Tensor<type>::cudaDotProduct(Tensor <type> &B, bool printTime, bool printStats)
{
    if(!checkThirdDimensionsUpMatch(B))
    {
        // Generate TensorException
        std::string err = "";
        // Using getDimsExceptionString helper function.
        err += "Tensor Size Mismatch in cudaDotProduct, this: " + getDimsExceptionString(this->dims) + " vs " + getDimsExceptionString(B.dims);
        throw Tensor<type>::TensorException(err.c_str(), __FILE__, __LINE__,
                                            "cudaDotProduct", "Both Tensors must match all dimensions above 2nd. ");
    }

    std::chrono::high_resolution_clock::time_point start;

    if(printTime)
        start = std::chrono::high_resolution_clock::now();

    if(this->getTotalDims() <= 2)
    {
        this->tensor[0]->cudaDotProduct(*B.tensor[0]);
    }
    else
    {
        unsigned long long rowIdx = this->dims.size() - 2;
        unsigned long long colIdx = this->dims.size() - 1;

        // Perform
        if(this->dims[colIdx] == B.dims[rowIdx])
        {
            // Create C Tensor with total size of the tensor array with rows from this, and cols from
            // B to follow do product rules.
            Tensor<type> C(this->tensor.size() , this->dims[rowIdx], B.dims[colIdx]);
            // Then copy dims from this to C, set last two dims to A rows, and B cols.
            C.dims = this->dims;
            C.dims[rowIdx] = this->dims[rowIdx];
            C.dims[colIdx] = B.dims[colIdx];

            // Declared for cudaMemGetInfo function.
            size_t free_bytes;
            size_t total_bytes;

            // Fill free and total bytes
            gpuErrchk(cudaMemGetInfo(&free_bytes, &total_bytes));

            // Get Matrix size of each tensor.
            unsigned long long aTotalMatrixSizeBytes = this->tensor[0]->getTotalSize() * sizeof(type);
            unsigned long long bTotalMatrixSizeBytes = B.tensor[0]->getTotalSize() * sizeof(type);
            unsigned long long cTotalMatrixSizeBytes = C.tensor[0]->getTotalSize() * sizeof(type);

            unsigned long long totalKernelMemorySizeBytes = aTotalMatrixSizeBytes + bTotalMatrixSizeBytes + cTotalMatrixSizeBytes;
            unsigned long long totalTensorsSizeBytes = totalKernelMemorySizeBytes * this->tensor.size();

            unsigned long long passes = std::ceil((double)totalTensorsSizeBytes / (double)free_bytes);
            unsigned long long matrixOffset = std::floor( (double)this->tensor.size() / (double) passes);
            unsigned long long remainingMatrices = this->tensor.size() - matrixOffset * passes;

            if(printStats) {
                std::cout << "[cudaDotProduct] Total Bytes Requested: " << totalTensorsSizeBytes << std::endl;
                std::cout << "[cudaDotProduct] Total Free Bytes:      " << free_bytes << std::endl;
                std::cout << "[cudaDotProduct] Total GPU Bytes:       " << total_bytes << std::endl;
                std::cout << "[cudaDotProduct] Total Passes:          " << passes << std::endl;
                std::cout << "[cudaDotProduct] Matrix Offset:         " << matrixOffset << std::endl;
                std::cout << "[cudaDotProduct] Kernel Byte Size:      " << totalKernelMemorySizeBytes << std::endl;
                std::cout << "[cudaDotProduct] Remaining Matrices:    " << remainingMatrices << std::endl;
            }

            // Declare number of streams and create Stream array.
            unsigned long long nStreams = this->tensor.size();
            cudaStream_t *streams = new cudaStream_t[nStreams];

            // Get stream size, in this case the size of the matrix of the tensor(....nxm) => nxm = streamSize.
            unsigned long long aStreamSize = this->tensor[0]->getTotalSize();
            unsigned long long bStreamSize = B.tensor[0]->getTotalSize();
            unsigned long long cStreamSize = C.tensor[0]->getTotalSize();
            unsigned long long aStreamBytes = aStreamSize * sizeof(type);
            unsigned long long bStreamBytes = bStreamSize * sizeof(type);
            unsigned long long cStreamBytes = cStreamSize * sizeof(type);

            for(unsigned long long i = 0; i < nStreams; i++)
            {
                gpuErrchk( cudaStreamCreateWithFlags(&(streams[i]), cudaStreamNonBlocking));
            }



            // block of threads.
            dim3 block_dim(32,32);

            // total blocks of threads in x direction
            unsigned blocksX = std::ceil(( (double)B.dims[colIdx]) / ( (double)block_dim.x) );
            // total blocks of threads in y direction
            unsigned blocksY = std::ceil(( (double)this->dims[rowIdx]) / ( (double)block_dim.y) );
            dim3 grid_dim(blocksX, blocksY);

            /*
             *
             * We calculate how much memory we have available to use.
             * Check the size of the TotalTensors in bytes. Calculate how many
             * "passes" we need to perform. E.g. if the Total GPU is 11GB(as mine is),
             * and the total size of the addition is 24.5GB, then we need at least
             * 3 passes since 11GB->22GB->24.5GB. Since I am using streams to create
             * ~11GB for 2 arrays, and storing back in A.
             */
            for(unsigned long long i = 0; i < passes; i++) {
                // Offset for the next for loop to use!
                unsigned long long matrixStartOffset = matrixOffset * i;

                /*
                 * Allocate total bytes for the device pointers.
                 * Since using streams, we are allocating the entire tensor
                 * to the GPU.
                 */
                type *dev_a, *dev_b, *dev_c;
                gpuErrchk(cudaMalloc((void **) &dev_a, matrixOffset * aTotalMatrixSizeBytes));
                gpuErrchk(cudaMalloc((void **) &dev_b, matrixOffset * bTotalMatrixSizeBytes));
                gpuErrchk(cudaMalloc((void **) &dev_c, matrixOffset * cTotalMatrixSizeBytes));

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
                    unsigned long long aOffset = j * aStreamSize;
                    unsigned long long bOffset = j * bStreamSize;
                    unsigned long long cOffset = j * cStreamSize;

                    gpuErrchk( cudaMemcpyAsync( &dev_a[aOffset], this->tensor[matrixStartOffset + j]->getArray(), aStreamBytes, cudaMemcpyHostToDevice, streams[matrixStartOffset + j]));
                    gpuErrchk( cudaMemcpyAsync( &dev_b[bOffset], B.tensor[matrixStartOffset + j]->getArray(), bStreamBytes, cudaMemcpyHostToDevice, streams[matrixStartOffset + j]));

                    // Call deviceAddMatrices with the current stream obj. Offset for the correct positions in the array.

                    deviceDotProduct<<<grid_dim,block_dim, 0, streams[matrixStartOffset + j]>>>(&dev_a[aOffset], &dev_b[bOffset], &dev_c[cOffset], this->dims[rowIdx], this->dims[colIdx], B.dims[colIdx]);

                    // Copy Device array back to pointer.
                    gpuErrchk( cudaMemcpyAsync( C.tensor[matrixStartOffset + j]->getArray(), &dev_c[cOffset], cStreamBytes, cudaMemcpyDeviceToHost, streams[matrixStartOffset + j]));

                }

                // Wait for a stream to finish, when finished, move the corresponding data into this.
                for(unsigned long long j = 0; j < matrixOffset; j++)
                {
                    gpuErrchk(cudaStreamSynchronize(streams[matrixStartOffset + j]));
                    this->tensor[matrixStartOffset + j] = std::move(C.tensor[matrixStartOffset + j]);
                }

                // Wait for device to finish.
                gpuErrchk( cudaDeviceSynchronize());

                // Free Memory in Device.
                gpuErrchk( cudaFree(dev_a));
                gpuErrchk( cudaFree(dev_b));
                gpuErrchk( cudaFree(dev_c));

            }
            if(remainingMatrices != 0) {
                // Get start index of first untouched matrix.
                unsigned long long startIdx = passes * matrixOffset;

                type *dev_a, *dev_b, *dev_c;
                gpuErrchk( cudaMalloc((void**) &dev_a, remainingMatrices * aTotalMatrixSizeBytes));
                gpuErrchk( cudaMalloc((void**) &dev_b, remainingMatrices * bTotalMatrixSizeBytes));
                gpuErrchk( cudaMalloc((void**) &dev_c, remainingMatrices * cTotalMatrixSizeBytes));

                // Start at first untouched index.
                for (unsigned long long r = 0; r < remainingMatrices; r++) {

                    unsigned long long aOffset = r * aStreamSize;
                    unsigned long long bOffset = r * bStreamSize;
                    unsigned long long cOffset = r * cStreamSize;

                    gpuErrchk( cudaMemcpyAsync( &dev_a[aOffset], this->tensor[startIdx + r]->getArray(), aStreamBytes, cudaMemcpyHostToDevice, streams[startIdx + r]));
                    gpuErrchk( cudaMemcpyAsync( &dev_b[bOffset], B.tensor[startIdx + r]->getArray(), bStreamBytes, cudaMemcpyHostToDevice, streams[startIdx + r]));

                    // Call deviceAddMatrices with the current stream obj. Offset for the correct positions in the array.
                    //cudaDotProduct<<<std::ceil((double)streamSize/(double)blockSize),blockSize, 0, streams[startIdx + r]>>>(&dev_a[offset], &dev_b[offset], streamSize);
                    deviceDotProduct<<<grid_dim,block_dim, 0, streams[startIdx + r]>>>(&dev_a[aOffset], &dev_b[bOffset], &dev_c[cOffset], this->dims[rowIdx], this->dims[colIdx], B.dims[colIdx]);

                    // Copy Device array back to pointer.
                    gpuErrchk( cudaMemcpyAsync( C.tensor[startIdx + r]->getArray(), &dev_c[cOffset], cStreamBytes, cudaMemcpyDeviceToHost, streams[startIdx + r]));

                }

                // Wait for a stream to finish, when finished, move the C tensor elements into this!
                for(unsigned long long j = 0; j < remainingMatrices; j++)
                {
                    gpuErrchk(cudaStreamSynchronize(streams[startIdx + j]));
                    this->tensor[startIdx + j] = std::move(C.tensor[startIdx + j]);
                }

                // Wait for device.
                gpuErrchk( cudaDeviceSynchronize());

                // Move C into A now.
                this->dims = C.dims;
                for(unsigned long long i = 0; i < this->tensor.size(); i++)
                {
                    this->tensor[i] = std::move(C.tensor[i]);
                }

                // Free Memory in Device.
                gpuErrchk( cudaFree(dev_a));
                gpuErrchk( cudaFree(dev_b));
                gpuErrchk( cudaFree(dev_c));
            }

            // Destroy All streams.
            for(unsigned long long i = 0; i < nStreams; i++) {
                gpuErrchk( cudaStreamDestroy(streams[i]));
            }


        }
        else
        {
            // Generate TensorException
            std::string err = "";
            // Using getDimsExceptionString helper function.
            err += "Tensor Size Mismatch in cudaDotProduct, this column must be the same size as others row size: " + getDimsExceptionString(this->dims) + " vs " + getDimsExceptionString(B.dims);
            throw Tensor<type>::TensorException(err.c_str(), __FILE__, __LINE__,
                                                "cudaDotProduct", "Both Tensors must have matching row to columns in respect of this to other. ");
        }
    }

    if(printTime){
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

        std::cout << std::left << std::setw(TIME_FORMAT) << " CUDA: Tensor Dot Product: " <<
                  std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                  " ms." << std::setw(TIME_WIDTH) << (this->getTotalSize() * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
    }
}