//
// Created by steve on 3/9/2021.
//
#include "../Propulsion.cuh"
#include "TensorHelpers.cu"


/*
 * Template packing verification making sure that the value(s) provided as
 * the dims are in fact convertible to unsigned values.
 *
 * E.g. Tensor T(10, 4, 4)      // FINE
 *      Tensor T(10, 4, 4.1, 3) // FINE but note that 4.1 is converted to unsigned int.
 *      Tensor T(10, 3, 3, "a") // FAILS
 *
 * Uses:    Tensor - Constructors
 *          at
 */
template<typename... Ts>
using AllUnsigned = typename
std::enable_if<std::conjunction<std::is_convertible<Ts, unsigned long long>...>::value>::type;

template<typename type>
class Propulsion::Tensor {
private:
    std::deque<std::unique_ptr<Propulsion::Matrix<type>>> tensor;
    std::deque<unsigned long long> dims;

    friend class Propulsion::Matrix<type>;
public:
    /**
     * Class:        TensorException
     *
     * Purpose:         The purpose of this class is for the user to be able to handle
     *              exceptions thrown for the Tensor class. These exceptions can be thrown
     *              for many various reasons all described in the throw. Child class of
     *              std::exception to inherit various methods.
     */
    class TensorException : public std::exception
    {
    private:
        const char* file;
        int line;
        const char* func;
        const char* info;
    public:
        TensorException(const char* msg, const char* file_,
                        int line_, const char* func_, const char* info_ = "") :
                std::exception(msg),
                file(file_),
                line(line_),
                func(func_),
                info(info_){}


        const char* get_file() const { return file; }
        int get_line() const { return line; }
        const char* get_func() const { return func; }
        const char* get_info() const { return info; }
    };

    /**
     * \brief       Constructor method for creating a Tensor Object. Args is template packed
     *
     * \details         Allow the user to construct an any dimension tensor for various reasons.
     *              Uses the last two dimensions (if) provided to create a Matrix object, which
     *              is then stored in a deque to dynamically allocate the tensor.
     *
     * \example     Propulsion::Tensor<int> T(6, 3, 4, 4); // Creates a 6x3x4x4 Tensor
     *
     * @throws      Propulsion::Tensor<type>::TensorException if zero is passed as an argument.
     *
     * @params      ...args A template packed argument for generating a tensor
     *              of the dims given. Expected values are to be a type unsigned long long.
     */
    template<typename... Ts, typename = AllUnsigned<Ts...>>
    Tensor(Ts const&... args)
    {
        // Unpack parameters.
        long long unsigned values[] = {(unsigned)args...};

        // Evaluate the values in values[] by placing them in dims container.
        for(auto v : values)
        {
            if(v == 0)
            {
                std::string err = "Tensor Dimension Error: 0 is not a valid size for a dimension.";
                throw Propulsion::Tensor<type>::TensorException(err.c_str(), __FILE__, __LINE__,
                                                                "Tensor", "Tensor Constructor requires that all dims be at least > 0");
            }
            this->dims.push_back(v);
        }

        // Now that the dims are stored, check if only 1 or 2 Dimensional Tensor
        if(this->dims.size() == 1)
        {
            this->tensor.push_back( std::unique_ptr<Propulsion::Matrix<type>> ( new Propulsion::Matrix<type>( 1, this->dims[0])));
        }
        else if(this->dims.size() == 2)
        {
            this->tensor.push_back( std::unique_ptr<Propulsion::Matrix<type>> ( new Propulsion::Matrix<type>( this->dims[0], this->dims[1])));
        }
        // Else its a Kx...MxN tensor.
        else
        {
            // Get the last two dimensions indexes.
            unsigned long long rowsIdx = this->dims.size() - 2;
            unsigned long long colsIdx = this->dims.size() - 1;

            // Calc. and store the total amount of matrices that must be created.
            unsigned long long totalM = 1;
            for(unsigned long long i = 0; i < this->dims.size() - 2; i++)
            {
                totalM *= this->dims[i];
            }

            // Now we populate tensor deque with matrices.
            for(unsigned long long i = 0; i < totalM; i++)
            {
                this->tensor.push_back( std::unique_ptr<Propulsion::Matrix<type>> ( new Propulsion::Matrix<type>( this->dims[rowsIdx], this->dims[colsIdx])));
            }
        }
    }

    /**
     * \brief           Copies the contents of the param Tensor into the other via
     *              a deep copy.
     *
     * \details         Deep copies the contents of the parameter Tensor into this
     *              Tensor. Uses Matrix provided deep copy method to copy each
     *              Matrix Obj. part of the Tensor.
     *
     * \example     Tensor<int> T(3,4,4);                                       <br>
     *              Tensor<int> copy = T;
     *
     * @throws      None.
     *
     * @param       copyT The Tensor being deep copied into this.
     *
     * @see         Tensor(...args)
     */
    Tensor(const Tensor& copyT)
    {
        // Copy Contents of the dims using deque operator=
        this->dims = copyT.dims;

        // If Matrix size is 2 or greater.
        if(this->dims.size() >= 2) {
            unsigned long long rowIdx = this->dims.size() - 2;
            unsigned long long colIdx = this->dims.size() - 1;

            for (unsigned i = 0; i < copyT.tensor.size(); i++) {
                this->tensor.push_back(std::unique_ptr<Propulsion::Matrix<type>>(
                        new Propulsion::Matrix<type>(this->dims[rowIdx], this->dims[colIdx])));
                this->tensor[i]->operator=(*copyT.tensor[i]);
            }
        }
        // else its size of 1.
        else if(this->dims.size() == 1)
        {
            unsigned long long colIdx = this->dims.size() - 1;
            this->tensor.push_back(std::unique_ptr<Propulsion::Matrix<type>>(
                        new Propulsion::Matrix<type>(this->dims[colIdx])));
            this->tensor[0]->operator=(*copyT.tensor[0]);
        }
    }

    /**
     * \brief           Shallow copy a Tensor Object that is of rvalue.
     *
     * \details         Shallow copy a Tensor Object that is of rvalue.
     *              Uses std::move to execute the shallow copy and deque
     *              operator= to copy the deque contents.
     *
     * @param       moveT Tensor obj meant to be shallow copied into this.
     */
    Tensor(Tensor&& moveT)
    {
        // set this dims to moveT.dims
        this->dims = moveT.dims;

        for (unsigned i = 0; i < moveT.tensor.size(); i++) {
            this->tensor.push_back(std::unique_ptr<Propulsion::Matrix<type>>(nullptr));
            this->tensor[i] = std::move(moveT.tensor[i]);
        }
    }

    /**
     * \brief           Returns a reference of the type from the given input as a
     *              k x ... x m x n indexing.
     *
     * \details         Return a reference of the index from the packed template
     *              args. This reference is so that values may be modified at will.
     *              Does not throw any exceptions. This method is intended only for
     *              the bold who write perfect code all the time. The index of the
     *              Matrix we index to, is calculated via:
     *              Kx...LxMxN. We drop M and N as they're only needed for the Matrix
     *              and we are left with Kx...xL. Which we use with the this->dim
     *              deque to calculate what index we need to be at.
     *
     * \example     Tensor<int> T(4,4,4,4); // Create 4x4x4x4 Tensor<br>
     *              T(3,2,1,0) = 123;       // Sets the Matrix at 3x2(14th Matrix) at i=1, j=0 to 123.
     *
     * @throws      None.
     *
     * @param       ...args Template packed parameter that only accepts unsigned
     *              long long convertible values.
     *
     * @see         at()
     *
     * @return      Reference of type value from the given index.
     */
    template<typename... Ts, typename = AllUnsigned<Ts...>>
    type& operator()(Ts const&... args) noexcept
    {
        // Unpack parameters.
        long long unsigned values[] = {(unsigned)args...};
        long long unsigned totalArgs = sizeof(values) / sizeof(unsigned long long);

        // Current index
        unsigned long long index = 0;

        // If the dimensions are of size 2 or 1.
        if(this->dims.size() <= 2)
        {
            if(totalArgs == 2)
            {
                return this->tensor[0]->operator()(values[0], values[1]);
            }
            else if(totalArgs == 1)
            {
                return this->tensor[0]->operator()(values[0]);
            }
        }
        else
        {
            // Find the index value of the tensor deque, that way we can return that deque value with
            // final two indexed provided in value as the rows and cols.
            for(unsigned i = 0; i < totalArgs - 2; i++)
            {
                // Store the first value of the desired index, that
                // way we can take the product of the following indices to
                // find which Matrix we need to be at.
                unsigned long long temp = values[i];

                for(unsigned j = i + 1; j < totalArgs - 2; j++)
                {
                    temp *= this->dims[j];
                }

                // Add to index val.
                index += temp;
            }
        }

        return this->tensor[index]->operator()(values[totalArgs - 2], values[totalArgs - 1]);
    };

    /**
     * \brief           Returns a reference of the type from the given input as a
     *              k x ... x m x n indexing. Throws TensorException if needed.
     *
     * \details         Return a reference of the index from the packed template
     *              args. This reference is so that values may be modified at will.
     *              Will throw an exception for incomplete index params and for out
     *              of range indices. The index of the Matrix we index to, is calculated
     *              via: Kx...LxMxN. We drop M and N as they're only needed for the
     *              Matrix and we are left with Kx...xL. Which we use with the this->dim
     *              deque to calculate what index we need to be at.
     *
     * \example     Tensor<int> T(4,4,4,4); // Create 4x4x4x4 Tensor            <br>
     *              T(3,2,1,0) = 123;       // Sets the Matrix at 3x2(14th Matrix) at i=1, j=0 to 123.
     *
     * @throws      TensorException If the supplied index is out-of-bounds or an incomplete list.
     *
     * @param       ...args Template packed parameter that only accepts unsigned
     *              long long convertible values.
     *
     * @see         operate() for no exception
     *
     * @return      Reference of type value from the given index.
     */
    template<typename... Ts, typename = AllUnsigned<Ts...>>
    type at(Ts const&... args)
    {
        // Unpack parameters.
        long long unsigned values[] = {(unsigned)args...};
        long long unsigned totalArgs = sizeof(values) / sizeof(unsigned long long);

        // Current index
        unsigned long long index = 0;

        // Check if the args provided match the dimensions of the tensor.
        // If they dont, throw a TensorException.
        if(totalArgs != dims.size())
        {
            std::string err = "";
            err += "Tensor Access Size Mismatch, user requested: ( ";
            for(unsigned i = 0; i < totalArgs - 1; i++)
            {
                err += std::to_string(values[i]) + " ,";
            }
            err += std::to_string(values[totalArgs - 1]) +  ")" + " vs: ( ";
            for(unsigned i = 0; i < this->dims.size() - 1; i++)
            {
                err += std::to_string(this->dims[i]) + ", ";
            }
            err += std::to_string(this->dims[this->dims.size()-1]) + ")";


            throw Tensor<type>::TensorException(err.c_str(), __FILE__, __LINE__,
                                                "at", "User specified an incomplete element. E.g, if you specify ( 2, 4) when the Tensor is of size ( 3, 5, 5), it doesn't know how to resolve that.");
            return (type) 0.0;
        }

        for(unsigned i = 0; i < totalArgs; i++)
        {
            if(values[i] >= this->dims[i])
            {
                std::string err = "Tensor Access Element out of Bounds, user requested: (";
                for(unsigned i = 0; i < totalArgs - 1; i++)
                {
                    err += std::to_string(values[i]) + " ,";
                }
                err += std::to_string(values[totalArgs - 1]) + ")" + " vs: ";

                err += "( ";
                for(unsigned i = 0; i < dims.size(); i++)
                {
                    if(i != dims.size() - 1)
                        err += std::to_string(dims[i]) + ", ";
                    else
                        err += std::to_string(dims[i]);
                }
                err += ") Dims.";

                throw Tensor<type>::TensorException(err.c_str(), __FILE__, __LINE__,
                                                    "at", "User specified an element of a dimension that exceeds it's Max Index. E.g. if Tensor(2,2,2) exists, then the user shouldn't allowed to access (2,1,1) since 2 >= 2 on index 0.");

                return (type) 0.0;
            }
        }

        // If the dimensions are of size 2 or 1.
        if(this->dims.size() <= 2)
        {
            if(totalArgs == 2)
            {
                return this->tensor[0]->operator()(values[0], values[1]);
            }
            else if(totalArgs == 1)
            {
                return this->tensor[0]->operator()(values[0]);
            }
        }
        else
        {
            // Find the index value of the tensor deque, that way we can return that deque value with
            // final two indexed provided in value as the rows and cols.
            for(unsigned i = 0; i < totalArgs - 2; i++)
            {
                // Store the first value of the desired index, that
                // way we can take the product of the following indices to
                // find which Matrix we need to be at.
                unsigned long long temp = values[i];

                for(unsigned j = i + 1; j < totalArgs - 2; j++)
                {
                    temp *= this->dims[j];
                }

                // Add to index val.
                index += temp;
            }
        }

        return this->tensor[index]->operator()(values[totalArgs - 2], values[totalArgs - 1]);
    }

    /**
     * \brief           Checks if the dims are all the same and returns T/F.
     *
     * \details         Checks if the dims are all the same and returns T/F.
     *              If the dims aren't the same size, checking element wise
     *              is skipped and returned false right away.
     *
     * @throws      None.
     *
     * @param       second Second Tensor to be compared against.
     *
     * @return      true/false if the tensors dims are the exact same.
     */
    bool checkAllDimensionsMatch(Propulsion::Tensor<type> &second)
    {
        // Check first if the total dimensions are the same.
        if(this->getTotalDims() != second.getTotalDims() )
        {
            return false;
        }
        // Last check if all dims match.
        else
        {
            for(unsigned long long i = 0; i < this->getTotalDims(); i++)
            {
                if(this->dims[i] != second.dims[i])
                    return false;
                else
                    continue;
            }
        }
        return true;
    }

    bool checkThirdDimensionsUpMatch(Propulsion::Tensor<type> &second)
    {
        // Check if the total dims are the same.
        if(this->getTotalDims() != second.getTotalDims())
        {
            return false;
        }
        // Last check if the dims from N....3rd Dim all match.
        else
        {
            if(this->getTotalDims() > 2) {
                for (unsigned long long i = 0; i < this->getTotalDims() - 2; i++) {
                    if(this->dims[i] != second.dims[i])
                        return false;
                    else
                        continue;
                }
            }
        }
        return true;
    }

    void dotProduct(Propulsion::Tensor<type> &B, bool printTime = false)
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

        // Check if dimensions are the same that way we can subtract them together.
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
                    this->tensor[i]->cudaDotProduct(*B.tensor[i]);
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
        }

        if(printTime){
            std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
            float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

            std::cout << std::left << std::setw(TIME_FORMAT) << " HOST: Tensor Dot Product: " <<
                      std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                      " ms." << std::setw(TIME_WIDTH) << (this->getTotalSize() * sizeof(type)) / milliseconds / 1e6 << " GB/s" << std::endl;
        }
    }


    void cudaDotProduct(Tensor<type> &B, bool printTime = false, bool printStats = false)
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
                    std::cout << "[cudaAdd] Total Bytes Requested: " << totalTensorsSizeBytes << std::endl;
                    std::cout << "[cudaAdd] Total Free Bytes:      " << free_bytes << std::endl;
                    std::cout << "[cudaAdd] Total GPU Bytes:       " << total_bytes << std::endl;
                    std::cout << "[cudaAdd] Total Passes:          " << passes << std::endl;
                    std::cout << "[cudaAdd] Matrix Offset:         " << matrixOffset << std::endl;
                    std::cout << "[cudaAdd] Kernel Byte Size:      " << totalKernelMemorySizeBytes << std::endl;
                    std::cout << "[cudaAdd] Remaining Matrices:    " << remainingMatrices << std::endl;
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

    /**
     * \brief           Adds the two tensors together, stores the summation
     *              in this tensor.
     *
     * \details         Uses Matrix add method to perform addition across all the
     *              Matrices in the Tensor deque. For each Matrix Element, call
     *              add for the same element-wise Matrix. The Tensors must both
     *              match in size for all dimensions. Typically, if the tensor is
     *              of a smaller size this method is the way to go as the Matrix
     *              add will decide whether or not to use CUDA or HOST. If the
     *              Tensor 3rd and up dimensions are relatively big compared to
     *              the Rows, Cols Dimensions... then I suggest using cudaAdd()
     *              as that takes advantage of streams which will speed up memory
     *              transfers.
     *
     * @throws      TensorException If the Tensor being added with does not match
     *              the dimension sizes of this.
     *
     * @param       B Tensor to be added against this.
     * @param       printTime bool for the method to print statistics about time. Default false.
     *              method overhead time.
     *
     * @see         cudaAdd(Tensor B)
     *
     * @return      None.
     */
    void add(Propulsion::Tensor<type> &B, bool printTime = false)
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

    /**
     * \brief           Adds this to Tensor B if they're the same dimensions. Throws
     *              otherwise a TensorException.
     *
     * \details         Adds this to Tensor B if the dimensions match. Uses CUDA streams
     *              to speed up operations roughly to 3x(for my setup) from the other
     *              method (add) when using CUDA as well. Essentially this method determines
     *              whether or not it can add B to itself. If it can, it then determines how
     *              many "passes" are needed to achieve adding both tensors to one another.
     *              Then using that, we can theoretically use the GPU for any size Tensor as
     *              long as the both Tensors ROWS and COL Dimensions can be fit in the GPU memory
     *              at any given time.
     *
     * \example     Tensor I(1000, 3, 224, 224);                                    <br>
     *              Tensor R(1000, 3, 224, 224);                                    <br>
     *              I.cudaAdd(R);       // I is modified, R is left alone.
     *
     * @throws      TensorException If the Tensor being added with does not match
     *              the dimension sizes of this.
     *
     * @param       B Tensor to add to this.
     * @param       printTime Bool to print stats about the overall operation overhead. Default false.
     * @param       printStats Print More Memory related information.
     *
     * @see         add(Tensor B)
     */
    void cudaAdd(Propulsion::Tensor<type> &B, bool printTime = false, bool printStats = false)
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

    /**
     * \brief           Subtract the two tensors together, stores the difference
     *              in this tensor.
     *
     * \details         Uses Matrix add method to perform subtraction across all the
     *              Matrices in the Tensor deque. For each Matrix Element, call
     *              subtract for the same element-wise Matrix. The Tensors must both
     *              match in size for all dimensions. Typically, if the tensor is
     *              of a smaller size this method is the way to go as the Matrix
     *              add will decide whether or not to use CUDA or HOST. If the
     *              Tensor 3rd and up dimensions are relatively big compared to
     *              the Rows, Cols Dimensions... then I suggest using cudaSubtract()
     *              as that takes advantage of streams which will speed up memory
     *              transfers.
     *
     * @throws      TensorException If the Tensor being subtracted with does not match
     *              the dimension sizes of this.
     *
     * @param       B Tensor to be subtracted against this.
     * @param       printTime bool for the method to print statistics about time. Default false.
     *              method overhead time.
     *
     * @see         cudaSubtract(Tensor B)
     *
     * @return      None.
     */
    void subtract(Propulsion::Tensor<type> &B, bool printTime = false)
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

    /**
     * \brief           Subtracts this to Tensor B if they're the same dimensions. Throws
     *              otherwise a TensorException.
     *
     * \details         Subtracts this to Tensor B if the dimensions match. Uses CUDA streams
     *              to speed up operations roughly to 3x(for my setup) from the other
     *              method (subtract) when using CUDA as well. Essentially this method determines
     *              whether or not it can subtract B to itself. If it can, it then determines how
     *              many "passes" are needed to achieve subtracting both tensors to one another.
     *              Then using that, we can theoretically use the GPU for any size Tensor as
     *              long as the both Tensors ROWS and COL Dimensions can be fit in the GPU memory
     *              at any given time.
     *
     * \example     Tensor I(1000, 3, 224, 224);                                    <br>
     *              Tensor R(1000, 3, 224, 224);                                    <br>
     *              I.cudaSubtract(R);       // I is modified, R is left alone.
     *
     * @throws      TensorException If the Tensor being added with does not match
     *              the dimension sizes of this.
     *
     * @param       B Tensor to subtract to this.
     * @param       printTime Bool to print stats about the overall operation overhead. Default false.
     * @param       printStats Print More Memory related information.
     *
     * @see         subtract(Tensor B)
     */
    void cudaSubtract(Propulsion::Tensor<type> &B, bool printTime = false, bool printStats = false)
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

    /**
     * \brief           Performs a element wise product on this and the given
     *              Tensor. Stores in this Tensor.
     *
     * \details         Multiplies two tensors element wise. Tensors both must match
     *              across all dimensions for this method to be performed. The resultant
     *              is stored in this. This method uses Matrix.schurProduct to perform
     *              across all the Matrix Elements in the deque.
     *
     * \example     Tensor<int> T(3, 4, 4);                                         <br>
     *              Tensor<int> S(3, 4, 4);                                         <br>
     *              ..... various code  .....                                       <br>
     *              T.schurProduct(S);      // good                                 <br>
     *              ..... Other example .....                                       <br>
     *              Tensor<int> BAD(2, 5, 5);                                       <br>
     *              Tensor<int> X(1, 4, 5);                                         <br>
     *              BAD.schurProduct(X);    // Throws TensorException
     *
     * \throws      TensorException If the Tensor being added with does not match
     *              the dimension sizes of this.
     *
     * @param       B Tensor that is to be schurProduct with this.
     * @param       printTimes Bool that prints overhead stats on the call. Default false.
     */
    void schurProduct(Tensor<type> &B, bool printTimes)
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

    /**
     * \brief           schurProduct this to Tensor B if they're the same dimensions. Throws
     *              otherwise a TensorException.
     *
     * \details         SchurProduct this to Tensor B if the dimensions match. Uses CUDA streams
     *              to speed up operations roughly to 3x(for my setup) from the other
     *              method (schurProduct) when using CUDA as well. Essentially this method determines
     *              whether or not it can schurProduct B to itself. If it can, it then determines how
     *              many "passes" are needed to achieve schurProduct both tensors to one another.
     *              Then using that, we can theoretically use the GPU for any size Tensor as
     *              long as the both Tensors ROWS and COL Dimensions can be fit in the GPU memory
     *              at any given time.
     *
     * \example     Tensor I(1000, 3, 224, 224);                                    <br>
     *              Tensor R(1000, 3, 224, 224);                                    <br>
     *              I.cudaSchurProduct(R);       // I is modified, R is left alone.
     *
     * @throws      TensorException If the Tensor being schurProduct with does not match
     *              the dimension sizes of this.
     *
     * @param       B Tensor to schurProduct to this.
     * @param       printTime Bool to print stats about the overall operation overhead. Default false.
     * @param       printStats Print More Memory related information.
     *
     * @see         schurProduct(Tensor B)
     */
    void cudaSchurProduct(Propulsion::Tensor<type> &B, bool printTime = false, bool printStats = false)
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

    /**
     * \brief           Multiplies the Tensor by the scalar provided as an argument.
     *
     * \details         Multiplies the Tensor by the scalar provided as an argument
     *              to the entire Tensor. Using CUDA streams to async transfer and
     *              launch kernels.
     *
     * \example     Tensor<double> D(1, 3, 3);                                  <br>
     *              D.scalarProduct(10);                                        <br>
     *              | 1 2 3 |       | 10 20 30 |                                <br>
     *              | 4 5 6 |   ->  | 40 50 60 |                                <br>
     *              | 7 8 9 |       | 70 80 90 |
     *
     * @throws      None.
     *
     * @param       scalar Type scalar to multiply the entire Tensor elements.
     * @param       printTime Bool to print overhead stats about the call.
     * @param       printStats Bool to print CUDA memory stat info for the streams.
     *
     * @return      None.
     */
    void scalarProduct(type scalar, bool printTime = false, bool printStats = false) {
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

    /**
     * \brief           Perform Tensor add with *this and Tensor rhs.
     *
     * \details         Adds *this and Param rhs using cudaAdd to perform
     *              the additions. Throws a TensorException when the dimensions
     *              do not match.
     *
     * \example     Propulsion::Tensor<int> rA = Propulsion::Tensor<int>(1,3,3);<br>
     *              Propulsion::Tensor<int> rB = Propulsion::Tensor<int>(1,3,3);<br>
     *              rA.populateWithRandomRealDistribution(-100000, 100000);     <br>
     *              rB.populateWithRandomRealDistribution(-100000, 100000);     <br>
     *                                                                          <br>
     *              Propulsion::Tensor<int> rC = rA + rB;
     *
     * @throw       TensorException If the dimensions do not match for Tensor addition.
     *
     * @param       rhs Tensor that is being added with *this.
     *
     * @return
     */
    Tensor<type> operator+(Tensor<type> &rhs)
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

    /**
     * \brief           Perform Tensor subtract with *this and Tensor rhs.
     *
     * \details         Subtracts *this and Param rhs using cudaSubtract to perform
     *              the difference. Throws a TensorException when the dimensions
     *              do not match.
     *
     * \example     Propulsion::Tensor<int> rA = Propulsion::Tensor<int>(1,3,3);<br>
     *              Propulsion::Tensor<int> rB = Propulsion::Tensor<int>(1,3,3);<br>
     *              rA.populateWithRandomRealDistribution(-100000, 100000);     <br>
     *              rB.populateWithRandomRealDistribution(-100000, 100000);     <br>
     *                                                                          <br>
     *              Propulsion::Tensor<int> rC = rA - rB;
     *
     * @throw       TensorException If the dimensions do not match for Tensor addition.
     *
     * @param       rhs Tensor that is being added with *this.
     *
     * @return
     */
    Tensor<type> operator-(Tensor<type> &rhs)
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

    /**
     * \brief           Checks if the tensors are equal to one
     *              another element wise.
     *
     * \details         Checks if the tensors are equal to one
     *              another element wise. Uses Matrix->equalTo
     *              method to check with the rhs.tensor[i] elem.
     *
     * @param       rhs Tensor to compare this with.
     *
     * @return      Bool True if the tensors are equal in dims and
     *              elements.
     */
    bool operator==(const Tensor<type> &rhs)
    {
        if(this->tensor.size() == rhs.tensor.size() && this->dims == rhs.dims) {
            for (unsigned long long i = 0; i < this->tensor.size(); i++) {
                if(!(this->tensor[i]->equalTo(*rhs.tensor[i])))
                    return false;
            }
        }
        else
        {
            return false;
        }
        return true;
    }

    /**
     * \brief           Populate Tensor with Random Real Distribution.
     *
     * \details         Populate Tensor with Random Real Distribution using
     *              Matrix method. Chain along the lRange and rRange values
     *              into the Matrices.
     *
     * \example     Tensor<double> T(2, 2, 2, 2);                               <br>
     *              T.populateWithRandomRealDistribution(-10.0, 10.0);
     *
     * \throws      None.
     *
     * @param       lRange The left exclusive bound of a distribution.
     * @param       rRange The right exclusive bound of a distribution.
     *
     * @return      Nothing.
     */
    void populateWithRandomRealDistribution(type lRange, type rRange)
    {
        for(unsigned i = 0; i < this->tensor.size(); i++) {
            Propulsion::Matrix<type>::randomRealDistribution(*this->tensor[i], lRange, rRange);
        }
    }



    /**
     * \brief           Returns the total amount of dimensions from the dims deque.
     *
     * \details         Returns the size() from this->dims to give the caller the
     *              total amount of dimensions if needed.
     *
     * \example     Tensor<int> T(1, 2, 3, 4, 5, 6);                            <br>
     *              std::cout << T.getTotalDims() << std::endl;                 <br>
     *
     *              *** outputs ***                                             <br>
     *              6                                                           <br>
     *              *** out end ***                                             <br>
     *
     * @return      size_t from this->dims.size() method.
     */
    size_t getTotalDims()
    {
        return this->dims.size();
    }

    /**
     * \brief           Returns the dims deque to the caller that way the dimensions
     *              are accessible outside the class if need be.
     *
     * \details         Returns the dims deque to the caller that way the dimensions
     *              are accessible outside the class. Such uses would be accessing
     *              the contents of the tensor and needing to know the sizes.
     *
     * \example     Tensor<int> T(2, 4, 4, 4);                                  <br>
     *              std::cout << T.getDims()[0] << std::endl;                   <br>
     *              *** outputs ***                                             <br>
     *              2                                                           <br>
     *              *** end out ***
     *
     * @return      const deque<int> reference which allows the user to access
     *              various methods if need be.
     */
    const std::deque<unsigned long long>& getDims()
    {
        return this->dims;
    }


    unsigned long long getTotalSize()
    {
        unsigned long long tmp = 1;
        for(unsigned long long i = 0; i < this->dims.size(); i++)
        {
            tmp *= this->dims[i];
        }
        return tmp;
    }

    /**
     * \brief           Prints the contents of the Tensor in order of 3rd
     *              dimension up to N dimensions.
     *
     * \details         Prints the contents of the Tensor in order of 3rd
     *              dimension up to the Nth Dimension. Uses Matrix.print(oStream)
     *              to pass along the std::ostream object along wih it.
     *
     * \example     Tensor<int> T(2, 2, 3, 3);                                  <br>
     *              T(0, 0, 0, 0) = 12;                                         <br>
     *              T(1, 1, 0, 0) = 13;                                         <br>
     *              T.print();                                                  <br>
     *
     *              *** outputs ***                                             <br>
     *              ( 0, 0, :, :)                                               <br>
     *              |  12   0   0 |                                             <br>
     *              |   0   0   0 |                                             <br>
     *              |   0   0   0 |                                             <br>
     *              ( 0, 1, :, :)                                               <br>
     *              .                                                           <br>
     *              .    .                                                      <br>
     *              .    .    .                                                 <br>
     *              ( 1, 1, :, :)                                               <br>
     *              |  0  0  0 |                                                <br>
     *              |  0  0  0 |                                                <br>
     *              |  0  0  0 |                                                <br>
     *              *** end out ***
     *
     * @param       oStream std::ostream object to handle the output
     *              of this method.
     */
    void print(std::ostream& oStream = std::cout)
    {
        if(this->tensor.size() == 1)
        {
            oStream << "( ";
            for(unsigned i = 0; i < this->dims.size() - 2; i++)
            {
                oStream << std::to_string(this->dims[i]) << ", ";
            }
            oStream << ":, :)" << std::endl;

            this->tensor[0]->print(oStream);
        }

        else {
            std::deque<unsigned long long> currIndexCount(this->dims.size() - 2, 0);
            unsigned long long cIdx = 0;

            // Case its a K x ... x M x N Tensor.
            for (unsigned long long i = 0; i < tensor.size(); i++)
            {
                /*
                 * Print the element number for this Matrix, using currIndexCount which is
                 * derived from the this->dims size-2.
                 */
                oStream << "( ";
                for(unsigned j = 0; j < currIndexCount.size() - 1; j++)
                {
                    oStream << currIndexCount[j] << ", ";
                }
                oStream << currIndexCount.back() << ", :, :)" << std::endl;

                //
                this->tensor[i]->print(oStream);



                // Update the cIdx, this way, next Matrix print we have the correct
                // information to display it.
                cIdx++;

                // Check if this updated value is now max size for 3rd dim. If so, create a
                // a "carry" chain for the next element(s).
                if(cIdx  == this->dims[this->dims.size()-3])
                {
                    // Reset Current Index on 3rd Dim to zero. That way Modulus works for the
                    // max size pf that dimension as intended.
                    cIdx = 0;
                    unsigned long long carry = 1;

                    // Carry chain.
                    for(unsigned long long j = currIndexCount.size(); j-- > 0;)
                    {
                        // If this index plus the carry has reached its max val, set to zero, and carry on.
                        if(currIndexCount[j] + carry == this->dims[j])
                        {
                            // Set to zero, update next index.
                            currIndexCount[j] = 0;
                        }
                        else
                        {
                            // Increment the current index.
                            currIndexCount[j] += carry;
                            break;
                        }


                        // Endless loop if not done since unsigned values
                    }
                }
                else
                {
                    currIndexCount[this->dims.size()-3] = cIdx;
                }
            }
        }
    }

    /*
     * Helper function, prints dims
     */
    void printDims(std::ostream& oStream = std::cout)
    {
        oStream << "( ";
        for(unsigned i = 0; i < dims.size(); i++)
        {
            if(i != dims.size() - 1)
                oStream << dims[i] << ", ";
            else
                oStream << dims[i];
        }
        oStream << ")" << std::endl;
    }
};