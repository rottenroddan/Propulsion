//
// Created by steve on 3/9/2021.
//
#pragma once
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
    std::deque<std::shared_ptr<Propulsion::Matrix<type>>> tensor;
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
     * \brief       Constructor method for creating a Tensor Object. Args is template packed
     *
     * \details         Allow the user to construct an any dimension tensor using an already
     *              existing array. The Dimensions must be specified via as a parameter along
     *              with the array pointer. That way the tensor is set up to suit that array
     *              best.
     *
     * \example     int arr[] = { 1, 0, 1,
     *                            0, 0, 1}
     *              Propulsion::Tensor<int> T(arr, 1, 2, 3); // Creates a 1x2x3 Tensor
     *
     * @throws      Propulsion::Tensor<type>::TensorException if zero is passed as an argument.
     *
     * @param       arr An array pointer to the array to construct a tensor around.
     * @param      ...args A template packed argument for generating a tensor
     *              of the dims given. Expected values are to be a type unsigned long long.
     *
     */
    template<typename... Ts, typename = AllUnsigned<Ts...>>
    Tensor(type* arr, Ts const&... args)
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
            this->tensor.push_back( std::unique_ptr<Propulsion::Matrix<type>> ( new Propulsion::Matrix<type>( arr, 1, this->dims[0])));
        }
        else if(this->dims.size() == 2)
        {
            this->tensor.push_back( std::unique_ptr<Propulsion::Matrix<type>> ( new Propulsion::Matrix<type>( arr ,this->dims[0], this->dims[1])));
        }
            // Else its a Kx...MxN tensor.
        else
        {
            // Get the last two dimensions indexes.
            unsigned long long rowsIdx = this->dims.size() - 2;
            unsigned long long colsIdx = this->dims.size() - 1;
            unsigned long long rowSize = this->dims[rowsIdx];
            unsigned long long colSize = this->dims[colsIdx];

            std::cout << rowSize << " : " << colSize << std::endl;


            // Calc. and store the total amount of matrices that must be created.
            unsigned long long totalM = 1;
            for(unsigned long long i = 0; i < this->dims.size() - 2; i++)
            {
                totalM *= this->dims[i];
            }

            // Now we populate tensor deque with matrices.
            for(unsigned long long i = 0; i < totalM; i++)
            {
                this->tensor.push_back( std::unique_ptr<Propulsion::Matrix<type>> ( new Propulsion::Matrix<type>( (arr + i*rowSize*colSize) , rowSize, colSize)));
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
        // Check if the tensor size are both one. If so, then return true.
        if (this->getTotalMatrices() == 1 && second.getTotalMatrices() == 1)
            return true;
        // Check if the total dims are the same.
        else if(this->getTotalDims() != second.getTotalDims())
            return false;
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

    void dotProduct(Propulsion::Tensor<type> &B, bool printTime = false);


    void cudaDotProduct(Tensor<type> &B, bool printTime = false, bool printStats = false);

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
    void add(Propulsion::Tensor<type> &B, bool printTime = false);

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
    void cudaAdd(Propulsion::Tensor<type> &B, bool printTime = false, bool printStats = false);


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
    void subtract(Propulsion::Tensor<type> &B, bool printTime = false);


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
    void cudaSubtract(Propulsion::Tensor<type> &B, bool printTime = false, bool printStats = false);

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
    void schurProduct(Tensor<type> &B, bool printTimes);

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
    void cudaSchurProduct(Propulsion::Tensor<type> &B, bool printTime = false, bool printStats = false);

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
    void scalarProduct(type scalar, bool printTime = false, bool printStats = false);

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
    Tensor<type> operator+(Tensor<type> &rhs);

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
    Tensor<type> operator-(Tensor<type> &rhs);

    /**
     * \brief           Perform Tensor dot product with *this and Tensor rhs.
     *
     * \details         Dot product *this and Param rhs using cudaDotProduct to perform
     *              the product. Throws a TensorException when the dimensions
     *              do not match for a valid matrix multiply
     *
     * \example     Propulsion::Tensor<int> rA = Propulsion::Tensor<int>(1,3,5);<br>
     *              Propulsion::Tensor<int> rB = Propulsion::Tensor<int>(1,5,3);<br>
     *              rA.populateWithRandomRealDistribution(-100000, 100000);     <br>
     *              rB.populateWithRandomRealDistribution(-100000, 100000);     <br>
     *                                                                          <br>
     *              Propulsion::Tensor<int> rC = rA * rB;
     *
     * @throw       TensorException If the dimensions do not match for Tensor dot product.
     *
     * @param       rhs Tensor that is being dot product with *this.
     *
     * @return
     */
    Tensor<type> operator*(Tensor<type> &rhs);

    /**
     *
     */
     Tensor<type>& operator=(const Tensor<type> &rhs);

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
     * \brief           Treats the parameter Tensor as a row vector and
     *              adds to this column wise.
     *
     * \details         Adds the contents of row vector tensor to each row
     *              in this. The tensor being used a row vector must only
     *              have one matrix object as is being treated as only one
     *              vector.
     *
     * \example     Tensor<double> input(input_Arr, 1, 5, 3);               <br>
     *              Tensor<double> bias(bias_arr, 1, 3);                    <br>
     *              input.addRowVector(bias);   // Contents of bias added to each row of input.     <br>
     *
     * \throws      TensorException if the the Tensor Row Vector Parameter has more than one dimension above 2 Dims.
     * \throws      TensorException if the row dimension is not a value of 1, or if the column dimension does not match
     *              that of this tensors column dim.
     *
     * @param       rowVector Tensor object being treated as a row vector.
     */
    void addRowVector(Tensor<type> &rowVector);

    /**
     * \brief           Treats the parameter Matrix as a row vector and
     *              adds to this column wise across all Matrices.
     *
     * \details         Adds the contents of row vector tensor to each row
     *              in this. The tensor being used a row vector must only
     *              have one matrix object as is being treated as only one
     *              vector.
     *
     * \example     Tensor<double> input(input_Arr, 1, 5, 3);               <br>
     *              Matrix<double> bias(bias_arr, 1, 3);                    <br>
     *              input.addRowVector(bias);   // Contents of bias added to each row of input.     <br>
     *
     * \throws      TensorException if the row dimension is not a value of 1, or if the column dimension does not match
     *              that of this tensors column dim.
     *
     * @param       rowVector Matrix object being treated as a row vector.
     */
    void addRowVector(Matrix<type> &rowVector);

    /**
     * \brief           Takes in two parameters. One is the Tensor object
     *              as a standard Tensor, where the second Parameter is a
     *              Tensor Object that is in shape of a row vector.
     *
     * \details         Excepts two parameters as Tensors. Treats the first
     *              parameter as a standard Tensor object, while the second
     *              parameter is treated as a row vector. Creates a new Tensor
     *              object that is returned to the caller.
     *
     * \example     Tensor<double> input(input_Arr, 1, 5, 3);               <br>
     *              Tensor<double> bias(bias_arr, 1, 3);                    <br>
     *              auto x = Tensor<double>::addRowVector(input, bias);     <br>
     *
     * \throws      TensorException if the the Tensor Row Vector Parameter has more than one dimension above 2 Dims.
     * \throws      TensorException if the row dimension is not a value of 1, or if the column dimension does not match
     *              that of this tensors column dim.
     *
     * @param       A Tensor that is of any size. The copied and modified value.
     * @param       rowVector Tensor object being treated as a row vector.
     *
     * @returns     A Tensor reference of the Tensor object rowVector added with.
     */
     static Tensor<type>& addRowVector(Tensor<type> &A, Tensor<type> &rowVector);

    /**
    * \brief           Treats the parameter Tensor as a col vector and
    *              adds to this row wise.
    *
    * \details         Adds the contents of col vector tensor to each col
    *              in this. The tensor being used a col vector must only
    *              have one matrix object as is being treated as only one
    *              vector.
    *
    * \example     Tensor<double> input(input_Arr, 1, 5, 3);               <br>
    *              Tensor<double> bias(bias_arr, 5, 1);                    <br>
    *              input.addColVector(bias);   // Contents of bias added to each col of input.     <br>
    *
    * \throws      TensorException if the the Tensor Col Vector Parameter has more than one dimension above 2 Dims.
    * \throws      TensorException if the Col dimension is not a value of 1, or if the row dimension does not match
    *              that of this tensors row dim.
    *
    * @param       colVector Tensor object being treated as a col vector.
    */
     void addColVector(Tensor<type> &colVector);

    /**
    * \brief           Treats the parameter Matrix as a col vector and
    *              adds to this row wise across all Matrices.
    *
    * \details         Adds the contents of col vector tensor to each row
    *              in this. The tensor being used a col vector must only
    *              have one matrix object as is being treated as only one
    *              vector.
    *
    * \example     Tensor<double> input(input_Arr, 1, 5, 3);               <br>
    *              Matrix<double> bias(bias_arr, 5, 1);                    <br>
    *              input.addColVector(bias);   // Contents of bias added to each row of input.     <br>
    *
    * \throws      TensorException if the row dimension is not a value of 1, or if the row dimension does not match
    *              that of this tensors row dim.
    *
    * @param       colVector Matrix object being treated as a col vector.
    */
    void addColVector(Matrix<type> &colVector);

    /**
     * \brief           Takes in two parameters. One is the Tensor object
     *              as a standard Tensor, where the second Parameter is a
     *              Tensor Object that is in shape of a col vector.
     *
     * \details         Excepts two parameters as Tensors. Treats the first
     *              parameter as a standard Tensor object, while the second
     *              parameter is treated as a col vector. Creates a new Tensor
     *              object that is returned to the caller.
     *
     * \example     Tensor<double> input(input_Arr, 1, 5, 3);               <br>
     *              Tensor<double> bias(bias_arr, 5, 1);                    <br>
     *              auto x = Tensor<double>::addColVector(input, bias);     <br>
     *
     * \throws      TensorException if the the Tensor Col Vector Parameter has more than one dimension above 2 Dims.
     * \throws      TensorException if the col dimension is not a value of 1, or if the row dimension does not match
     *              that of this tensors row dim.
     *
     * @param       A Tensor that is of any size. The copied and modified value.
     * @param       colVector Tensor object being treated as a col vector.
     *
     * @returns     A Tensor reference of the Tensor object colVector added with.
     */
    static Tensor<type>& addColVector(Tensor<type> &A, Tensor<type> &colVector);

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

    /**
     * \brief           Returns the total size of the Tensor deque.
     *
     * @return      unsigned value representing how many matrices exist to
     *              represent this tensor.
     */
    unsigned long long getTotalMatrices()
    {
        return this->tensor.size();
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

    void test() {}
};


#include "TensorArithmeticOperations.cu"
#include "TensorCopy.cu"
#include "TensorDotProduct.cu"
#include "TensorVectorOperations.cu"