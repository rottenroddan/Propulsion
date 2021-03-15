//
// Created by steve on 3/9/2021.
//
#include "Propulsion.cuh"
#include "TensorHelpers.cu"


/*
     * Template packing verification making sure that the value(s) provided as
     * the dims are in fact convertible to unsigned values.
     *
     * E.g. Tensor T(10, 4, 4)      // FINE
     *      Tensor T(10, 4, 4.1, 3) // FINE but note that 4.1 is converted to unsigned int.
     *      Tensor T(10, 3, 3, "a") // FAILS
     *
     * Uses:    Tensor
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
                std::string err = "Tensor Dimension Error: 0 is not a valid size for a dimension, maybe you mean 1?";
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
     * \brief       Returns a reference of the type from the given input as a
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
     * \brief       Returns a reference of the type from the given input as a
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

    bool checkAllDimensionsMatch(Propulsion::Tensor<type>& second)
    {
        // Check first if the total dimensions are the same.
        if(this->getTotalDims() != second.getTotalDims() )
        {
            return false;
        }
        // Last check if all dims match.
        else
        {
            for(unsigned i = 0; i < this->getTotalDims(); i++)
            {
                if(this->dims[i] != second.dims[i])
                    return false;
                else
                    continue;
            }
        }

        return true;
    }

    void add(Propulsion::Tensor<type>& B)
    {
        // Check if dimensions are the same that way we can add them together.
        if(this->checkAllDimensionsMatch(B))
        {

        }
        else
        {
            std::string err = "";
            err += "Tensor Size Mismatch in Add, this: " + getDimsExceptionString(this->dims) + " vs " + getDimsExceptionString(B.dims);
            throw Tensor<type>::TensorException(err.c_str(), __FILE__, __LINE__,
                                                "add", "Both Tensors must match all dimensions with one another, as its element wise addition.");
        }
    }

    /***
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
    size_t getTotalDims() noexcept
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