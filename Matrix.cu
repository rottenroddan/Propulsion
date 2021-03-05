//
// Created by steve on 7/31/2020.
//

#include "Propulsion.cuh"

template<typename type>
void Propulsion::Matrix<type>::generateMatrix(MatrixInitVal miv, MatrixInitType mit, type customVal, type *array)
{
    /*unsigned sz = this->cols * this->rows;
    M = new type[sz];*/

    switch(miv)
    {
        // Case if requested a zero matrix
        case MatrixInitVal::zero:
            this->M = generateZeroMatrixArray(this->rows * this->cols);
            break;

        // Case if requested a null matrix.
        // Same as zero matrix but NULL instead, in case of objects used.
        case MatrixInitVal::null:
            this->M = generateNullMatrixArray(this->rows * this->cols);
            break;

        // Other value than zero.
        default:
            type val;
            if(miv == MatrixInitVal::ones)
            {
                val = 1; // Enum is one
            }
            else if(miv == MatrixInitVal::twos)
            {
                val = 2; // Enum is two
            }
            else if(miv == MatrixInitVal::custom)
            {
                val = customVal;    // Enum is a custom val
            }


            switch(mit)
            {
                // Default matrix, filled with custom val or a provided array of type now.
                case MatrixInitType::def:
                    this->M = generateDefaultMatrixArray(this->rows * this->cols, val, array);
                    break;
                // Diagonal Matrix to generate.
                case MatrixInitType::diagonal:
                    this->M = generateDiagonalMatrixArray(this->rows * this->cols, val, array);
                    break;
            }
    }
}


template<typename type>
Propulsion::Matrix<type>::Matrix()
{
    this->rows = 1;
    this->cols = 1;
    this->totalSize = 1;
    this->M = std::make_unique<type[]>(1);
    this->M[0] = 0;
}

template<typename type>
Propulsion::Matrix<type>::Matrix(const Matrix<type>& copyM)
{
    this->rows = copyM.rows;
    this->cols = copyM.cols;
    this->totalSize = copyM.totalSize;
    this->M = std::make_unique<type[]>(copyM.totalSize);

    for(unsigned i = 0; i < this->totalSize; i++)
    {
        this->M[i] = copyM.M[i];
    }
}


template<typename type>
Propulsion::Matrix<type>::Matrix(unsigned rowAndColSize, MatrixInitVal miv, type customVal, MatrixInitType mit )
{
    if(rowAndColSize > 0)
    {
        this->rows = rowAndColSize;
        this->cols = rowAndColSize;
        this->totalSize = rowAndColSize*rowAndColSize;
        generateMatrix(miv, mit, customVal, nullptr);
    }
}

template<typename type>
Propulsion::Matrix<type>::Matrix(unsigned rows, unsigned cols, MatrixInitVal miv, type customVal, MatrixInitType mit)
{
    if(rows > 0 && cols > 0)
    {
        this->rows = rows;
        this->cols = cols;
        this->totalSize = rows*cols;
        generateMatrix(miv, mit, customVal, nullptr);
    }
}

template <typename type>
Propulsion::Matrix<type>::Matrix(type *array, unsigned rowAndColSize)
{
    if(rowAndColSize > 0)
    {
        this->rows = rowAndColSize;
        this->cols = rowAndColSize;
        this->totalSize = rowAndColSize*rowAndColSize;
        generateMatrix(MatrixInitVal::custom, MatrixInitType::def, NULL, array);
    }
}

template<typename type>
Propulsion::Matrix<type>::Matrix(type *array, unsigned rows, unsigned cols, MatrixInitVal miv, type customVal, MatrixInitType mit)
{
    if(rows > 0 && cols > 0)
    {
        this->rows = rows;
        this->cols = cols;
        this->totalSize = rows*cols;
        generateMatrix(miv, mit, customVal, array);
    }
}

template<typename type>
Propulsion::Matrix<type>::~Matrix()
{
    //M.reset();
    //this->M = nullptr;
}

template<typename type>
void Propulsion::Matrix<type>::print()
{
    unsigned spaceCount = 0;
    std::string digitStr;
    for(unsigned i = 0; i < rows*cols; i++)
    {
        std::string digitStr = std::to_string(M[i]);
        if(digitStr.length() > spaceCount)
        {
            spaceCount = digitStr.length();
        }
        digitStr.clear();
    }


    for(unsigned i = 0; i < rows; i++)
    {
        std::cout << "|";
        for(unsigned j = 0; j < cols; j++)
        {
            std::cout << std::setw(spaceCount + 2) << M[i*cols + j];
        }
        std::cout << " |" << std::endl;
    }
}


template<typename type>
void Propulsion::Matrix<type>::print(type *a, unsigned rows, unsigned cols)
{
    unsigned spaceCount = 0;
    std::string digitStr;
    for(unsigned i = 0; i < rows * cols; i++)
    {
        std::string digitStr = std::to_string(a[i]);
        if(digitStr.length() > spaceCount)
        {
            spaceCount = digitStr.length();
        }
    }


    for(unsigned i = 0; i < rows; i++)
    {
        std::cout << "|";
        for(unsigned j = 0; j < cols; j++)
        {
            std::cout << std::setw(spaceCount + 2) << a[i*rows + j];
        }
        std::cout << " |" << std::endl;
    }
}

template <typename type>
std::unique_ptr<type[]>  Propulsion::Matrix<type>::generateNullMatrixArray(unsigned rowAndColSize)
{
    // Get size of matrix.
    unsigned sz = rowAndColSize;
    auto r = std::make_unique<type[]>(sz);

    for(unsigned i = 0; i < sz; i++)
    {
        r[i] = NULL;
    }
    return r;
}

template <typename type>
std::unique_ptr<type[]>  Propulsion::Matrix<type>::generateZeroMatrixArray(unsigned rowAndColSize)
{
    // Get size of matrix.
    unsigned sz = rowAndColSize;
    auto r = std::make_unique<type[]>(sz);

    for(unsigned i = 0; i < sz; i++)
    {
        r[i] = (type)0;
    }
    return r;
}

template<typename type>
std::unique_ptr<type[]>  Propulsion::Matrix<type>::generateDefaultMatrixArray(unsigned rowAndColSize, type customVal, type *array)
{
    auto r = std::make_unique<type[]>(rowAndColSize);

    if(array == nullptr)
    {
        for(unsigned i = 0; i < rowAndColSize; i++)
        {
            r[i] =customVal;
        }
    }
    else
    {
        for(unsigned i = 0; i < rowAndColSize; i++)
        {
            r[i] = array[i];
        }
    }

    return r;
}

template <typename type>
std::unique_ptr<type[]>  Propulsion::Matrix<type>::generateDiagonalMatrixArray(unsigned rowAndColSize, type customVal, type *array)
{
    auto r = std::make_unique<type[]>(rowAndColSize);

    if(array == nullptr)
    {
        for(unsigned i = 0; i < this->rows; i++)
        {
            for(unsigned j = 0; j < this->cols; j++)
            {
                if(i == j)
                {
                    r[i*this->cols + j] = customVal;
                }
                else
                {

                    r[i*this->cols + j] = NULL;
                }
            }
        }
    }
    else
    {
        for(unsigned i = 0; i < this->rows; i++)
        {
            for(unsigned j = 0; j < this->cols; j++)
            {
                if(i == j)
                {
                    r[i*this->cols + j] = array[i];
                }
                else
                {
                    r[i*this->cols + j] = NULL;
                }
            }
        }
    }
    return r;
}

template <typename type>
void Propulsion::Matrix<type>::T()
{
    // Generate new array as we need a new one.
    auto temp = std::make_unique<type[]>(this->rows * this->cols);

    for(unsigned i = 0; i < this->rows; i++)
    {
        for(unsigned j = 0; j < this->cols; j++)
        {
            temp[j*this->rows + i] = this->M[i*this->cols + j];
        }
    }

    // Alter the rows and cols to their new representation.
    unsigned t = this->rows;
    this->rows = this->cols;
    this->cols = t;

    this->M = std::move(temp);
}

template <typename type>
void Propulsion::Matrix<type>::add(const Matrix<type>& b, bool printTime)
{
    if(b.rows == this->rows && b.cols == this->cols)
    {
        auto temp = std::make_unique<type[]>(this->rows * this->cols);
        // Use CUDA to speed up the process.
        if(this->totalSize >= MATRIX_CUDA_ADD_DIFF_ELEM_SIZE)
        {
            cudaAdd1DArraysWithStride(this->M.get(), b.M.get(), temp.get(), this->rows * this->cols, printTime);
            this->M = std::move(temp);
        }
        // Else just do it via HOST avx.
        else
        {
            Propulsion::hostAdd1DArraysAVX256(this->getArray(), b.M.get(), temp.get(), this->totalSize, printTime);
        }
    }
    else
    {
        std::string err = "Matrix Size Mismatch, ("+ std::to_string(this->rows) + ", " + std::to_string(this->cols)  + ") vs. (" + std::to_string(b.rows) +", " + std::to_string(b.cols) + ")";
        throw Propulsion::Matrix<type>::MatrixException(err.c_str(),
                                                        __FILE__, __LINE__, "add" , "Addition Requires all dimension sizes to be the same as the operation is element wise.");
    }
}

template<typename type>
Propulsion::Matrix<type> Propulsion::Matrix<type>::addRowVector(Matrix<type> &b)
{
    // Create return matrix. Initialized as 1x1 zero matrix.
    Propulsion::Matrix<type> ret;

    // Check if the rowVector can even be added to this. And check if b is a vector.
    if(this->cols == b.cols && b.rows == 1)
    {
        // Set the return matrix to the size of this.
        ret.rows = this->rows;
        ret.cols = this->cols;
        ret.totalSize = this->totalSize;
        ret.M = std::make_unique<type[]>(this->totalSize);

        // Loop through every element of this, add the jth element from be to every (i,j) of this.
        for(unsigned i = 0; i < this->rows; i++)
        {
            for(unsigned j = 0; j < this->cols; j++)
            {
                ret(i,j) = this->at(i,j) + b(j);
            }
        }
    }
    else
    {
        // Error for add row vector
        std::string err = "Matrix Size Mismatch, ("+ std::to_string(this->rows) + ", " + std::to_string(this->cols)  +
                          ") vs. (" + std::to_string(b.rows) +", " + std::to_string(b.cols) + ")" + ". Expected second Matrix to be ( 1, " + std::to_string(b.cols) + ")";
        throw Propulsion::Matrix<type>::MatrixException(err.c_str(),__FILE__, __LINE__, "addRowVector" ,
                                                        "addRowVector Requires that the argument be a row vector such that it is 1xn in DIM.");
    }
    return ret;
}

template<typename type>
Propulsion::Matrix<type> Propulsion::Matrix<type>::addRowVector(Matrix<type> &&b)
{
    // Create return matrix. Initialized as 1x1 zero matrix.
    Propulsion::Matrix<type> ret;

    // Check if the rowVector can even be added to this. And check if b is a vector.
    if(this->cols == b.cols && b.rows == 1)
    {
        // Set the return matrix to the size of this.
        ret.rows = this->rows;
        ret.cols = this->cols;
        ret.totalSize = this->totalSize;
        ret.M = std::make_unique<type[]>(this->totalSize);

        // Loop through every element of this, add the jth element from be to every (i,j) of this.
        for(unsigned i = 0; i < this->rows; i++)
        {
            for(unsigned j = 0; j < this->cols; j++)
            {
                ret(i,j) = this->at(i,j) + b(j);
            }
        }
    }
    else
    {
        // Error for add row vector
        std::string err = "Matrix Size Mismatch, ("+ std::to_string(this->rows) + ", " + std::to_string(this->cols)  +
                          ") vs. (" + std::to_string(b.rows) +", " + std::to_string(b.cols) + ")" + ". Expected second Matrix to be ( 1, " + std::to_string(b.cols) + ")";
        throw Propulsion::Matrix<type>::MatrixException(err.c_str(),__FILE__, __LINE__, "addRowVector" ,
                                                        "addRowVector Requires that the argument be a row vector such that it is 1xn in DIM.");
    }
    return ret;
}

template<typename type>
Propulsion::Matrix<type> Propulsion::Matrix<type>::addColVector(Matrix<type> &b)
{
    // Create return matrix. Initialized as 1x1 zero matrix.
    Propulsion::Matrix<type> ret;

    // Check if the colVector can even be added to the matrix.
    if(this->rows == b.rows && b.cols == 1)
    {
        // Set the return matrix to the size of this.
        ret.rows = this->rows;
        ret.cols = this->cols;
        ret.totalSize = this->totalSize;
        ret.M = std::make_unique<type[]>(this->totalSize);

        // Loop through every element of this, add the ith element from be to every (i,j) of this.
        for(unsigned i = 0; i < this->rows; i++)
        {
            for(unsigned j = 0; j < this->cols; j++)
            {
                ret(i,j) = this->at(i,j) + b(i);
            }
        }
    }
    else
    {
        // Error for add row vector
        std::string err = "Matrix Size Mismatch, ("+ std::to_string(this->rows) + ", " + std::to_string(this->cols)  +
                          ") vs. (" + std::to_string(b.rows) +", " + std::to_string(b.cols) + ")" + ". Expected second Matrix to be ( " + std::to_string(b.rows) + ", 1)";
        throw Propulsion::Matrix<type>::MatrixException(err.c_str(),__FILE__, __LINE__, "addColVector" ,
                                                        "addColVector Requires that the argument be a row vector such that it is nx1 in DIM.");
    }
    return ret;
}

template<typename type>
Propulsion::Matrix<type> Propulsion::Matrix<type>::addColVector(Matrix<type> &&b)
{
    // Create return matrix. Initialized as 1x1 zero matrix.
    Propulsion::Matrix<type> ret;

    // Check if the colVector can even be added to the matrix.
    if(this->rows == b.rows && b.cols == 1)
    {
        // Set the return matrix to the size of this.
        ret.rows = this->rows;
        ret.cols = this->cols;
        ret.totalSize = this->totalSize;
        ret.M = std::make_unique<type[]>(this->totalSize);

        // Loop through every element of this, add the ith element from be to every (i,j) of this.
        for(unsigned i = 0; i < this->rows; i++)
        {
            for(unsigned j = 0; j < this->cols; j++)
            {
                ret(i,j) = this->at(i,j) + b(i);
            }
        }
    }
    else
    {
        // Error for add row vector
        std::string err = "Matrix Size Mismatch, ("+ std::to_string(this->rows) + ", " + std::to_string(this->cols)  +
                          ") vs. (" + std::to_string(b.rows) +", " + std::to_string(b.cols) + ")" + ". Expected second Matrix to be ( " + std::to_string(b.rows) + ", 1)";
        throw Propulsion::Matrix<type>::MatrixException(err.c_str(),__FILE__, __LINE__, "addColVector" ,
                                                        "addColVector Requires that the argument be a row vector such that it is nx1 in DIM.");
    }
    return ret;
}

template <typename type>
void Propulsion::Matrix<type>::subtract(const Matrix<type> &b)
{
    if(b.rows == this->rows && b.cols == this->cols)
    {
        // Use CUDA to speed up the process.
        if(this->totalSize >= MATRIX_CUDA_ADD_DIFF_ELEM_SIZE)
        {
            std::cout << "Here!" << std::endl;
            auto temp = std::make_unique<type[]>(this->rows * this->cols);
            cudaSubtract1DArraysWithStride(this->M.get(), b.M.get(), temp.get(), this->rows * this->cols);
            this->M = std::move(temp);
        }
        for(unsigned i = 0; i < this->totalSize; i++)
        {
            this->M[i] = this->M[i] - b.M[i];
        }
    }
    else
    {
        std::string err = "Matrix Size Mismatch, ("+ std::to_string(this->rows) + ", " + std::to_string(this->cols)  + ") vs. (" + std::to_string(b.rows) +", " + std::to_string(b.cols) + ")";
        throw Propulsion::Matrix<type>::MatrixException(err.c_str(),
                                                        __FILE__, __LINE__, "subtract" , "subtract Requires all dimension sizes to be the same as the operation is element wise.");
    }
}

template<typename type>
void Propulsion::Matrix<type>::cudaDotProduct(const Matrix<type> &b, bool printTime)
{
    if(this->cols == b.rows) {
        // Create Matrix with A row size and b col size as nxm * mxk = nxk
        Propulsion::Matrix<type> temp(this->getRowSize(), b.cols, MatrixInitVal::zero);

        // Using CUDA from Propulsion to handle.
        Propulsion::cudaDotProduct(this->getArray(), b.M.get(), temp.getArray(), this->getRowSize(), this->cols,
                                  b.cols, printTime);

        // Move the pointer from Temp to M now.
        this->M = std::move(temp.M);
        this->rows = temp.rows;
        this->cols = temp.cols;
        this->totalSize = temp.totalSize;
    }
    else
    {
        std::string err = "Matrix Size Mismatch, ("+ std::to_string(this->rows) + ", " + std::to_string(this->cols)  + ") vs. (" + std::to_string(b.rows) +", " + std::to_string(b.cols) + ")";
        throw Propulsion::Matrix<type>::MatrixException(err.c_str(),
                                                        __FILE__, __LINE__, "cudaMultiplyMatrices" , "cudaMultiplyMatrices Requires the second dimension of matrix A match first dimension of matrix B");
    }
}

template <typename type>
void Propulsion::Matrix<type>::dot(const Matrix<type> &b, bool printTime)
{
    if(this->cols == b.rows)
    {
        // Get size of new Matrix
        unsigned newSize = this->rows * b.cols;
        auto multiplyArray = std::make_unique<type[]>(newSize);

        // Case its a 1x1.
        if(this->totalSize == 1 && b.totalSize == 1)
        {
            multiplyArray[0] = this->M[0] * b.M[0];
        }
        else {
            /*
            for (unsigned r = 0; r < this->rows; r++) {
            for (unsigned c = 0; c < b.cols; c++) {
            for (unsigned i = 0; i < this->cols; i++) {
            sum += at(r, i) * b.M[i * b.cols + c];
            }
            multiplyArray[n] = sum;
            sum = 0;
            n++;
            }
            }*/

            Propulsion::hostDotProduct(this->getArray(), b.M.get(), multiplyArray.get(), this->rows, this->cols, b.cols, printTime);
        }

        this->cols = b.cols;        // If you know, you know. AB: A * B is 2x3 - 3x3: New Matrix is 2(this rows)x3(b cols).
        this->totalSize = newSize;  // set the totalSize of this to the new size of the product matrix.
        this->M = std::move(multiplyArray);
    }
    else
    {
        std::string err = "Matrix Size Mismatch, ("+ std::to_string(this->rows) + ", " + std::to_string(this->cols)  + ") vs. (" + std::to_string(b.rows) +", " + std::to_string(b.cols) + ")";
        throw Propulsion::Matrix<type>::MatrixException(err.c_str(),
                                                        __FILE__, __LINE__, "dot" , "dot Requires the second dimension of matrix A match first dimension of matrix B");
    }
}

template<typename type>
void Propulsion::Matrix<type>::schurProduct(const Matrix<type> &b, bool printTime)
{
    if(this->rows == b.rows && this->cols == b.cols)
    {
        auto schurArray = std::make_unique<type[]>(this->totalSize);

        if (this->totalSize >= MATRIX_CUDA_ADD_DIFF_ELEM_SIZE)
        {
            cudaSchurProduct(this->getArray(), b.M.get(), schurArray.get(), this->totalSize, printTime);
        }
        else
        {
            hostSchurProduct(this->getArray(), b.M.get(), schurArray.get(), this->totalSize, printTime);
        }

        this->M = std::move(schurArray);
    }
    else
    {
        std::string err = "Matrix Size Mismatch, ("+ std::to_string(this->rows) + ", " + std::to_string(this->cols)  + ") vs. (" + std::to_string(b.rows) +", " + std::to_string(b.cols) + ")";
        throw Propulsion::Matrix<type>::MatrixException(err.c_str(),
                __FILE__, __LINE__, "schurProduct" , "Schurs Product Requires all dimension sizes to be the same, as the proudct is element wise.");
    }
}

template<typename type>
void Propulsion::Matrix<type>::multiply(type scalar)
{
    for(unsigned i = 0; i < this->totalSize; i++)
    {
        this->M[i] *= scalar;
    }
}

template<typename type>
void Propulsion::Matrix<type>::strassenMultiplication(const Matrix<type> &rb)
{
    // check if we can dot first.
    if(this->cols == rb.rows)
    {
        // get the log2(rows/cols) as we need to make a nxn matrix that is divisible into 4 partitions.
        double firstRowPowerOfTwo  = std::log2(this->rows);
        double firstColPowerOfTwo  = std::log2(this->cols); // This value is the same as b.cols, so we only need to check the Column.
        double secondColPowerOfTwo = std::log2(rb.cols);

        double largestNPower = 0.0;

        // then find the largest value of the 3, store into largestNPower, so we can pad later with zeroes to the ceiling
        // of largestNPower.
        if(firstRowPowerOfTwo >= firstColPowerOfTwo)
        {
            largestNPower = firstRowPowerOfTwo;
        }
        else
        {
            largestNPower = firstColPowerOfTwo;
        }
        if(largestNPower <= secondColPowerOfTwo)
        {
            largestNPower = secondColPowerOfTwo;
        }

        // ceiling to the next 2^largestNPower.
        unsigned squareR = (unsigned)(std::pow(2.0, std::ceil(largestNPower)));

        // the rows and cols of the matrices are now altered if they need to be.
        Propulsion::Matrix<type> A = *this;
        Propulsion::Matrix<type> B = rb;

        // pad the matrices with zeros for Strassen Multiplication.
        A.pad(squareR, squareR);
        B.pad(squareR, squareR);

        // Divide A and B into 8 partitions starting with 4 from A.
        auto a = A.getRangeMatrix(0, A.rows / 2 - 1, 0, A.cols / 2 - 1);
        auto b = A.getRangeMatrix(0, A.rows / 2 - 1, A.cols / 2, A.cols - 1);
        auto c = A.getRangeMatrix(A.rows / 2, A.rows - 1, 0, A.cols / 2 - 1);
        auto d = A.getRangeMatrix(A.rows / 2, A.rows - 1, A.cols / 2, A.cols - 1);
        auto e = B.getRangeMatrix(0, B.rows / 2 - 1, 0, B.cols / 2 - 1);
        auto f = B.getRangeMatrix(0, B.rows / 2 - 1, B.cols / 2, B.cols - 1);
        auto g = B.getRangeMatrix(B.rows / 2, B.rows - 1, 0, B.cols / 2 - 1);
        auto h = B.getRangeMatrix(B.rows / 2, B.rows - 1, B.cols / 2, B.cols - 1);

        // Create 7 async threads for the Strassen recursion.
        auto p1 = std::async(recursiveStrassen, a, f - h);
        auto p2 = std::async(recursiveStrassen, a + b, h);
        auto p3 = std::async(recursiveStrassen, c + d, e);
        auto p4 = std::async(recursiveStrassen, d, g - e);
        auto p5 = std::async(recursiveStrassen, a + d, e + h);
        auto p6 = std::async(recursiveStrassen, b - d, g + h);
        auto p7 = std::async(recursiveStrassen, a - c, e + f);

        // Get Values from threads for future use.
        auto pr1 = p1.get();
        auto pr2 = p2.get();
        auto pr3 = p3.get();
        auto pr4 = p4.get();
        auto pr5 = p5.get();
        auto pr6 = p6.get();
        auto pr7 = p7.get();

        auto c1 = pr5 + pr4 - pr2 + pr6;
        auto c2 = pr1 + pr2;
        auto c3 = pr3 + pr4;
        auto c4 = pr1 + pr5 - pr3 - pr7;

        auto C = c1.mergeRight(c2);
        auto CB = c3.mergeRight(c4);
        C = C.mergeBelow(CB);

        // trim the matrix to the original pxn*mxq = pxq
        for(unsigned i = 0; i < squareR - rb.cols; i++)
        {
            C = C.removeCol(b.cols);    // Remove Last col i times.
        }

        for(unsigned i = 0; i < squareR - this->rows; i++)
        {
            C = C.removeRow(this->rows);    // Remove Last row i times.
        }

        *this = C;
    }
    else
    {
        std::string err = "Matrix Size Mismatch, ("+ std::to_string(this->rows) + ", " + std::to_string(this->cols)  + ") vs. (" + std::to_string(rb.rows) +", " + std::to_string(rb.cols) + ")";
        throw Propulsion::Matrix<type>::MatrixException(err.c_str(),
                                                        __FILE__, __LINE__, "dot" , "dot Requires the second dimension of matrix A match first dimension of matrix B");
    }
}

template<typename type>
Propulsion::Matrix<type> Propulsion::Matrix<type>::recursiveStrassen(Propulsion::Matrix<type> A, Propulsion::Matrix<type> B) {
    // check the size of the leaf matrix. That way we can just use O(n^3)
    // once we get to a manageable size.
    if(A.totalSize <= HOST_STRASSEN_LEAF_SIZE)
    {
        return A * B;
    }

    // get range matrices for a,b,c,d,e,f,g and h for strassen multiplication. You know you know.
    auto a = A.getRangeMatrix(0, A.rows / 2 - 1, 0, A.cols / 2 - 1);
    auto b = A.getRangeMatrix(0, A.rows / 2 - 1, A.cols / 2, A.cols - 1);
    auto c = A.getRangeMatrix(A.rows / 2, A.rows - 1, 0, A.cols / 2 - 1);
    auto d = A.getRangeMatrix(A.rows / 2, A.rows - 1, A.cols / 2, A.cols - 1);
    auto e = B.getRangeMatrix(0, B.rows / 2 - 1, 0, B.cols / 2 - 1);
    auto f = B.getRangeMatrix(0, B.rows / 2 - 1, B.cols / 2, B.cols - 1);
    auto g = B.getRangeMatrix(B.rows / 2, B.rows - 1, 0, B.cols / 2 - 1);
    auto h = B.getRangeMatrix(B.rows / 2, B.rows - 1, B.cols / 2, B.cols - 1);

    auto p1 = recursiveStrassen(a, f - h);
    auto p2 = recursiveStrassen(a + b, h);
    auto p3 = recursiveStrassen(c + d, e);
    auto p4 = recursiveStrassen(d, g - e);
    auto p5 = recursiveStrassen(a + d, e + h);
    auto p6 = recursiveStrassen(b - d, g + h);
    auto p7 = recursiveStrassen(a - c, e + f);

    auto c1 = p5 + p4 - p2 + p6;
    auto c2 = p1 + p2;
    auto c3 = p3 + p4;
    auto c4 = p1 + p5 - p3 - p7;

    auto C = c1.mergeRight(c2);
    auto CB = c3.mergeRight(c4);
    C = C.mergeBelow(CB);


    return C;
}

template<typename type>
type* Propulsion::Matrix<type>::getArray()
{
    return this->M.get();
}

template<typename type>
unsigned Propulsion::Matrix<type>::getColSize()
{
    return cols;
}

template<typename type>
unsigned Propulsion::Matrix<type>::getRowSize()
{
    return rows;
}

template<typename type>
unsigned Propulsion::Matrix<type>::getTotalSize()
{
    return cols * rows;
}

template <typename type>
bool Propulsion::Matrix<type>::equalTo(const Matrix<type> &b)
{
    if( this->rows != b.rows || this->cols != b.cols)
    {
        return false;
    }
    else
    {
        for(unsigned i = 0; i < this->totalSize; i++)
        {
            if(this->M[i] != b.M[i])
            {
                return false;
            }
        }
        return true;
    }
}

template<typename type>
bool Propulsion::Matrix<type>::operator==(const Propulsion::Matrix<type> &rhs) {
    return equalTo(rhs);
}

template<typename type>
bool Propulsion::Matrix<type>::isUpperTriangular()
{
    // Not nxn.
    if(this->rows != this->cols){return false;}

    for(unsigned i = 0; i < this->rows; i++)
    {
        for(unsigned j = 0; j <= i; j++)
        {
            // On the diagonal.
            if(i == j)
            {
                // if diagonal is nonzero, move on.
                if(at(i,j) != 0)
                {
                    continue;
                }
                else
                {
                    return false;
                }
            }

            if(at(i,j) == 0)
            {
                continue;
            }
            else
            {
                return false;
            }
        }
    }
    return true;
}

template<typename type>
bool Propulsion::Matrix<type>::isLowerTriangular()
{
    // Not nxn.
    if(this->rows != this->cols){return false;}

    for(unsigned i = 0; i < this->rows; i++)
    {
        for(unsigned j = i; j < this->cols; j++)
        {
            // On the diagonal.
            if(i == j)
            {
                if(at(i,j) != 0)
                {
                    continue;
                }
                else
                {
                    return false;
                }
            }
            if(at(i,j) == 0)
            {
                continue;
            }
            else
            {
                return false;
            }
        }
    }
    return true;
}


template <typename type>
Propulsion::Matrix<type> Propulsion::Matrix<type>::getRowMatrix(unsigned int row)
{
    Propulsion::Matrix<type> ret;
    // Check if in range.
    if(row < this->rows) {
        ret.rows = 1;
        ret.cols = this->cols;
        ret.totalSize = this->cols;
        ret.M = std::make_unique<type[]>(ret.totalSize);

        for (unsigned i = 0; i < ret.cols; i++) {
            ret.M[i] = this->M[row * this->cols + i];
        }
    }

    return ret;
}

template <typename type>
Propulsion::Matrix<type> Propulsion::Matrix<type>::getColMatrix(unsigned int col)
{
    Propulsion::Matrix<type> ret;
    // Check if in range.
    if(col < this->cols) {
        ret.rows = this->rows;
        ret.cols = 1;
        ret.totalSize = this->rows;
        ret.M = std::make_unique<type[]>(ret.totalSize);

        for (unsigned i = 0; i < ret.rows; i++) {
            ret.M[i] = this->M[this->cols * i + col];
        }
    }

    return ret;
}

template <typename type>
Propulsion::Matrix<type> Propulsion::Matrix<type>::getRangeMatrix(unsigned rowStart, unsigned rowEnd, unsigned colStart, unsigned colEnd)
{
    Propulsion::Matrix<type> ret;
    // Check if in range.
    if(rowStart <= rowEnd && rowEnd < this->rows && colStart <= colEnd && colEnd < this->cols)
    {
        unsigned rElement = 0; // Iterator for the nth element in the return array. Incremented in second for loop.
        ret.rows = rowEnd - rowStart + 1;
        ret.cols = colEnd - colStart + 1;
        ret.totalSize = ret.rows*ret.cols;
        ret.M = std::make_unique<type[]>(ret.totalSize);


        for(unsigned i = rowStart; i <= rowEnd; i++)
        {
            for(unsigned j = colStart; j <= colEnd; j++)
            {
                ret.M[rElement] = at(i,j);
                rElement++;
            }
        }
    }

    return ret;
}

template<typename type>
Propulsion::Matrix<type> Propulsion::Matrix<type>::mergeRight( Matrix<type> &b)
{
    Propulsion::Matrix<type> ret;
    // Check whether or not they have the same rows.
    if(this->rows == b.rows)
    {
        ret.rows = this->rows;
        ret.cols = this->cols + b.cols;
        ret.totalSize = this->totalSize + b.totalSize;
        ret.M = std::make_unique<type[]>(ret.totalSize);

        for(unsigned i = 0; i < ret.rows; i++)
        {
            for(unsigned j = 0; j < ret.cols; j++)
            {
                if(j < this->cols)
                {
                    ret.at(i,j) = at(i,j);
                }
                else
                {
                    ret.at(i,j) = b.at(i, j - this->cols);
                }
            }
        }
        return ret;
    }

    return *this;
}

template<typename type>
Propulsion::Matrix<type> Propulsion::Matrix<type>::mergeBelow( Matrix<type> &b)
{
    Propulsion::Matrix<type> ret;
    // Check whether or not they have the same rows.
    if(this->cols == b.cols)
    {
        ret.rows = this->rows + b.rows;
        ret.cols = this->cols;
        ret.totalSize = ret.rows * ret.cols;
        ret.M = std::make_unique<type[]>(ret.totalSize);

        for(unsigned i = 0; i < ret.rows; i++)
        {
            for(unsigned j = 0; j < ret.cols; j++)
            {
                if(i < this->rows)
                {
                    ret.at(i,j) = at(i,j);
                }
                else
                {
                    ret.at(i,j) = b.at(i - this->rows, j);
                }
            }
        }
        return ret;
    }

    // return the object called from which is unaltered.
    return *this;
}

template <typename type>
type& Propulsion::Matrix<type>::at(unsigned i)
{
    if(i < rows * cols)
        return this->M[i];
    else {
        /*std::cout << "Accessing Matrice Outside of Bounds with i: " << i
                  << ", DIMS is [" << rows << "," << cols << "] = " << rows * cols << std::endl;*/
        throw std::out_of_range("Accessing Matrix Outside of Bounds with i: " + std::to_string(i) + ", DIMS is [" + std::to_string(rows)
                                + "," + std::to_string(cols) + "] = " + std::to_string(rows*cols) + "\n");
    }
}

template <typename type>
type& Propulsion::Matrix<type>::at(unsigned i, unsigned j)
{
    if(i < rows && j < cols)
    {
        return this->M[i*cols + j];
    }
    else
    {
        throw std::out_of_range("Accessing Matrix Outside of Bounds with i: " + std::to_string(i) + " & j: " + std::to_string(j) + ", DIMS is ["
                                + std::to_string(rows) + "," + std::to_string(cols) + "] = " + std::to_string(rows*cols) + "\n");
    }
}

template <typename type>
type& Propulsion::Matrix<type>::operator()(unsigned i)
{
    return at(i);
}

template <typename type>
type& Propulsion::Matrix<type>::operator()(unsigned i, unsigned j)
{
    return at(i,j);
}


template <typename type>
Propulsion::Matrix<type> Propulsion::Matrix<type>::operator+(Matrix<type> &rhs)
{
    Propulsion::Matrix<type> ret;
    if(rows == rhs.rows && cols == rhs.cols)
    {
        ret.rows = rows;
        ret.cols = cols;
        ret.totalSize = totalSize;
        ret.M = std::make_unique<type[]>(totalSize);

        hostAdd1DArraysAVX256(this->getArray(), rhs.getArray(), ret.getArray(),this->getTotalSize());
        /*
        for(unsigned i = 0; i < totalSize; i++)
        {
            ret.M[i] = rhs.M[i] + M[i];
        }*/
    }
    return ret;
}

template <typename type>
Propulsion::Matrix<type> Propulsion::Matrix<type>::operator-(Matrix<type> &rhs)
{
    Propulsion::Matrix<type> ret;
    if(rows == rhs.rows && cols == rhs.cols)
    {
        ret.rows = rows;
        ret.cols = cols;
        ret.totalSize = totalSize;
        ret.M = std::make_unique<type[]>(totalSize);

        hostSubtract1DArraysAVX256(this->getArray(), rhs.getArray(), ret.getArray(), this->getTotalSize());
        /*
        for(unsigned i = 0; i < totalSize; i++)
        {
            ret.M[i] = M[i] - rhs.M[i];
        }*/
    }
    return ret;
}

template <typename type>
Propulsion::Matrix<type> Propulsion::Matrix<type>::operator*(const Matrix<type> &rhs) {
    Propulsion::Matrix<type> ret = *this;    // Copy the contents of this to the return value.

    // Choose whether we use regular multiplication or strassen multiplication.
    if (ret.totalSize <= HOST_STRASSEN_LEAF_SIZE*HOST_STRASSEN_LEAF_SIZE)
    {
        ret.dot(rhs);      // Use already defined object method to dot by the right hand side matrix.
    }
    else
    {
        // Use Strassen to create seven threads to solve.
        ret.strassenMultiplication(rhs);
    }

    return ret;
}

template <typename type>
Propulsion::Matrix<type> Propulsion::Matrix<type>::operator*(type rhs)
{
    Propulsion::Matrix<type> ret = *this;    // Copy the contents of this to the return value.
    ret.multiply(rhs);      // Use already defined object method to dot by the right hand side scalar.
    return ret;
}


template<typename type>
Propulsion::Matrix<type>& Propulsion::Matrix<type>::operator=(const Matrix<type> &rhs)
{
    // If same return this
    if(this == &rhs) {return *this;}
    else {
        // else, make a deep copy.
        this->totalSize = rhs.totalSize;
        this->rows = rhs.rows;
        this->cols = rhs.cols;
        this->M = std::make_unique<type[]>(this->totalSize);

        for (unsigned i = 0; i < rhs.totalSize; i++) {
            this->M[i] = rhs.M[i];
        }
    }
    return *this;
}

/*
template<typename type>
Propulsion::Matrix<type>& Propulsion::Matrix<type>::operator=(Matrix<type> &r)
{
    if(this == &r) {return *this;}
    else {
        delete[] this->M;
        this->M = new type[r.totalSize];
        this->rows = r.rows;
        this->cols = r.cols;
        this->totalSize = r.totalSize;

        for (unsigned i = 0; i < r.totalSize; i++) {
            this->M[i] = r.M[i];
        }
    }
    return *this;
}*/

template <typename type>
void Propulsion::Matrix<type>::pad(unsigned rows, unsigned cols)
{
    if(this->rows <= rows && this->cols <= cols)
    {
        // the number of rows/cols to add. E.g 6-4=2 rows to add.
        unsigned rowsToAdd = rows - this->rows;
        unsigned colsToAdd = cols - this->cols;


        Propulsion::Matrix<type> bottomRows(rowsToAdd, this->cols);
        Propulsion::Matrix<type> rightCols(rows, colsToAdd);


        *this = mergeBelow(bottomRows);
        *this = mergeRight(rightCols);
    }
}

template<typename type>
void Propulsion::Matrix<type>::populateWithUniformDistribution(type lRange, type rRange)
{
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution((double)lRange, (double)rRange);

    for(unsigned i = 0; i < this->totalSize;i++)
    {
        this->M[i] = (type)distribution(generator);
    }
}

template<typename type>
Propulsion::Matrix<type> Propulsion::Matrix<type>::removeRow(unsigned int rowToRem) {
    Propulsion::Matrix<type> ret;
    unsigned rIter = 0;
    if(rowToRem < this->rows)
    {
        ret.rows = this->rows - 1;
        ret.cols = this->cols;
        ret.totalSize = ret.rows * ret.cols;
        ret.M = std::make_unique<type[]>(ret.totalSize);

        for(unsigned i = 0; i < ret.rows; i++)
        {
            if(i == rowToRem){rIter++;}
            for(unsigned j = 0; j < ret.cols; j++)
            {
                ret.at(i,j) = at(rIter,j);
            }

            rIter++;
        }
        return ret;
    }

    return *this;
}

template<typename type>
Propulsion::Matrix<type> Propulsion::Matrix<type>::removeCol(unsigned int colToRem) {
    Propulsion::Matrix<type> ret;
    unsigned cIter = 0;
    if(colToRem < this->cols)
    {
        ret.rows = this->rows;
        ret.cols = this->cols - 1;
        ret.totalSize = ret.rows * ret.cols;
        ret.M = std::make_unique<type[]>(ret.totalSize);

        for(unsigned i = 0; i < ret.rows; i++)
        {
            cIter = 0;
            for(unsigned j = 0; j < ret.cols; j++)
            {
                if(j == colToRem){cIter++;}
                ret.at(i,j) = at(i,cIter);
                cIter++;
            }
        }
        return ret;
    }

    return *this;
}



template<typename type>
Propulsion::Matrix<type> Propulsion::Matrix<type>::copy(Matrix<type> a)
{
    Propulsion::Matrix<type> b(a.getRowSize(), a.getColSize());

    if(a.totalSize > MATRIX_COPY_SIZE_DIFF) {
        Propulsion::cudaCopyArray(a.M.get(), b.M.get(), a.getTotalSize());
        return b;
    }
    else
    {
        b = a;
        return b;
    }
}


template<typename type>
void Propulsion::Matrix<type>::randomRealDistribution(Matrix<type> &A, type lVal, type rVal)
{
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(lVal,rVal);

    for(unsigned i = 0; i < A.getTotalSize(); i++)
    {
        A(i) = dist(e2);
    }
}

template<typename type>
void Propulsion::Matrix<type>::randomRealDistribution(std::shared_ptr<Matrix<type>> A, type lVal, type rVal)
{
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(lVal,rVal);

    for(unsigned i = 0; i < A->getTotalSize(); i++)
    {
        A->at(i) = dist(e2);
    }
}

template<typename type>
Propulsion::Matrix<type> Propulsion::Matrix<type>::sumRows(Matrix<type> &&A)
{
    // create return object, give it size of the rows from A, 1 for columns.
    Propulsion::Matrix<type> ret;
    ret.rows = A.rows;
    ret.cols = 1;
    ret.totalSize = A.rows;
    ret.M = std::make_unique<type[]>(ret.totalSize);

    for(unsigned i = 0; i < A.rows; i++)
    {
        // Sum starts from zero.
        type sum = (type)0;

        // Loop through all of A, adding the elements on the same row.
        for(unsigned j = 0; j < A.cols; j++)
        {
            // populate the sum var.
            sum += A.at(i,j);
        }

        // Return matrix is populated with sum at every i value.
        ret.at(i) = sum;
    }

    return ret;
}

template<typename type>
Propulsion::Matrix<type> Propulsion::Matrix<type>::sumRows(Matrix<type> &A)
{
    // create return object, give it size of the rows from A, 1 for columns.
    Propulsion::Matrix<type> ret;
    ret.rows = A.rows;
    ret.cols = 1;
    ret.totalSize = A.rows;
    ret.M = std::make_unique<type[]>(ret.totalSize);

    for(unsigned i = 0; i < A.rows; i++)
    {
        // Sum starts from zero.
        type sum = (type)0;

        // Loop through all of A, adding the elements on the same row.
        for(unsigned j = 0; j < A.cols; j++)
        {
            // populate the sum var.
            sum += A.at(i,j);
        }

        // Return matrix is populated with sum at every i value.
        ret.at(i) = sum;
    }

    return ret;
}

template<typename type>
Propulsion::Matrix<type> Propulsion::Matrix<type>::sumCols(Matrix<type> &&A)
{
    // create return object, give it size of the rows from A, 1 for rows.
    Propulsion::Matrix<type> ret(1, A.cols);

    for(unsigned j = 0; j < A.cols; j++)
    {
        // Sum starts from zero.
        type sum = (type)0;

        // Loop through all of A, adding the elements on the same row.
        for(unsigned i = 0; i < A.rows; i++)
        {
            // populate the sum var.
            sum += A.at(i,j);
        }

        // Return matrix is populated with sum at every i value.
        ret.at(j) = sum;
    }

    return ret;
}

template<typename type>
Propulsion::Matrix<type> Propulsion::Matrix<type>::sumCols(Matrix<type> &A)
{
    // create return object, give it size of the rows from A, 1 for columns.
    Propulsion::Matrix<type> ret;
    ret.rows = 1;
    ret.cols = A.cols;
    ret.totalSize = A.cols;
    ret.M = std::make_unique<type[]>(ret.totalSize);

    for(unsigned j = 0; j < A.cols; j++)
    {
        // Sum starts from zero.
        type sum = (type)0;

        // Loop through all of A, adding the elements on the same row.
        for(unsigned i = 0; i < A.rows; i++)
        {
            // populate the sum var.
            sum += A.at(i,j);
        }

        // Return matrix is populated with sum at every i value.
        ret.at(j) = sum;
    }

    return ret;
}

template<typename type>
type Propulsion::Matrix<type>::getMax()
{
    // The most unlikely event in this case.
    type max = M[0];

    for(unsigned i = 1; i < this->totalSize; i++)
    {
        // if M of i is greater than the current max value.
        if(M[i] > max)
        {
            max = M[i];
        }
    }

    return max;
}

template<typename type>
type Propulsion::Matrix<type>::getMin()
{
    // The most unlikely event in this case.
    type min = M[0];

    for(unsigned i = 1; i < this->totalSize; i++)
    {
        // If M of i is less than the current min value.
        if(M[i] < min)
        {
            min = M[i];
        }
    }

    return min;
}

template<typename type>
Propulsion::Matrix<type> Propulsion::Matrix<type>::addBroadScalar(Matrix<type> &A, type s)
{
    Propulsion::Matrix<type> ret = A;

    for(unsigned i = 0; i < ret.totalSize; i++)
    {
        ret.at(i) += s;
    }

    return ret;
}

template<typename type>
Propulsion::Matrix<type> Propulsion::Matrix<type>::subtractBroadScalar(Matrix<type> &A, type s)
{
    Propulsion::Matrix<type> ret = A;

    for(unsigned i = 0; i < ret.totalSize; i++)
    {
        ret.at(i) -= s;
    }

    return ret;
}
