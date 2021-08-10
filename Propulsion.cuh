/*
 * Created by Steven Roddan on 7/3/2020.
 *
 * This program is made for the simple reason of getting experience
 * with CUDA and the likes of it. Majority of these methods might be
 * "reinventing the wheel" and that's perfectly okay with me. The
 * goal is to make template driven methods to except a wide variety
 * of data. Some of these functions are host and device defined, via
 * that way anyone can use these for their own purpose they find fit.
 * (Note: this requires the NVCC compiler to compile and use on GPUs,
 * T.F this project isn't as portable as it may be desired)
 *
 * Goals of This Project:
 * 1D Array Functions(Add,Diff,Mult,Avg,Max/Min, Maybe sort(?),etc...
 * 2D Array Functions ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^
 * 3D Array Functions ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^
 * Numerical Analysis Functions
 * Statistical Mathematics
 * Artificial Intelligence
 *
 */

#ifndef PROPULSION_CUH
#define PROPULSION_CUH

#include <chrono>       // chrono objects
#include <cstdio>       // Hmmmm?
#include <cmath>        // floor, ceil.
#include <numeric>
#include <exception>    // Exception handling.
#include <assert.h>     // assert

#include <immintrin.h>  // AVX256
#include <smmintrin.h>
#include <zmmintrin.h>  // AVX512 which I dont have
#include <typeinfo>     // For template info on AVX. Really dumb but only dynamic way I can think of?



#include <iomanip>      // Print formatting, Matrix and Propulsion Functions
#include <iostream>     // cout, endl.
#include <memory>       // S.M.A.R.T. POINTERS because i'M nOT No dUmMy.
#include <random>       // uniform_real_distribution
#include <string>       // toString and Length for Matrix

#include <time.h>       // random

#include <mutex>        // Mutex come on.
#include <future>       // promises, futures
#include <thread>       // Multithreading.
#include <list>
#include <deque>


#include <windows.h>
#include <gdiplus.h>


// Cuda Library etc.
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


// 256bits -> 32 bytes
#define AVX256BYTES 32

#define HOST_STRASSEN_LEAF_SIZE 256

#define MAX_THREADS 1024
#define MAX_BLOCKS 65535
#define CUDA_NO_STRIDE_MAX_THREADS 65536

#define TIME_FORMAT 40
#define TIME_WIDTH 12
#define TIME_PREC 7

#define MATRIX_CUDA_STRASSEN_LEAF_SIZE 2048
// threshold where a 9900k starts to lose to the 2080ti.
#define MATRIX_CUDA_ADD_DIFF_ELEM_SIZE 200000
#define MATRIX_CUDA_DOT_ELEM_SIZE 128
#define MATRIX_COPY_SIZE_DIFF 2000000





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



namespace Propulsion {

    /**
     * \brief           A deleter function used in Matrix unique_ptr as a way to
     *              call the cudaFreeHost method. Since we have pinned memory and
     *              paged memory.
     */
    template<typename type>
    void freePinnedMemory(type* ptr_)
    {
        gpuErrchk(cudaFreeHost(ptr_));
    }

    /**
     * \brief           A deleter function used in Matrix unique_ptr as a way to
     *              call the cudaFreeHost or delete [] type*. Since we can have
     *              both pinned or paged memory.
     */
    template<typename type>
    void freePagedMemory(type* ptr_)
    {
        delete [] ptr_;
    }


    /*
     * Class:       Matrix
     *
     * Purpose:         The purpose of this class is to manage a Matrix class as
     *              part of the Propulsion namespace. Simply, the Matrix class is
     *              a type array, in which is managed by two dimension vars: rows
     *              & cols. This class is also encapsulated by the Artificial Neural
     *              Network classes and Mandelbrot class. This will also be used by
     *              future classes.
     *
     *                  This class also uses CUDA and/or AVX2 when it can where speed-ups
     *              are implemented to do so. While many of the algorithms are naive, the
     *              speed boosts are significant enough anyways.
     *
     *
     */
    template<typename type>
    class Matrix
    {
    public:
        /*
         * Class:        MatrixException
         *
         * Purpose:         The purpose of this class is for the user to be able to handle
         *              exceptions thrown for the matrix class. These exceptions can be thrown
         *              for many various reasons all described in the throw. Child class of
         *              std::exception to inherit various methods.
         */
        class MatrixException : public std::exception
        {
        private:
            const char* file;
            int line;
            const char* func;
            const char* info;
        public:
            MatrixException(const char* msg, const char* file_,
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

        // Enumerators for the values that can be initialized into a constructor of Matrix.
        enum MatrixInitVal { zero, null, ones, twos, custom, random};
        // Enumerators for the types of Matrix that can be constructed.
        enum MatrixInitType { def, diagonal, upperTriangle, lowerTriangle};

        enum MatrixMemType {paged, pinned};


        /*
         * Function     Matrix()
         * @params:     -nothing
         *
         * Purpose:         Default constructor if no row size and col size is provided and/or an
         *              an array to base the entries off. First value is initialized as zero.
         *
         * Author:      Steven Roddan on 8/4/2020.
         */
        Matrix();

        /*
         * Function:    Matrix(const Matrix& copyM)
         * @params:     @copyM - Matrix to be deep copied.
         *
         * Purpose:         Constructor to deep copy another Matrix. 
         */
        Matrix(const Matrix& copyM);


        Matrix(const Matrix& copyM, MatrixMemType memType);


        Matrix(Matrix&&);

        /*
         * Function:    Matrix(unsigned rowsAndColSize, MatrixInitVal mit, type customVal, MatrixInitType miv)
         * @params:     @rowsAndColSize - NxN matrix.
         *              @miv - Enumerator of MatrixInitVal, tells what value you want populated with the matrix.
         *              @customVal - If Enumerator 'custom' is passed in miv, then the user is to provide the custom val
         *              @mit - Enumerator of MatrixInitType, tells what kind of matrix you want generated.
         *
         * Purpose:         Constructor type that builds a matrix with a few arguments in mind. MatrixInitVal and
         *              MatrixInitType are defaulted to their respected values, that way using these functions without
         *              the values specified will automatically generate a zero matrix by the nxn size.
         *
         * Usage:       Propulsion::Matrix<int> M(4,Propulsion::Matrix<int>::custom, 5,Propulsion::Matrix<int>::def);
         *                  ... or
         *              Propulsion::Marix<int> Q(3) // Generate 3x3 zero matrix.
         *
         * Author:      Steven Roddan on 8/4/2020.
         */
        explicit Matrix(unsigned rowAndColSize, MatrixMemType memType = MatrixMemType::pinned, MatrixInitVal = zero, type customVal = 0, MatrixInitType = def );

        /*
         * Function:    Matrix(unsigned rows, unsigned rows, MatrixInitVal mit, type customVal, MatrixInitType miv)
         * @params:     @rows - n of a nxm matrix.
         *              @cols - m of a nxm matrix.
         *              @miv - Enumerator of MatrixInitVal, tells what value you want populated with the matrix.
         *              @customVal - If Enumerator 'custom' is passed in miv, then the user is to provide the custom val
         *              @mit - Enumerator of MatrixInitType, tells what kind of matrix you want generated.
         *
         * Purpose:         Constructor type that builds a matrix with a few arguments in mind. MatrixInitVal and
         *              MatrixInitType are defaulted to their respected values, that way using these functions without
         *              the values specified will automatically generate a zero matrix by the nxm size.
         *
         * Usage:       Propulsion::Matrix<int> M(4,4,Propulsion::Matrix<int>::custom, 5,Propulsion::Matrix<int>::def);
         *                  ... or
         *              Propulsion::Matrix<int> Q(3,4) // Generate 3x4 zero matrix.
         *
         * Author:      Steven Roddan on 8/4/2020.
         */
        Matrix(unsigned, unsigned, MatrixMemType memType = MatrixMemType::pinned, MatrixInitVal = zero, type customVal = 0, MatrixInitType = def);

        /*
         * Function:    Matrix(type *array, unsigned rowAndColSize)
         * @params:     @array - Pointer to an array that will be used to populate the matrice.
         *              @rowAndColSize - Unsigned value of the size of the matrice. Note: This must be the array size.
         *
         * Purpose:         Allows the user to create an NxN matrix with a provided array. If the array pointer is
         *              NULL, then the generated matrix will be a zero square matrix.
         *
         * Usage:       Propulsion::Matrix<int> Q(arr, 3)   // generates matrix 3x3 with arr elements.
         *
         * Author:      Steven Roddan on 8/4/2020.
         */
        Matrix(type *, unsigned,  MatrixMemType memType = MatrixMemType::pinned);

        /*
         * Function:    Matrix(type *array, unsigned rows, unsigned cols, MatrixInitVal miv = MatrixInitVal::custom,
         *                      type customVal = NULL, MatrixInitType mit = MatrixInitType::def)
         * @params:     @array - Pointer to an array that will be used to populate the matrice.
         *              @rows - N of NxM.
         *              @cols - M of NxM.
         *              @miv - The value to be used as populating the matrix of that Matrix Type.
         *              @customVal - If custom is selected as miv.
         *              @mit - Type of matrix.
         *
         * Purpose:         Allows the user to create an NxM matrix with a provided array. If the array pointer is
         *              NULL, then the generated matrix will be a zero rectangle matrix of NxM.
         *
         * Usage:       Propulsion::Matrix<int> Q(arr, 3,4)   // generates matrix 3x4 with arr elements.
         *
         * Author:      Steven Roddan on 8/4/2020.
         */
        Matrix(type *, unsigned , unsigned ,  MatrixMemType memType = MatrixMemType::pinned, MatrixInitVal miv = MatrixInitVal::custom, type customVal = NULL, MatrixInitType mit = MatrixInitType::def);



        /*
         * Method:      print()
         * @params      -none-
         *
         * Purpose:         To print the matrix object as the way it is represented. Goes through each element
         *              and retrieves the max length of every digit. Adds a padding of 2 spaces to each digit for
         *              printing plus the max to give each number the same amount of spacing for even cols/rows.
         *
         * Usage:       int d[] = { 1,2,3,4,
         *                          7,8,9,5,
         *                          4,5,67,8,
         *                          4,4,4,4};
         *
         *              Propulsion::Matrix<int> M(d,4,4);
         *              M.print();
         *              ***prints:
         *              |   1   2   3   4 |
         *              |   7   8   9   5 |
         *              |   4   5  67   8 |
         *              |   4   4   4   4 |***
         *
         * &returns:    &void
         * Author:      Steven Roddan on 8/4/2020.
         */
        void print(std::ostream& oStream = std::cout);

        /*
         * Method:      print(type *array, unsigned rows, unsigned cols)
         * @params:     @array - Pointer to the array to print.
         *              @rows - Total amount of rows.
         *              @cols - Total amount of cols.
         *
         * Purpose:         Static void method, used to print the contents of the provided array and treated as if
         *              it was a matrix. Doesn't change the contents of the array or anything like so.
         *
         * Usage:       int d[] = { 1,2,3,4,
         *                          7,8,9,5,
         *                          4,5,67,8,
         *                          4,4,4,4};
         *              Propulsion::Matrix<int>::print(d,4,4);
         *              ***prints on command:
         *              |   1   2   3   4 |
         *              |   7   8   9   5 |
         *              |   4   5  67   8 |
         *              |   4   4   4   4 |***
         *
         * &returns:    &void
         * Author:      Steven Roddan on 8/4/2020.
         */
        static void print(type *, unsigned, unsigned );


        /**
         * \brief           Adds Parameter matrix to this Matrix element wise.
         *
         * \details         Adds parameter matrix to this Matrix element wise. If
         *              the size of the Matrices is greater than a defined constant
         *              (MATRIX_CUDA_ADD_DIFF_ELEM_SIZE) then CUDA is used, else
         *              AVX2 addition is used.
         *
         * @throws      MatrixException if the dimensions do not match.
         *
         * @param       B Matrix that is being added to this.
         * @param       printTime Bool that prints stats about the overhead of the operation.
         *
         * @return      None.
         */
        void add(const Matrix<type> &B, bool printTime = false);

        /**
         * \brief           Adds a row vector to each row of this matrix. Returns
         *              a Matrix.
         *
         * \details         Adds a row vector to each row of this matrix. If the
         *              size of the row vector param isn't of (1xN) and N doesn't match
         *              the size of the this columns, then a MatrixException is thrown
         *              as the operation can not be executed.
         *
         * @throws      MatrixException if Dimensions aren't (MxN) & (1xN) respectively.
         *
         * @param       b Matrix parameter that should be (1xN) size.
         *
         * @return      Matrix with the added row vector!
         */
        Matrix<type> addRowVector(Matrix<type> &b);

        /**
         * \brief           Adds a row vector to each row of this matrix. Returns
         *              a Matrix.
         *
         * \details         Adds a row vector to each row of this matrix. If the
         *              size of the row vector param isn't of (1xN) and N doesn't match
         *              the size of the this columns, then a MatrixException is thrown
         *              as the operation can not be executed.
         *
         * @throws      MatrixException if Dimensions aren't (MxN) & (1xN) respectively.
         *
         * @param       b Matrix parameter that should be (1xN) size.
         *
         * @return      Matrix with the added row vector!
         */
        Matrix<type> addRowVector(Matrix<type> &&b);

        /**
         * \brief           Adds a col vector to each col of this matrix. Returns
         *              a Matrix.
         *
         * \details         Adds a col vector to each col of this matrix. If the
         *              size of the col vector param isn't of (Nx1) and N doesn't match
         *              the size of the this rows, then a MatrixException is thrown
         *              as the operation can not be executed.
         *
         * @throws      MatrixException if Dimensions aren't (MxN) & (Nx1) respectively.
         *
         * @param       b Matrix parameter that should be (Nx1) size.
         *
         * @return      Matrix with the added col vector!
         */
        Matrix<type> addColVector(Matrix<type> &b);

        /**
         * \brief           Adds a col vector to each col of this matrix. Returns
         *              a Matrix.
         *
         * \details         Adds a col vector to each col of this matrix. If the
         *              size of the col vector param isn't of (Nx1) and N doesn't match
         *              the size of the this rows, then a MatrixException is thrown
         *              as the operation can not be executed.
         *
         * @throws      MatrixException if Dimensions aren't (MxN) & (Nx1) respectively.
         *
         * @param       b Matrix parameter that should be (Nx1) size.
         *
         * @return      Matrix with the added col vector!
         */
        Matrix<type> addColVector(Matrix<type> &&b);

        /**
         * \brief           Performs Matrix Difference on B parameter with this.
         *
         * \details         Performs Matrix Difference on B parameter with this.
         *              The resultant is stored in this Matrix, and only throws
         *              a MatrixException if the dimensions are mismatched with
         *              one another.
         *
         * @throws      MatrixException If the dimensions don't match.
         *
         * @param       B Matrix that is being subtracted to this.
         * @param       printTime Bool that prints stats about the overhead of the operation.
         *
         * @return      None.
         */
        void subtract(const Matrix<type> &B, bool printTime = false);

        /**
         * \brief           Performs dot product on the given Matrix using
         *              CUDA.
         *
         * \details         Performs dot product on the given Matrix using
         *              CUDA. Makes use of Propulsions static method cudaDotProduct
         *              to handle the operation. Throws an exception if This Matrix
         *              columns don't match B's columns. Returns none.
         *
         * @throws      MatrixException If this->cols != B->cols.
         *
         * @param       B Matrix that is being added to this.
         * @param       printTime Bool that prints stats about the overhead of the operation.
         *
         * @return      None.
         */
        void cudaDotProduct(const Matrix<type> &B, bool printTime = false);

        /**
         * \brief           Performs dot product on the given Matrix using
         *              host.
         *
         * \details         Performs dot product on the given Matrix using
         *              host code. Calls Propulsion static method hostDotProduct
         *              to execute the operation if nothing is thrown from this
         *              method. Throws if this matrix's columns don't match B's
         *              columns. Returns none.
         *
         * @throws      MatrixException If this->cols != B->cols
         *
         * @param       B Matrix that is being added to this.
         * @param       printTime Bool that prints stats about the overhead of the operation.
         *
         * @return      None.
         */
        void dot(const Matrix<type> &B, bool printTime = false);

        /**
         * \brief           Performs Element Wise product on this Matrix and
         *              B Matrix Param.
         *
         * \details         Performs Element Wise product on this Matrix and
         *              B Matrix Param. Uses either CUDA or HOST depending
         *              on whether or not the elements is great enough to
         *              get a speedup on CUDA. IF not, resorts to HOST.
         *              Throws MatrixException if the dimensions do not match.
         *
         * @throws      MatrixException If the dimensions do not match.
         *
         * @param       B Matrix that is being added to this.
         * @param       printTime Bool that prints stats about the overhead of the operation.
         *
         * @return      None.
         */
        void schurProduct(const Matrix<type> &B, bool printTime = false);

        /**
         * \brief           Multiplies this Matrix with the provided scalar.
         *
         * \details         Multiplies this Matrix with the provided scalar.
         *              Uses HOST to do this operation.
         *
         * @param       scalar Type param that we are using to mutiply this with.
         *
         * @return      None.
         */
        void multiply( type scalar) noexcept ;

        /**
         * \brief           Dot Product of this and B Matrix using Host
         *              Strassen Recursion.
         *
         * \details         Dot Product of this and B Matrix using Strassen
         *              method to reduce the number of additions and multiplications
         *              to get the desired result. Uses recursion to perform
         *              such operation, giving it a Time Complexity of O(n^2.8).
         *              Throws MatrixException if the col and row doesn't match with
         *              this and B.
         *
         * @throws      MatrixException If the col of this doesnt match the row of B.
         *
         * @param       B Matrix to dot product agaisnt this.
         *
         * @return      None.
         */
        void strassenMultiplication(const Matrix<type> &B);

        /**
         * \brief           Performs Transpose on this Matrix.
         *
         * \details         Performs Transpose on this Matrix. Creates a
         *              Temporary Matrix and moves the unique ptr to that.
         *
         * @return      None.
         */
        void T();

        /**
         * \brief           Gets the Max value in the Matrix.
         *
         * \details         Gets the Max value in the Matrix assuming that
         *              the type has the operator >.
         *
         * @return      Type Max value in the Matrix.
         */
        type getMax();

        /**
         * \brief           Gets the Min value in the Matrix.
         *
         * \details         Gets the Min value in the Matrix assuming that
         *              the type has the operator >.
         *
         * @return      Type Min value in the Matrix.
         */
        type getMin();

        /**
         * \brief           Broadcast s type added to A Matrix Param.
         *
         * \details         Adds the type scalar S to A Matrix in every
         *              element.
         *
         * @param       A Matrix to be added with S to element wise.
         *
         * @param       s Scalar type to be added to A with.
         *
         * @return      Matrix<type> with Broad scalar added to A param.
         */
        static Matrix<type> addBroadScalar(Matrix<type> &A, type s);

        /**
         * \brief           Broadcast s type subtracted to A Matrix Param.
         *
         * \details         Subtract the type scalar S to A Matrix in every
         *              element.
         *
         * @param       A Matrix to be subtracted with S to element wise.
         *
         * @param       s Scalar type to be subtracted to A with.
         *
         * @return      Matrix<type> with Broad scalar subtracted to A param.
         */
        static Matrix<type> subtractBroadScalar(Matrix<type> &A, type s);

        /**
         * \brief           Sum the rows of the Matrix into a vector of row sums.
         *
         * \details         Sums the rows of the Matrix param and returns a Matrix
         *              that is of a vector shape with the sums along the rows.
         *
         * @param       A Matrix that is having each row summed into a Matrix vector.
         *
         * @return      None.
         */
        static Matrix<type> sumRows(Matrix<type> &&A);

        /**
         * \brief           Sum the rows of the Matrix into a vector of row sums.
         *
         * \details         Sums the rows of the Matrix param and returns a Matrix
         *              that is of a vector shape with the sums along the rows.
         *
         * @param       A Matrix that is having each row summed into a Matrix vector.
         *
         * @return      None.
         */
        static Matrix<type> sumRows(Matrix<type> &A);

        /**
         * \brief           Sum the cols of the Matrix into a vector of col sums.
         *
         * \details         Sums the cols of the Matrix param and returns a Matrix
         *              that is of a vector shape with the sums along the cols.
         *
         * @param       A Matrix that is having each col summed into a Matrix vector.
         *
         * @return      None.
         */
        static Matrix<type> sumCols(Matrix<type> &&A);

        /**
         * \brief           Sum the cols of the Matrix into a vector of col sums.
         *
         * \details         Sums the cols of the Matrix param and returns a Matrix
         *              that is of a vector shape with the sums along the cols.
         *
         * @param       A Matrix that is having each col summed into a Matrix vector.
         *
         * @return      None.
         */
        static Matrix<type> sumCols(Matrix<type> &A);


        /**
         * \brief           Returns the pointer of the unique_ptr if needed. This
         *              is unsafe as it doesn't gurantee the lifeline of that pointer.
         *
         * \details         Returns the pointer of the unique_ptr storing the Matrix
         *              data if the user needs it. It is unadvised to do so, as the
         *              pointer is not guranteed to be valid. Discretion is advised.
         *
         * @return      Type array pointer of the Matrix. If needed its provided.
         */
        type* getArray();

        /**
         * \brief           Returns column size of the this.
         *
         * @return      Unsigned Column size.
         */
        unsigned getColSize();

        /**
         * \brief           Returns row size of the this.
         *
         * @return      Unsigned Row size.
         */
        unsigned getRowSize();

        /**
         * \brief           Returns the total size of the matrix.
         *
         * @return      Unsigned Total Size(Rows * Cols)
         */
        unsigned getTotalSize();

        /**
         * \brief           Checks if the Matrix B is equal to this.
         *
         * \details         Checks if the Matrix B is equal to this. First
         *              checks if the dims even match, if so check if the
         *              values are equal.
         *
         * @param       B Matrix to check upon equality.
         *
         * @return      Bool Whether the Matrices are equal in value.
         */
        bool equalTo(const Matrix<type> &B);

        /**
         * \brief           Checks if the Matrix B is equal to this.
         *
         * \details         Checks if the Matrix B is equal to this. First
         *              checks if the dims even match, if so check if the
         *              values are equal.
         *
         * @param       B Matrix to check upon equality.
         *
         * @return      Bool Whether the Matrices are equal in value.
         */
        bool operator==(const Matrix<type> &rhs);

        /**
         * \brief           Checks if this Matrix is an Upper Triangle
         *
         * \details         Checks the left values of the diagonal to
         *              see if they're all zero. Returns false if not,
         *              true if true.
         *
         * @return      Bool Whether this is an Upper Triangle Matrix.
         */
        bool isUpperTriangular();

        /**
         * \brief           Checks if this Matrix is an Lower Triangle
         *
         * \details         Checks the right values of the diagonal to
         *              see if they're all zero. Returns false if not,
         *              true if true.
         *
         * @return      Bool Whether this is an Lower Triangle Matrix.
         */
        bool isLowerTriangular();

        /*
         * Unimplemented so far. No need yet.
         */
        bool isIdentityMatrix();
        /*
         * Unimplemented so far. No need yet.
         */
        bool isSymmetric();

        /**
         * \brief           Returns a Matrix of the row requested.
         *
         * \details         Returns a Matrix obj of the row requested
         *              as row Param. Indexing starting at zero like
         *              normal.
         *
         * @param       row Unsigned param to get that row.
         *
         * @return      Matrix of size 1xN of the row requested.
         */
        Matrix<type> getRowMatrix(unsigned row);

        /**
         * \brief           Returns a Matrix of the col requested.
         *
         * \details         Returns a Matrix obj of the col requested
         *              as col Param. Indexing starting at zero like
         *              normal.
         *
         * @param       col Unsigned param to get that col.
         *
         * @return      Matrix of size Nx1 of the row requested.
         */
        Matrix<type> getColMatrix(unsigned col);

        /**
         * \brief           Returns a Matrix that is partitioned as the
         *              parameters provided from the original matrix.
         *
         * \details         Returns a Matrix that is partitioned as the
         *              parameter provided from the original matrix. If
         *              the params are bad values(overshot the index),
         *              then a Matrix with zero values filled is returned.
         *
         * @param       rowStart Unsigned starting row position.
         * @param       rowEnd Unsigned ending row position.
         * @param       colStart Unsigned starting row position.
         * @param       colEnd Unsigned ending row position.
         *
         * @return      Matrix Of the provided ranges.
         */
        Matrix<type> getRangeMatrix(unsigned rowStart, unsigned rowEnd, unsigned colStart, unsigned colEnd);

        /**
         * \brief           Merges a Matrix to the right of this Matrix.
         *
         * \details         Merges a Matrix to the right of this Matrix
         *              only if it has the same amount of rows.
         *
         * @param       B Matrix to be merged with this.
         *
         * @return      Matrix obj. with the now Merged Matrices.
         */
        Matrix<type> mergeRight(Matrix<type> &B);

        /**
         * \brief           Merges a Matrix to the bottom of this Matrix.
         *
         * \details         Merges a Matrix to the bottom of this Matrix
         *              only if it has the same amount of cols.
         *
         * @param       B Matrix to be merged with this.
         *
         * @return      Matrix obj. with the now Merged Matrices.
         */
        Matrix<type> mergeBelow(Matrix<type> &B);

        /**
         * \brief           Returns a reference to the value at index
         *              i.
         *
         * \details         Returns a reference to the value at index
         *              i. At will throw MatrixException if the index
         *              is out of range.
         *
         * @throw       MatrixException If index is out of range.
         *
         * @param       i Unsigned index value to get reference of.
         *
         * @return      Type Reference to value at i index.
         */
        type& at(unsigned i);

        /**
         * \brief           Returns a reference to the value at index
         *              i,j.
         *
         * \details         Returns a reference to the value at index
         *              i,j. At will throw MatrixException if the index
         *              is out of range.
         *
         * @throw       MatrixException If index is out of range.
         *
         * @param       i Unsigned row index value to get reference of.
         * @param       j Unsigned col index value to get reference of.
         *
         * @return      Type Reference to value at i,j index.
         */
        type& at(unsigned i, unsigned j);

        /**
         * \brief           Returns a reference to the value at index
         *              i.
         *
         * \details         Returns a reference to the value at index
         *              i. This method will not throw any exceptions
         *              due to overhead and at() method availability.
         *
         * @throw       MatrixException If index is out of range.
         *
         * @param       i Unsigned index value to get reference of.
         *
         * @return      Type Reference to value at i index.
         */
        type& operator()(unsigned i);

        /**
         * \brief           Returns a reference to the value at index
         *              i,j.
         *
         * \details         Returns a reference to the value at index
         *              i,j. At will throw MatrixException if the index
         *              is out of range.
         *
         * @throw       MatrixException If index is out of range.
         *
         * @param       i Unsigned row index value to get reference of.
         * @param       j Unsigned col index value to get reference of.
         *
         * @return      Type Reference to value at i,j index.
         */
        type& operator()(unsigned i ,unsigned j);

        // Operator Functions
        Matrix<type> operator+(Matrix<type> &rhs);
        Matrix<type> operator-(Matrix<type> &rhs);
        Matrix<type> operator*(type);
        Matrix<type> operator*(const Matrix<type> &rhs);
        Matrix<type>& operator=(const Matrix<type>  &rhs);

        /**
         * \brief           Zero pads this Matrix with specified
         *              rows and cols. Zero can be accepted.
         *
         * \details         Zero pads this Matrix with specified
         *              rows and cols. Zero can be provided as a
         *              parameter, as it will not add it to the
         *              matrix for the desired row/col.
         *
         * @param       rows Unsigned how many rows to zero pad.
         * @param       cols Unsigned how many cols to zero pad.
         */
        void pad(unsigned rows, unsigned cols);

        /**
         * \brief           Populate Matrix with Uniform Distro Method.
         *
         * @param       lRange Type value to exclusive start from on left.
         * @param       rRange Type value to exclusive start from on right.
         *
         * @return      None.
         */
        void populateWithUniformDistribution(type lRange, type rRange);

        /**
         * \brief           Removes the desired row from the Matrix.
         *
         * @param       rowToRem Unsigned index of row to remove.
         *
         * @return      None.
         */
        Matrix<type> removeRow(unsigned rowToRem);

        /**
         * \brief           Removes the desired col from the Matrix.
         *
         * @param       colToRem Unsigned index of col to remove.
         *
         * @return      None.
         */
        Matrix<type> removeCol(unsigned colToRem);

        /**
         * \brief           Deep copies the contents of param Matrix copyM
         *              into the return Matrix.
         *
         * @return      Matrix copy of specified param.
         */
        static Matrix<type> copy(Matrix<type> copyM);

        /**
         * \brief           Generates a random real distrobution into Matrix
         *              A with ranges lVal to rVal exclusive.
         *
         * @param       A Matrix to random fill with values.
         * @param       lVal type value for left bounds exclusive.
         * @param       rVal type value for right bounds exclusive.
         *
         * @return      None.
         */
        static void randomRealDistribution(Matrix<type> &A, type lVal, type rVal);

        /**
         * \brief           Generates a random real distrobution into Matrix
         *              A with ranges lVal to rVal exclusive.
         *
         * @param       A Matrix to random fill with values.
         * @param       lVal type value for left bounds exclusive.
         * @param       rVal type value for right bounds exclusive.
         *
         * @return      None.
         */
        static void randomRealDistribution(std::shared_ptr<Matrix<type>> A, type lVal, type rVal);

        /**
         * \brief           Destructor!
         */
        ~Matrix();
    private:
        /// unique_ptr to type array. Has a custom deleter for paged memory and pinned memory arrays.
        std::unique_ptr<type[], void(*)(type*)> M; // Matrix Array as 1D.

        /// rows, cols, and totalSize attributes.
        unsigned rows, cols, totalSize;
        MatrixMemType memT = MatrixMemType::pinned;


        void generateMatrix(MatrixInitVal, MatrixInitType, type customVal, type * = nullptr);
        std::unique_ptr<type[], void(*)(type*)> generateZeroMatrixArray(unsigned );
        std::unique_ptr<type[], void(*)(type*)> generateNullMatrixArray(unsigned );
        std::unique_ptr<type[], void(*)(type*)> generateDefaultMatrixArray(unsigned, type, type *);
        std::unique_ptr<type[], void(*)(type*)> generateDiagonalMatrixArray(unsigned, type, type *);

        void createPinnedMemory();
        void createPagedMemory();

        static Propulsion::Matrix<type> recursiveStrassen(Matrix<type> A, Matrix<type> B);
    };

    template<typename type>
    class Tensor;


    class Mandelbrot {
    private:
        std::unique_ptr< Propulsion::Matrix< int>> Mandel = nullptr;
        std::unique_ptr< Propulsion::Matrix< unsigned>> epoch = nullptr;
        std::shared_ptr< Propulsion::Matrix< int>> colorPicker = nullptr;


        unsigned iterations;
        unsigned currentEpochSelection = 0;


        long windowWidthPixels;
        long windowHeightPixels;
        long clientWidthPixels;
        long clientHeightPixels;

        double leftBound;
        double rightBound;
        double topBound;
        double bottomBound;
        double zoomFactor;

        bool redraw = false;
        bool calculateWithCUDA = true;


        WNDCLASSA* windowClass;
        HWND hwnd = nullptr;


        std::mutex mandelMutex;

        void paintWindow(int device = 0);
        void zoomInOnCursor();
        void zoomOutOnCursor();
        void generateColorScheme(unsigned totalColors);
        void generateColorSchemeV2(unsigned totalColors);

    public:
        /*
         * @brief Constructor to create a Mandlebrot object.
         * @param width Total size of horizontal pixels on window creation.
         * @param height Total size of height pixels on window creation.
         */
        explicit Mandelbrot(unsigned width = 640, unsigned height = 480, double leftBound = -2.0, double rightBound = 1.0, double topBound = 2.0, double bottomBound = -2.0, double zoomFactor = .125);

        void simulate();

        std::unique_ptr<Propulsion::Matrix<int>> static test(std::shared_ptr<Propulsion::Matrix<double>>);

        std::unique_ptr<Propulsion::Matrix<int>> static calculateMandelSingleThreaded(unsigned wPixels, unsigned hPixels, double leftBound,
                                                                                      double rightBound, double topBound, double bottomBound,
                                                                                      unsigned maxIterations,std::shared_ptr< Propulsion::Matrix< int>> colorPicker);
        std::unique_ptr<Propulsion::Matrix<int>> static calculateMandelAVX256(unsigned wPixels, unsigned hPixels, double leftBound,
                                                                                      double rightBound, double topBound, double bottomBound,
                                                                                      unsigned maxIterations,std::shared_ptr< Propulsion::Matrix< int>> colorPicker, bool printTime = false);
        std::unique_ptr<Propulsion::Matrix<int>> static calculateMandelMultiThreaded(unsigned threads, unsigned wPixels, unsigned hPixels, double leftBound, double rightBound,
                                                                            double topBound, double bottomBound, unsigned maxIterations,
                                                                            std::shared_ptr< Propulsion::Matrix< int>> colorPicker);

        std::unique_ptr<Propulsion::Matrix<int>> static calculateMandelCUDA(unsigned wPixels, unsigned hPixels, double leftBound, double rightBound,
                                                                            double topBound, double bottomBound, unsigned maxIterations,
                                                                            std::shared_ptr< Propulsion::Matrix< int>> colorPicker);
    };


    class ArtificialNeuralNetwork
    {
    public:
        class LayerDense
        {
        private:
            std::shared_ptr<Matrix<double>> weights = nullptr;
            std::shared_ptr<Matrix<double>> biases = nullptr;
            std::shared_ptr<Matrix<double>> outputLayer = nullptr;
        public:
            LayerDense(unsigned nInputs, unsigned nNeurons);

            void forward(LayerDense &inputs);
            void forward(Matrix<double> &inputs);

            std::shared_ptr<Matrix<double>> getWeights();
            std::shared_ptr<Matrix<double>> getBiases();
            std::shared_ptr<Matrix<double>> getOutputLayer();

            void printWeights();
            void printBiases();
            void printOutputLayer();
        };

        class ActivationReLU
        {
        private:
            std::shared_ptr<Matrix<double>> outputLayer;
        public:
            void forward(LayerDense &input);
            std::shared_ptr<Matrix<double>> getOutputLayer();
            void printOutputLayer();
        };

        class ActivationSigmoid
        {
        private:
            std::shared_ptr<Matrix<double>> outputLayer;
        public:
            void forward(LayerDense &input);
            std::shared_ptr<Matrix<double>> getOutputLayer();
            void printOutputLayer();
        };

        class ActivationSoftmax
        {
        private:
            std::shared_ptr<Matrix<double>> outputLayer;
        public:
            void forward(LayerDense &input);
            std::shared_ptr<Matrix<double>> getOutputLayer();
            void printOutputLayer();
        };

        class Loss
        {
        private:
            double loss;
        public:
            double regularization(Matrix<double> &output, Matrix<double> &y);
        };

        class LossCategoricalCrossentropy : public Loss
        {
        private:
        public:
            bool calculate(std::shared_ptr<Matrix<double>> y_pred, std::shared_ptr<Matrix<double>> y_true);
        };



        void test();
    };

    static void helloWorld();

    /*
     * ____   ____             _
     *|_  _| |_  _|           / |_
     *  \ \   / /.---.  .---.`| |-' .--.   _ .--.
     *   \ \ / // /__\\/ /'`\]| | / .'`\ \[ `/'`\]
     *    \ ' / | \__.,| \__. | |,| \__. | | |
     *     \_/   '.__.''.___.'\__/ '.__.' [___]
     * Vector Related Functions Below.
     */



    /*
     * Function:    vectorAdd(type *a, type *b, unsigned size)
     * @params:     @a - Vector A
     *              @b - Vector B
     *              @size - Size of the vectors.
     *
     * Purpose:         The purpose of this function is to handle adding vectors to one another.
     *              Its a host driven static member that returns a pointer to a vector, which is
     *              the sum of the two vectors A & B. Requires the size parameter as its an array
     *              that is required to be passed via pointer.
     *
     * Usage:       int x[3] = {1,2,3};
     *              int y[3] = {3,2,1};
     *              auto z = Propulsion::vectorAdd(&x,&y,3);
     *
     * Returns:     &type *C - The sum of Vectors A and B.
     * Author:      Steven Roddan on 7/26/2020.
     */
    template <typename type> static type* vectorAdd(type*, type*, unsigned );




    /*
     * Function:    vectorSubtract(type *a, type *b, unsigned size)
     * @params:     @a - Vector A
     *              @b - Vector B
     *              @size - Size of the vectors.
     *
     * Purpose:         The purpose of this function is to handle the difference between vectors
     *              A and B. Similiar to vectorAdd and Scalar. Host driven for loop
     *              that generates Vector C from A and B.
     *
     * Usage:       int x[3] = {1,2,3};
     *              int y[3] = {3,2,1};
     *              auto z = Propulsion::vectorSubtract(&x,&y,3);
     *
     * Returns:     &type *C - The Diff of Vectors A and B.
     * Author:      Steven Roddan on 7/26/2020.
     */
    template <typename type> static type* vectorSubtract(type*, type*, unsigned );




    /*
     * Function     vectorScalar(type *a, type *b, unsigned size)
     * @params:     @a - Vector A
     *              @Scalar - Scalar to Multiply
     *              @size - Size of the vectors
     *
     * Purpose:         The purpose of this function is to mulitply a vector by a scalar. This in
     *              a way is a two part function. Instead of defining a divide function, it is best
     *              to use the scalar to divide as well. E.g. A / (3/7) can be rewritten as A * 7/3...
     *
     * Usage:       float x[3] = {1,2,3}
     *              float d = 3/7
     *
     *              // Where we want to divide by 3.7 hence 1/d...
     *              auto z = Propulsion::vectorScalar(x, 1/d, 3);
     *
     * Returns:     &type *r - The Vector after scalar applied to it
     * Author:      Steven Roddan on 7/26/2020
     */
    template <typename type> static type* vectorScalar(type*,type, unsigned );




    /*
     * Function:    vectorCrossProduct(type *a, type *b)
     * @params:     @a - Vector A
     *              @b - Vector B
     *
     * Purpose:         The purpose of this function is to take the vector cross product of two
     *              vectors and return the cross product in the form of a vector. Creates a new
     *              one therefore no orig. vector is modified. Cross Product is defined in such
     *              way that A x B = C[0] = A[1]*B[2] - A[2]*A[1];
     *                               C[1] = A[2]*B[0] - A[0]*B[2];
     *                               C[2] = A[0]*B[1] - A[1]*B[0];
     *
     * Usage:       int x[3] = {1,2,3};
     *              int y[3] = {3,2,1};
     *              int *z = Propulsion::vectorCrossProduct(x,y);
     *              // z = { -4, 8, -4 }
     *
     * Returns:     &type *c - Cross Product Vector of A and B
     * Author:      Steven Roddan on 7/26/2020
     */
    template <typename type> static type* vectorCrossProduct(type*, type*);



    /*
     * Function:    vectorDotProduct(aType *a, aType *b, unsigned size)
     * @params:     @a - Vector A
     *              @b - Vector B
     *              @size - Size of Vectors
     *
     * Purpose:         The purpose of this function is to calculate the dot product of two
     *              vectors. Host driven function that uses a for loop to achieve this, by storing
     *              the sum of the vector elements. The first template type is defined as type. This
     *              is the return type of the summation of the multiplied elements. aType is the
     *              second template type which stands for Array type. Therefore it is possible to
     *              have different types.
     *
     * Usage:       int x[3] = {1,2,3};
     *              int y[3] = {3,2,1};
     *              int z = Propulsion::vectorDotProduct<int, int>(x,y, 3);
     *              // z = 10...]
     *
     * Returns:     &dotProduct - The Dot Product of the Two Vectors
     * Author:      Steven Roddan on 7/26/2020
     */
    template <typename type, typename aType> static type vectorDotProduct(aType *, aType *, unsigned );


    /*
     * Fuction:     vectorMagnitude(aType *a, unsigned size)
     * @params:     @a - Vector A
     *              @size - Size of Vector A
     *
     * Purpose:         The purpose of this function is to calculate the magnitude of the vector
     *              A. This is the total distance the vector covers. This can be calculated via
     *              x^2+y^2+z^2....n_var^2. Returns a value of the type specified.
     *
     * Usage:       int x[3] = {1,2,3};
     *              int m = Propulsion::vectorMagnitude<int,int>(x,3);
     *              // m = 13;
     *
     * Returns      &magnitude - Magnitude of Vector A
     * Author:      Steven Roddan on 7/26/2020
     */
    template <typename type, typename aType> static type vectorMagnitude(aType*, unsigned );


    /*
     * Function:    vectorAngleBetweenVectors(type
     * @params:     @angle - Reference to the radian of the two vectors
     *              @a - Vector A
     *              @b - Vector B
     *              @size - Size of Vectors
     *
     * Purpose:         The purpoe of this function is to calculate the angle in radians of
     *              the vectors A and B. This can be calculated by taking the dot product of
     *              A and B, then dividing that by the Magnitudes of A and B multiplied by
     *              one another. Returns a -1 if dividing by zero. Else returns 0 for success
     *              as the angle variable is modified via reference.
     *
     * Usage:       int x[3] = {1,2,3};
     *              int y[3] = {3,2,1};
     *              double radian = 0;
     *              int temp = Propulsion::vectorAngleBetweenVectors(radian, x,y,3);
     *
     *              if(temp == -1)
     *              {
     *                  .... handle the zero vector
     *              }
     *
     * Returns      &int - The value returned is in regards of if a zero vector was used. -1 for
     *              a zero vector. 0 For okay. 
     */
    template <typename type,typename aType> static int vectorAngleBetweenVectors( type &,aType *, aType *, unsigned );

    /*
     * Function:    hostAdd1D
     */
    template <typename type> static void cudaAdd1DArrays(type *, type *, type *, unsigned, bool = false);
    template <typename type> static void cudaAdd1DArraysWithStride(type *, type *, type *, unsigned, bool = false);

    template <typename type> static void hostAdd1DArrays(type *, type *, type *, unsigned, bool = false);
    static void hostAdd1DArraysInt16AVX256(short *, short *, short *, unsigned, bool = false);
    static void hostAdd1DArraysInt32AVX256(int *, int *, int *, unsigned, bool = false);
    static void hostAdd1DArraysUInt32AVX256(unsigned *, unsigned *, unsigned *, unsigned, bool = false);
    static void hostAdd1DArraysFloat32AVX256(float *, float *, float *, unsigned, bool = false);
    static void hostAdd1DArraysDouble64AVX256(double *, double *, double *, unsigned, bool = false);
    template <typename type> static void hostAdd1DArraysAVX256(type *, type *, type *, unsigned, bool = false);


    // 1D Difference Functions
    template <typename type> static void cudaSubtract1DArrays(type *, type *, type *, unsigned, bool = false);
    template <typename type> static void cudaSubtract1DArraysWithStride(type *, type *, type *, unsigned, bool = false);

    template <typename type> static void hostSubtract1DArrays(type *, type *, type *, unsigned, bool = false);
    static void hostSubtract1DArraysInt16AVX256(short *, short *, short *, unsigned, bool = false);
    static void hostSubtract1DArraysInt32AVX256(int *, int *, int *, unsigned, bool = false);
    static void hostSubtract1DArraysUInt32AVX256(unsigned *, unsigned *, unsigned *, unsigned, bool = false);
    static void hostSubtract1DArraysFloat32AVX256(float *, float *, float *, unsigned, bool = false);
    static void hostSubtract1DArraysDouble64AVX256(double *, double *, double *, unsigned, bool = false);
    template <typename type> static void hostSubtract1DArraysAVX256(type *, type *, type *, unsigned, bool = false);


    // 1D Multiplication Functions
    template <typename type> static void hostMultiply1DArrays(type *, type *, type *, unsigned, bool = false);
    template <typename type> static void hostMultiply1DArrayByScalar(type *, type, unsigned, bool = false);
    template<typename type> static void hostDotProduct(type *, type*, type *, unsigned aRows, unsigned aColsBRows, unsigned bCols, bool = false);
    template <typename type> static void cudaDotProduct(type *a, type *b, type *c, unsigned, unsigned, unsigned, bool = false);
    template <typename type> static void cudaMultiply1DArrayByScalar(type *, type, unsigned, bool = false);

    // Schurs/Hadamard Product
    template<typename type> static void hostSchurProduct(type *, type *, type *, unsigned, bool = false);
    template<typename type> static void cudaSchurProduct(type *, type *, type *, unsigned, bool = false);

    // 1D Division Functions
    template <typename type> static void hostDivide1DArrays(type *, type *, type *, unsigned, bool = false);
    template <typename type> static void cudaDivide1DArrays(type *, type *, type *, unsigned, bool = false);

    // 1D Stencil Functions
    template <typename type> static void hostStencilSum1DArrays(type *, type *, unsigned, unsigned, bool = false);
    template <typename type> static void cudaStencilSum1DArrays(type *, type *, unsigned, unsigned, bool = false);

    // 2D Addition Functions
    template <typename type> static void hostAdd2DMatrices(type *[], type *[], type *[], unsigned rSize, unsigned cSize);
    template <typename type> static void cudaAdd2DMatrices(type *[], type *[], type *[], unsigned rSize, unsigned cSize);

    template <typename type>
    static void cudaCopyArray(type *, type *, unsigned totalSize, bool printTime = false);
};

#include "Tensor.cuh"
#include "ANN/ArtificialNeuralNetwork.cu"
#include "ANN/ActivationReLU.cu"
#include "ANN/ActivationSigmoid.cu"
#include "ANN/ActivationSoftmax.cu"
#include "ANN/LayerDense.cu"
#include "ANN/LossFunctions.cu"
#include "Mandelbrot/Mandelbrot.cu"
#include "Mandelbrot/MandelbrotKernel.cu"
#include "Matrix.cu"
#include "MatrixNumerical.cu"
#include "PropulsionVectorOperations.cu"        // Vector Operations
#include "Propulsion.cu"                        // One Dimensional Matrix Operations






#endif //PROPULSION_CUH
