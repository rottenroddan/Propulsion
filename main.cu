#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <string>
#include "Propulsion.cuh"
#include <windows.system.h>
#include "cuda_runtime.h"
#include "cublas.h"
#include "device_launch_parameters.h"

void printExceptionTestResults(std::string msg, bool passedOrFailed, std::string fileName, unsigned lineNum)
{
    CONSOLE_SCREEN_BUFFER_INFO info;
    HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
    GetConsoleScreenBufferInfo(hStdout, &info);
    WORD origAttr = info.wAttributes;

    if(passedOrFailed)
    {
        SetConsoleTextAttribute(hStdout, 10);
        std::cout << "Passed: " << msg << std::endl;
    }
    else
    {
        SetConsoleTextAttribute(hStdout, 12);
        std::cout << "Failed: " << msg << " On line: " << lineNum << " of File: " << fileName << std::endl;
    }

    SetConsoleTextAttribute(hStdout, origAttr);
}

template <typename type> void printError1D(type *hostC, type *cudaC, unsigned C)
{
    int right = 0;
    for(unsigned i = 0; i < C; i++)
    {
        if(hostC[i] == cudaC[i])
        {
            right++;
        }
    }

    std::cout << "Accr: " << (double)right/C * 100.00 << "%" << std::endl;
}

double t1()
{
    std::cout << "This is a thread boi!" << std::endl;
    return 100.7;
}

void test_one_dimensional_array_operations()
{
    const long long COLS = 2000000;  // Array size.
    int elements = COLS;        // Idk why I did this but its okay...
    int radius = 4;

    /*
    static int a[COLS];
    static int b[COLS];
    static int host_c[COLS];
    static int cuda_c[COLS];*/

    int *a = new int[COLS];
    int *b = new int[COLS];
    int *host_c = new int[COLS];
    int *cuda_c = new int[COLS];



    for(int i = 0; i < COLS; i++)
    {
        a[i] = i + 1;
        b[i] = i + 1;
        host_c[i] = 0;
        cuda_c[i] = 0;
    }


    std::cout << std::endl <<"1D Array Summation on  " << elements << " elements!" << std::endl
              << "[Device]---------------[Test]-----------------[Time]--------[Bandwidth]--" << std::endl;

    Propulsion::hostAdd1DArrays(a, b, host_c, COLS, true);
    Propulsion::hostAdd1DArrays(a, b, host_c, COLS, true);
    Propulsion::hostAdd1DArrays(a, b, host_c, COLS, true);
    Propulsion::hostAdd1DArrays(a, b, host_c, COLS, true);
    Propulsion::hostAdd1DArrays(a, b, host_c, COLS, true);
    Propulsion::hostAdd1DArrays(a, b, host_c, COLS, true);

    Propulsion::hostAdd1DArraysInt32AVX256(a, b, host_c, COLS, true);
    Propulsion::hostAdd1DArraysInt32AVX256(a, b, host_c, COLS, true);
    Propulsion::hostAdd1DArraysInt32AVX256(a, b, host_c, COLS, true);
    Propulsion::hostAdd1DArraysInt32AVX256(a, b, host_c, COLS, true);
    Propulsion::hostAdd1DArraysInt32AVX256(a, b, host_c, COLS, true);
    Propulsion::hostAdd1DArraysInt32AVX256(a, b, host_c, COLS, true);

    //Propulsion::hostAdd1DArraysInt32AVX512(a, b, host_c, COLS);

    Propulsion::cudaAdd1DArrays(a, b, cuda_c, COLS, true);
    Propulsion::cudaAdd1DArrays(a, b, cuda_c, COLS, true);
    Propulsion::cudaAdd1DArrays(a, b, cuda_c, COLS, true);
    Propulsion::cudaAdd1DArrays(a, b, cuda_c, COLS, true);
    Propulsion::cudaAdd1DArrays(a, b, cuda_c, COLS, true);
    Propulsion::cudaAdd1DArrays(a, b, cuda_c, COLS, true);

    printError1D(host_c, cuda_c, COLS);

    Propulsion::cudaAdd1DArraysWithStride(a, b, cuda_c, COLS, true);
    Propulsion::cudaAdd1DArraysWithStride(a, b, cuda_c, COLS, true);
    Propulsion::cudaAdd1DArraysWithStride(a, b, cuda_c, COLS, true);
    Propulsion::cudaAdd1DArraysWithStride(a, b, cuda_c, COLS, true);
    Propulsion::cudaAdd1DArraysWithStride(a, b, cuda_c, COLS, true);
    Propulsion::cudaAdd1DArraysWithStride(a, b, cuda_c, COLS, true);

    printError1D(host_c, cuda_c, COLS);

    std::cout << std::endl <<"1D Array Difference on  " << elements << " elements!" << std::endl
              << "[Device]---------------[Test]-----------------[Time]--------[Bandwidth]--" << std::endl;

    Propulsion::hostSubtract1DArrays(a, b, host_c, COLS, true);
    Propulsion::hostSubtract1DArrays(a, b, host_c, COLS, true);
    Propulsion::hostSubtract1DArrays(a, b, host_c, COLS, true);
    Propulsion::hostSubtract1DArrays(a, b, host_c, COLS, true);
    Propulsion::hostSubtract1DArrays(a, b, host_c, COLS, true);
    Propulsion::hostSubtract1DArrays(a, b, host_c, COLS, true);

    Propulsion::cudaSubtract1DArrays(a, b, cuda_c, COLS, true);
    Propulsion::cudaSubtract1DArrays(a, b, cuda_c, COLS, true);
    Propulsion::cudaSubtract1DArrays(a, b, cuda_c, COLS, true);
    Propulsion::cudaSubtract1DArrays(a, b, cuda_c, COLS, true);
    Propulsion::cudaSubtract1DArrays(a, b, cuda_c, COLS, true);
    Propulsion::cudaSubtract1DArrays(a, b, cuda_c, COLS, true);

    printError1D(host_c, cuda_c, COLS);

    Propulsion::cudaSubtract1DArraysWithStride(a, b, cuda_c, COLS, true);
    Propulsion::cudaSubtract1DArraysWithStride(a, b, cuda_c, COLS, true);
    Propulsion::cudaSubtract1DArraysWithStride(a, b, cuda_c, COLS, true);
    Propulsion::cudaSubtract1DArraysWithStride(a, b, cuda_c, COLS, true);
    Propulsion::cudaSubtract1DArraysWithStride(a, b, cuda_c, COLS, true);
    Propulsion::cudaSubtract1DArraysWithStride(a, b, cuda_c, COLS, true);

    printError1D(host_c, cuda_c, COLS);

    std::cout << std::endl <<"1D Array Scalars on  " << elements << " elements!" << std::endl
              << "[Device]---------------[Test]-----------------[Time]--------[Bandwidth]--" << std::endl;

    Propulsion::hostMultiply1DArrayByScalar(a,4,COLS, true);
    Propulsion::hostMultiply1DArrayByScalar(a,4,COLS, true);
    Propulsion::hostMultiply1DArrayByScalar(a,4,COLS, true);
    Propulsion::hostMultiply1DArrayByScalar(a,4,COLS, true);
    Propulsion::hostMultiply1DArrayByScalar(a,4,COLS, true);
    Propulsion::hostMultiply1DArrayByScalar(a,4,COLS, true);

    Propulsion::cudaMultiply1DArrayByScalar(a,4,COLS, true);
    Propulsion::cudaMultiply1DArrayByScalar(a,4,COLS, true);
    Propulsion::cudaMultiply1DArrayByScalar(a,4,COLS, true);
    Propulsion::cudaMultiply1DArrayByScalar(a,4,COLS, true);
    Propulsion::cudaMultiply1DArrayByScalar(a,4,COLS, true);
    Propulsion::cudaMultiply1DArrayByScalar(a,4,COLS, true);

    printError1D(host_c, cuda_c, COLS);

    std::cout << std::endl <<"1D Array Stencil on  " << elements << " elements with radius: " << radius << std::endl
              << "[Device]---------------[Test]-----------------[Time]--------[Bandwidth]--" << std::endl;


    Propulsion::hostStencilSum1DArrays(a,host_c,COLS,radius, true);
    Propulsion::hostStencilSum1DArrays(a,host_c,COLS,radius, true);
    Propulsion::hostStencilSum1DArrays(a,host_c,COLS,radius, true);
    Propulsion::hostStencilSum1DArrays(a,host_c,COLS,radius, true);
    Propulsion::hostStencilSum1DArrays(a,host_c,COLS,radius, true);
    Propulsion::hostStencilSum1DArrays(a,host_c,COLS,radius, true);


    Propulsion::cudaStencilSum1DArrays(a,cuda_c,COLS,radius, true);
    Propulsion::cudaStencilSum1DArrays(a,cuda_c,COLS,radius, true);
    Propulsion::cudaStencilSum1DArrays(a,cuda_c,COLS,radius, true);
    Propulsion::cudaStencilSum1DArrays(a,cuda_c,COLS,radius, true);
    Propulsion::cudaStencilSum1DArrays(a,cuda_c,COLS,radius, true);
    Propulsion::cudaStencilSum1DArrays(a,cuda_c,COLS,radius, true);

    printError1D(host_c, cuda_c, COLS);


    int x[3] = {1,2,3};
    int y[3] = {3,2,1};
    double angle;
    Propulsion::vectorAngleBetweenVectors<double, int>(angle,x,y,3);

    auto z = Propulsion::vectorCrossProduct(x,y);
    auto w = Propulsion::vectorScalar(x,42,3);


    std::cout << "Magnitude of z: " << Propulsion::vectorMagnitude<float, int>(z,3) << std::endl;
    std::cout << "Dot Product of x and y: " << Propulsion::vectorDotProduct<int,int>(x,y,3) << std::endl;
    std::cout << "Angle Between x and y: " << angle << std::endl;
    std::cout << "Scalar of 42*x is [" << w[0] << " , " << w[1] << " , " << w[2] << "]" << std::endl;
    std::cout << "Cross Product of x and y is [" << z[0] << " , " << z[1] << " , " << z[2] << "]"<<std::endl;



    std::cout << std::endl;
    
    return; // End function.
}

void matrixFunctions()
{
    //Matrix<int> m(9);
    int d[] = {1,2,3,4,
               7,8,9,5,
               4,5,67,8,
               4,4,4,4};

    Propulsion::Matrix<int> M(4,4,Propulsion::Matrix<int>::custom, 5,Propulsion::Matrix<int>::def);
    M.print();
    Propulsion::Matrix<int> Q(d, 4,4);
    Q.print();

    for(unsigned i = 0; i < 4; i++)
    {
        for(unsigned j = 0; j < 4; j++)
        {
            Q.at(i,j) = 66;
            std::cout << Q.at(i,j) << " ";
        }
        std::cout << std::endl;
    }

    try {
        Q.at(99);
    }
    catch (std::out_of_range &e1)
    {
        std::cout << e1.what();
    }

    float H[] = {1,2,3,4};
    Propulsion::Matrix<int> Mat(4,3,Propulsion::Matrix<int>::MatrixInitVal::custom, 3, Propulsion::Matrix<int>::diagonal);
    Mat.print();

    std::cout << Mat.at(3,2) << std::endl;

    // Program this constuctor later.
    Propulsion::Matrix<float> Nat(H,4,4, Propulsion::Matrix<float>::MatrixInitVal::custom, Propulsion::Matrix<float>::diagonal);

    float xyz[] = {0,0,1,
                   1,1,1,
                   1,0,1,
                   0,1,1};

    float zyx[] = {0,1,1,0};

    Propulsion::Matrix<float> in(xyz, 4,3);
    Propulsion::Matrix<float> out(zyx, 4,1);
    Propulsion::Matrix<float> ex = in;

    float jkl[] = {4,3,2,5,6,7};

    Propulsion::Matrix<float> tEx(jkl,2,3);


    ex.print();
    std::cout << "Debug" << std::endl;
    in.print();
    out.print();

    tEx.print();
    tEx.T();


    std::cout << "Transpose: " << std::endl;
    tEx.print();

    tEx = ex;

    tEx.print();

    auto GM = Propulsion::Matrix<float>::copy(tEx);
    std::cout << "Copy: " << std::endl;
    GM.print();

    std::cout << "Add:" << std::endl;
    GM.add(tEx);
    GM.print();

    float dotA[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};
    float dotB[] = {1,2,3,4,5,6,7,8,9,10,11,12};
    float dotC[] = {9,1,2,3,4,5,6,7,8,97,11,12};

    Propulsion::Matrix<float> dotAMat(dotA,3,6);
    Propulsion::Matrix<float> dotBMat(dotB,6,2);
    Propulsion::Matrix<float> dotCMat(dotC,6,2);


    std::cout << "Multiplying:" << dotAMat(1,1) << std::endl;
    dotAMat.print();
    std::cout << "By:" << std::endl;
    dotBMat.print();
    std::cout << "Yields:" << std::endl;

    dotAMat.dot(dotBMat);
    dotAMat.print();

    std::cout << "Matrix Addition with Operator" << std::endl;

    dotAMat.print();


    dotAMat.print();
    //examp.print();
    dotAMat = dotBMat + dotCMat;



    std::cout << "I HATE HOW LONG IT TOOK ME FOR THIS: " << std::endl;

    dotAMat.print();


    Propulsion::Matrix<float> dotDMat(dotA,3,6);
    Propulsion::Matrix<float> dotEMat(dotB,6,2);

    std::cout << "Multiplication with operators" << std::endl;
    (dotDMat * dotEMat).print();

    std::cout << "Multiplication with operators(scalar)" << std::endl;
    (dotDMat * 5.0).print();


    std::cout << "Get Row Matrix Now" << std::endl;
    (dotDMat * 5.0).getRowMatrix(0).print();
    (dotDMat * 5.0).getRowMatrix(1).print();
    (dotDMat * 5.0).getRowMatrix(2).print();

    (dotDMat * 5.0).getColMatrix(0).print();
    (dotDMat * 5.0).getColMatrix(1).print();
    (dotDMat * 5.0).getColMatrix(2).print();
    (dotDMat * 5.0).getColMatrix(3).print();
    (dotDMat * 5.0).getColMatrix(4).print();
    (dotDMat * 5.0).getColMatrix(5).print();

    (dotDMat * 5.0).getRangeMatrix(0,1,1,4).print();

    std::cout << "Merge Right" << std::endl;
    ((dotDMat*5.0).mergeRight((dotDMat))).print();


    std::cout << "Merge Below" << std::endl;
    auto xyy = dotDMat.getRowMatrix(2);
    dotDMat.mergeBelow(xyy).print();

    int strasA[] = {5,2,6,1,
                    0,6,2,0,
                    3,8,1,4,
                    1,8,5,6};
    int strasB[] = {7,5,8,0,
                    1,8,2,6,
                    9,4,3,8,
                    5,3,7,9};
    Propulsion::Matrix<int> SA(strasA,4,4);
    Propulsion::Matrix<int> SB(strasB,4,4);

    std::cout << "Strassen, O(N^3) first: " << std::endl;
    auto SCC = (SA * SB);


    std::cout << "SA: " << std::endl;
    SA.print();
    std::cout << "SB: " << std::endl;
    SB.print();
    std::cout << "Strassen: " << std::endl;
    SA.print();
    std::cout << " * " << std::endl;
    SB.print();

    SA.strassenMultiplication(SB);

    if(SA == SCC)
    {
        std::cout << "Yes!" << std::endl;
    }


    int strasC[] = {2,1,9,7};
    int strasD[] = {9,1,2,0};
    Propulsion::Matrix<int> SC(strasC,2,2);
    Propulsion::Matrix<int> SD(strasD,2,2);

    (SC * SD).print();


    SC.strassenMultiplication(SD);
    SC.print();


    Propulsion::Matrix<double> randomNumbers(15,4);
    randomNumbers.populateWithUniformDistribution(0.0, 2.0);
    randomNumbers.print();


    std::cout << "Onto Random Numbers: " << std::endl;
    Propulsion::Matrix<int> randomA(512*2,512*2);
    Propulsion::Matrix<int> randomB(512*2, 512*2);

    randomA.populateWithUniformDistribution(-1, 120);
    randomB.populateWithUniformDistribution(-1, 999);

    Propulsion::Matrix<int> removeA(6,6);
    removeA.populateWithUniformDistribution(0, 1000);
    std::cout << "Testing Removing Row" << std::endl;
    removeA.print();

    std::cout << "After Removing Row" << std::endl;
    auto removeC = removeA.removeRow(5);
    removeC.print();

    removeC = removeC.removeCol(5);
    removeC.print();


    if(randomA.equalTo(randomB))
    {
        std::cout << "Wow they're equal!" << std::endl;
    }
    else
    {
        std::cout << "Random A != Random B" << std::endl;
    }

    if(randomA == randomB)
    {
        std::cout << "Something went wrong!" << std::endl;

    }
    else
    {
        std::cout << "Equal Operator Says A != B" << std::endl;
    }

    if(randomA == randomA)
    {
        std::cout << "Operator says RandomA == RandomA!" << std::endl;
    }
    else
    {
        std::cout << "Something went wrong!" << std::endl;
    }

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    auto randomC = (randomA * randomB);
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    float milliseconds = (float)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() / 1000;

    std::cout << "O(n^3) Took: " << milliseconds <<  " milliseconds." << std::endl;


    std::chrono::high_resolution_clock::time_point start2 = std::chrono::high_resolution_clock::now();
    randomA.strassenMultiplication(randomB);
    std::chrono::high_resolution_clock::time_point end2 = std::chrono::high_resolution_clock::now();
    float milliseconds2 = (float)std::chrono::duration_cast<std::chrono::microseconds>(end2-start2).count() / 1000;

    std::cout << "O(n^2.873..) Took: " << milliseconds2 << " milliseconds." << std::endl;


    if(randomA == randomC)
    {
        std::cout << "It works!" << std::endl;
    }

    Propulsion::Matrix<int> tA(2048, 2048);
    Propulsion::Matrix<int> tB(2048, 2048);

    tA.populateWithUniformDistribution(0, 11232312);
    tB.populateWithUniformDistribution(0, 34524351);

    std::chrono::high_resolution_clock::time_point start3 = std::chrono::high_resolution_clock::now();
    tA.strassenMultiplication(tB);
    std::chrono::high_resolution_clock::time_point end3 = std::chrono::high_resolution_clock::now();
    float milliseconds3 = (float)std::chrono::duration_cast<std::chrono::microseconds>(end3-start3).count() / 1000;

    std::cout << "O(n^2.873..) With 7 Threads Took: " << milliseconds3 << " milliseconds." << std::endl;



    Propulsion::Matrix<int> zA(3,6);
    Propulsion::Matrix<int> zB(6,9);

    zA.populateWithUniformDistribution(0, 99);
    zB.populateWithUniformDistribution(0, 99);

    zA.print();
    zB.print();

    zA.strassenMultiplication(zB);
    zA.print();

    /*
    int SX[] = {4};
    int SY[] = {8};
    Propulsion::Matrix<int> StrasX(SX,1,1);
    Propulsion::Matrix<int> StrasY(SY,1,1);
    (StrasX * StrasY).print();*/


    Propulsion::Matrix<int> cA(2048*2,2048*2);
    Propulsion::Matrix<int> cB(2048*2,2048*2);
    Propulsion::Matrix<int> cC(2048*2,2048*2);


    cA.populateWithUniformDistribution(0, 99);
    cB.populateWithUniformDistribution(0, 99);

    cC = cA + cB;
    cA.add(cB);

    if(cC == cA)
    {
        std::cout << "Yes boi" << std::endl;
    }

    int diagonalA[]={8, 0, 0,
                     0, 3, 0,
                     0, 0, 1};
    int diagonalB[]={2,1,0,
                     0,3,9,
                     0,0,1};
    int diagonalC[]={2,0,0,
                     1,3,0,
                     0,2,1};

    Propulsion::Matrix<int> diagA(diagonalA, 3,3);
    Propulsion::Matrix<int> diagB(diagonalB, 3,3);
    Propulsion::Matrix<int> diagC(diagonalC, 3,3);

    if(diagA.isUpperTriangular() && diagA.isLowerTriangular())
    {
        std::cout << "DiagA is an upper Triangle and Lower Triangle!" << std::endl;
    }

    if(diagB.isUpperTriangular() && !diagB.isLowerTriangular())
    {
        std::cout << "DiagB is just an upper Triangle and not Lower!" << std::endl;
    }

    if(diagC.isLowerTriangular() && !diagB.isLowerTriangular())
    {
        std::cout << "DiagC is just a lower Triangle and not an Upper!" << std::endl;
    }

    /*
    Propulsion::ArtificialNeuralNetwork ANN(&in, &out);
    ANN.createSynapsesLayer(1);
    ANN.printInputLayer();
    ANN.printSynapsesLayer();

    ANN.trainModel();

    ANN.printOutputLayer();*/
}

bool compareMultiplicationOperations()
{
    unsigned rSz = 512;
    unsigned cSz = 512;

    // create matrices.
    Propulsion::Matrix<int> A(rSz, cSz);
    Propulsion::Matrix<int> Aa(rSz, cSz);
    Propulsion::Matrix<int> B(cSz, rSz);
    Propulsion::Matrix<int> Cstrassen(A.getRowSize(), B.getColSize());
    Propulsion::Matrix<int> Ccuda(A.getRowSize(), B.getColSize());


    // Populate Matrices.
    A.populateWithUniformDistribution(0.0,100.0);

    // Copy.
    Aa = A;
    B.populateWithUniformDistribution(0.0,100.0);

    /*
     * O(n^3)
     */

    /*
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    Aa.dot(B);
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    float milliseconds = (float)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() / 1000;

    std::cout << "O(n^3) Took: " << milliseconds <<  " milliseconds." << std::endl;
     */


    /*
     * Strassen with 7 threads.
     */
    std::chrono::high_resolution_clock::time_point start2 = std::chrono::high_resolution_clock::now();
    Cstrassen = A * B;
    std::chrono::high_resolution_clock::time_point end2 = std::chrono::high_resolution_clock::now();
    float milliseconds = (float)std::chrono::duration_cast<std::chrono::microseconds>(end2-start2).count() / 1000;

    std::cout << " Strassen Took: " << milliseconds <<  " milliseconds." << std::endl;

    /*
     * CUDA O(n^3)
     */
    Aa.cudaDotProduct(B, true);

    return Aa == Cstrassen;
}

void secondMatrixTests()
{
    Propulsion::Matrix<double> A(200, 200);
    Propulsion::Matrix<double>::randomRealDistribution(A, -2.0, 2.0);

    auto B = A;
    if(A == B)
    {
        printExceptionTestResults("(Equality Operator & Copy Constructor)", true, __FILE__, __LINE__);
        //std::cout << "Passed(Equality Operator & Copy Constructor)" << std::endl;

    }
    else
    {
        printExceptionTestResults("(Equality Operator or Copy Constructor)", false, __FILE__, __LINE__);
    }

    auto C = A * B;
    if(C == B)
    {
        printExceptionTestResults("(rValue and Copy Constructor)-> C should not equal B", false, __FILE__, __LINE__);
    }
    else if(C == A)
    {
        printExceptionTestResults("(rValue and Copy Constructor)-> C should not equal A", false, __FILE__, __LINE__);
    }
    else
    {
        printExceptionTestResults("(rValue and Copy Constructor)", true, __FILE__, __LINE__);
    }

    // Verify Addition works
    try{
        Propulsion::Matrix<double> addA(677, 888);
        Propulsion::Matrix<double> addB(687, 888);

        Propulsion::Matrix<double>::randomRealDistribution(addA, 1.0, 100.0);
        Propulsion::Matrix<double>::randomRealDistribution(addB, 1.0, 100.0);

        addA.add(addB);
    }
    catch (Propulsion::Matrix<double>::MatrixException &e)
    {
        printExceptionTestResults(std::string("(Add Exception throw)->Exception: ") + e.what(), true, __FILE__, __LINE__);
    }
    try{
        Propulsion::Matrix<double> addA(687, 888);
        Propulsion::Matrix<double> addB(687, 888);

        Propulsion::Matrix<double>::randomRealDistribution(addA, 1.0, 100.0);
        Propulsion::Matrix<double>::randomRealDistribution(addB, 1.0, 100.0);

        addA.add(addB);

        printExceptionTestResults(std::string("(Add Didn't Throw Exception on good data)"), true, __FILE__, __LINE__);
    }
    catch (Propulsion::Matrix<double>::MatrixException &e)
    {
        printExceptionTestResults(std::string("(Add Should not Throw Exception)->Exception: ") + e.what(), false, __FILE__, __LINE__);
    }

    /*
     * Verify Add Row Vector
     */
    try{
        Propulsion::Matrix<double> arvA(3, 3);
        Propulsion::Matrix<double> arvB(1,3);

        Propulsion::Matrix<double>::randomRealDistribution(arvA, -1.0, 1.0);
        Propulsion::Matrix<double>::randomRealDistribution(arvB, -1.0, 1.0);

        auto arvC = arvA.addRowVector(arvB);

        printExceptionTestResults("(addRowVector)", true, __FILE__, __LINE__);
    }
    catch (Propulsion::Matrix<double>::MatrixException &e)
    {
        printExceptionTestResults(std::string("(addRowVector)") + e.what(), false, __FILE__, __LINE__);
    }
    try{
        Propulsion::Matrix<double> arvA(3, 2);
        Propulsion::Matrix<double> arvB(1,3);

        Propulsion::Matrix<double>::randomRealDistribution(arvA, -1.0, 1.0);
        Propulsion::Matrix<double>::randomRealDistribution(arvB, -1.0, 1.0);

        auto arvC = arvA.addRowVector(arvB);

        printExceptionTestResults(std::string("(addRowVector should throw an exception)"), false, __FILE__, __LINE__);
    }
    catch (Propulsion::Matrix<double>::MatrixException &e)
    {
        printExceptionTestResults(std::string("Passed(addRowVector should throw an exception)->Exception: ") + e.what(), true, __FILE__, __LINE__);
    }

    /*
     * Verify addColVectors
     */
    try{
        Propulsion::Matrix<double> acvA(3, 3);
        Propulsion::Matrix<double> acvB(3,1);

        Propulsion::Matrix<double>::randomRealDistribution(acvA, -1.0, 1.0);
        Propulsion::Matrix<double>::randomRealDistribution(acvB, -1.0, 1.0);

        auto acvC = acvA.addColVector(acvB);

        printExceptionTestResults(std::string("(addColVector)"), true, __FILE__, __LINE__);
    }
    catch (Propulsion::Matrix<double>::MatrixException &e)
    {
        printExceptionTestResults(std::string("(addColVector should not throw an exception)->Exception: ") + e.what(), true, __FILE__, __LINE__);
    }
    try{
        Propulsion::Matrix<double> acvA(2, 3);
        Propulsion::Matrix<double> acvB(3,1);

        Propulsion::Matrix<double>::randomRealDistribution(acvA, -1.0, 1.0);
        Propulsion::Matrix<double>::randomRealDistribution(acvB, -1.0, 1.0);

        auto acvC = acvA.addColVector(acvB);

        printExceptionTestResults(std::string("Failed(addColVector should throw an exception)"), false, __FILE__, __LINE__);
    }
    catch (Propulsion::Matrix<double>::MatrixException &e)
    {
        printExceptionTestResults(std::string("(addColVector threw an exception)->Exception: ") + e.what(), true, __FILE__, __LINE__);
    }

    /*
     * Verify Subtract
     */
    try{
        Propulsion::Matrix<double> subA(4, 4);
        Propulsion::Matrix<double> subB(4,4);

        Propulsion::Matrix<double>::randomRealDistribution(subA, -1.0, 1.0);
        Propulsion::Matrix<double>::randomRealDistribution(subB, -1.0, 1.0);

        subA.subtract(subB);

        printExceptionTestResults(std::string("(subtract)"), true, __FILE__, __LINE__);

    }
    catch (Propulsion::Matrix<double>::MatrixException &e)
    {
        printExceptionTestResults(std::string("(subtract should not throw an exception)->Exception: ") + e.what(), false, __FILE__, __LINE__);
    }

    try{
        Propulsion::Matrix<double> subA(4, 123);
        Propulsion::Matrix<double> subB(123,4);

        Propulsion::Matrix<double>::randomRealDistribution(subA, -1.0, 1.0);
        Propulsion::Matrix<double>::randomRealDistribution(subB, -1.0, 1.0);

        subA.subtract(subB);
        printExceptionTestResults(std::string("(subtract should throw an exception)"), false, __FILE__, __LINE__);
    }
    catch (Propulsion::Matrix<double>::MatrixException &e)
    {
        printExceptionTestResults(std::string("(subtract should throw an exception)->Exception: ") + e.what(), true, __FILE__, __LINE__);
    }

    /*
     * Verify cudaMultiplyMatrices
     */
    try{
        Propulsion::Matrix<double> multA(4, 122);
        Propulsion::Matrix<double> multB(123,4);

        Propulsion::Matrix<double>::randomRealDistribution(multA, -1.0, 1.0);
        Propulsion::Matrix<double>::randomRealDistribution(multB, -1.0, 1.0);

        multA.cudaDotProduct(multB);

        printExceptionTestResults(std::string("Failed(cudaMultiplyMatrices should throw an exception)"), false, __FILE__, __LINE__);
    }
    catch (Propulsion::Matrix<double>::MatrixException &e)
    {
        printExceptionTestResults(std::string("Passed(cudaMultiplyMatrices should throw an exception)->Exception: ") + e.what(), true, __FILE__, __LINE__);
    }
    try{
        Propulsion::Matrix<double> multA(4, 123);
        Propulsion::Matrix<double> multB(123,4);

        Propulsion::Matrix<double>::randomRealDistribution(multA, -1.0, 1.0);
        Propulsion::Matrix<double>::randomRealDistribution(multB, -1.0, 1.0);

        multA.cudaDotProduct(multB);

        printExceptionTestResults(std::string("(cudaMultiplyMatrices)"), true, __FILE__, __LINE__);
    }
    catch (Propulsion::Matrix<double>::MatrixException &e)
    {
        printExceptionTestResults(std::string("(cudaMultiplyMatrices should not throw an exception)->Exception: ") + e.what(), false, __FILE__, __LINE__);
    }

    /*
     * Verify DotProduct
     */
    try{
        Propulsion::Matrix<double> multA(4, 122);
        Propulsion::Matrix<double> multB(123,4);

        Propulsion::Matrix<double>::randomRealDistribution(multA, -1.0, 1.0);
        Propulsion::Matrix<double>::randomRealDistribution(multB, -1.0, 1.0);

        multA.cudaDotProduct(multB);

        printExceptionTestResults(std::string("(dot should throw an exception)"), false, __FILE__, __LINE__);
    }
    catch (Propulsion::Matrix<double>::MatrixException &e)
    {
        printExceptionTestResults(std::string("(dot should throw an exception)->Exception: ") + e.what(), true, __FILE__, __LINE__);
    }
    try{
        Propulsion::Matrix<int> multA(206, 123);
        Propulsion::Matrix<int> multB(123,206);

        Propulsion::Matrix<int>::randomRealDistribution(multA, 0.0, 5.0);
        Propulsion::Matrix<int>::randomRealDistribution(multB, 0.0, 25.0);

        auto cudaA = multA;
        auto cudaB = multB;

        cudaA.cudaDotProduct(cudaB, true);
        multA.dot(multB, true);

        if(multA == cudaA)
        {
            printExceptionTestResults(std::string("(dot vs cudaDot equality)"), true, __FILE__, __LINE__);
        }
        else
        {
            printExceptionTestResults(std::string("(dot vs cudaDot equality)"), false, __FILE__, __LINE__);
        }
    }
    catch (Propulsion::Matrix<double>::MatrixException &e)
    {
        printExceptionTestResults(std::string("Failed(dot should not throw an exception)->Exception: ") + e.what(), false, __FILE__, __LINE__);
    }
    std::cout << "Testing Matrix Dot Product Speed..." << std::endl
            << "Results: " << std::endl;

    if(!compareMultiplicationOperations())
    {
        std::cout << "Dot Product Failed Equality Test" << std::endl;
    }
    else
    {

    }


    /*
     * Verify Schurs Product Works
     */
    try {
        Propulsion::Matrix<double> schursA(1000, 1000);
        Propulsion::Matrix<double> schursB(1000, 1001);

        schursA.schurProduct(schursB);
    }
    catch (Propulsion::Matrix<double>::MatrixException &e)
    {
        std::cout << "Passed(SchurProduct Exception throw)->Exception: " << e.what() << std::endl;
    }
}

void matrixMultiplicationTests()
{
    // Declare clock times.
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;

    // square matrix sizes to test speeds on
    unsigned SZ[] = {128, 256, 512, 1024, 2048, 4096, 8192, 16384};


    double cpuTimes[sizeof(SZ)];
    double strassenTimes[sizeof(SZ)];
    double cudaTimes[sizeof(SZ)];

    // Stats header for the user.
    std::cout << "Matrix Multiplication Performance Tests" << std::endl
              << "-----(DIMS)-----|-----CPU NAIVE----|---CPU STRASSEN---|----CUDA NAIVE----|" << std::endl;

    for(unsigned i = 0; i < sizeof(SZ) / sizeof(unsigned); i++)
    {
        // Create new A & B matrices with the SZxSZ as dims.
        Propulsion::Matrix<float> A(SZ[i], SZ[i]);
        Propulsion::Matrix<float> B(SZ[i], SZ[i]);

        // Populate with random values.
        Propulsion::Matrix<float>::randomRealDistribution(A, -1.0, 1.0);
        Propulsion::Matrix<float>::randomRealDistribution(B, -1.0, 1.0);

        Propulsion::Matrix<float> aCpuCopy = A;
        Propulsion::Matrix<float> aStrassenCopy = A;

        A.add(A + B);


        // Time the cpu dot product of B
        start = std::chrono::high_resolution_clock::now();
        if(SZ[i] < 8192) {
            aCpuCopy.dot(B, false);
            end = std::chrono::high_resolution_clock::now();
            cpuTimes[i] = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        }
        else
        {
            cpuTimes[i] = 0.0;
        }


        // Time the cpu on strassen multiplication
        start = std::chrono::high_resolution_clock::now();
        aStrassenCopy.strassenMultiplication(B);
        end = std::chrono::high_resolution_clock::now();
        strassenTimes[i] = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

        // Time the cuda dot product of B
        start = std::chrono::high_resolution_clock::now();
        A.cudaDotProduct(A + B, false);
        end = std::chrono::high_resolution_clock::now();
        cudaTimes[i] = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;


        std::ostringstream oStr;

        // Print Stats to the user:
        if(cpuTimes[i] != 0.0) {

            oStr << std::fixed << std::setprecision(2) << cpuTimes[i];
        }
        else
        {
            oStr << "...";
        }

        std::cout << std::setw(7) << std::right << std::to_string(SZ[i]) << " x " << std::setw(6) << std::left << std::to_string(SZ[i]) << "|" << std::right
                  << std::setprecision(2) << std::fixed
                  << std::setw(14) << oStr.str() << " ms." << "|"
                  << std::setw(14) << strassenTimes[i] << " ms." << "|"
                  << std::setw(14) << cudaTimes[i] << " ms." <<  "|" << std::endl;

    }

    /*
    std::cout << "Matrix Multiplication Performance Tests" << std::endl
            << "-----(DIMS)-----|---CPU NAIVE------|---CPU STRASSEN---|---CUDA NAIVE-----|" << std::endl;


    for(unsigned i = 0; i < sizeof(SZ)/sizeof(unsigned); i++)
    {
        std::cout << std::setw(7) << std::right << std::to_string(SZ[i]) << " x " << std::setw(6) << std::left << std::to_string(SZ[i]) << "|" << std::right
                    << std::setprecision(2) << std::fixed
                    << std::setw(14) << cpuTimes[i] << " ms." << "|"
                    << std::setw(14) << strassenTimes[i] << " ms." << "|"
                    << std::setw(14) << cudaTimes[i] << " ms." <<  "|" << std::endl;
    }*/

}

void aiClassifier()
{
    auto A = Propulsion::ArtificialNeuralNetwork();
    A.test();
}

int main()
{
    /*
    std::cout << std::endl << std::endl <<
              " 222222222222222         DDDDDDDDDDDDD\n" <<
              "2:::::::::::::::22       D::::::::::::DDD\n" <<
              "2::::::222222:::::2      D:::::::::::::::DD\n" <<
              "2222222     2:::::2      DDD:::::DDDDD:::::D\n" <<
              "            2:::::2        D:::::D    D:::::D\n" <<
              "            2:::::2        D:::::D     D:::::D\n" <<
              "         2222::::2         D:::::D     D:::::D\n" <<
              "    22222::::::22          D:::::D     D:::::D\n" <<
              "  22::::::::222            D:::::D     D:::::D\n" <<
              " 2:::::22222               D:::::D     D:::::D\n" <<
              "2:::::2                    D:::::D     D:::::D\n" <<
              "2:::::2                    D:::::D    D:::::D\n" <<
              "2:::::2       222222     DDD:::::DDDDD:::::D\n" <<
              "2::::::2222222:::::2     D:::::::::::::::DD\n" <<
              "2::::::::::::::::::2     D::::::::::::DDD\n" <<
              "22222222222222222222     DDDDDDDDDDDDD\n\n"; //        arrays....*/

    test_one_dimensional_array_operations();
    secondMatrixTests();



    Propulsion::Mandelbrot M(640,480);
    M.simulate();

    /*
    Propulsion::Matrix<double> R(1000,1000);
    Propulsion::Matrix<double>::randomRealDistribution(R, -1.0, 1.0);

    auto A = Propulsion::Matrix<double>::copy(R);

    if(A == R)
    {
        std::cout << "Wow" << std::endl;
    }*/

    matrixMultiplicationTests();



    return 0;
}