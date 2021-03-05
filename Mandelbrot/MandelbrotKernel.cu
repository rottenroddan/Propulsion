//
// Created by steve on 1/1/2021.
//

__global__ void mandelbrotCalculate(int *d_output, int *d_colorPicker, int wPixels, int hPixels,
                                    double leftBound, double rightBound, double topBound,
                                    double bottomBound, unsigned maxIterations)
{
    unsigned tRow = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned tCol = blockIdx.y * blockDim.y + threadIdx.y;

    if(tRow < hPixels && tCol < wPixels)
    {
        // Find the value our cuda thread is going to be from the graph.
        double realIncrementer = ( rightBound - leftBound ) / wPixels;
        double complexIncrementer = (topBound - bottomBound) / hPixels;
        double realXValue = leftBound + realIncrementer * tCol;
        double complexYValue = topBound - complexIncrementer * tRow;

        double zx = 0.0;
        double zy = 0.0;
        unsigned n = 0;
        while(zx * zx + zy * zy <= 4 && n < maxIterations)
        {
            double tempx = zx * zx - zy * zy + realXValue;
            zy = 2 * zx * zy + complexYValue;
            zx = tempx;
            n++;
        }

        if(n == maxIterations)
        {
            d_output[tRow * wPixels + tCol] = d_colorPicker[n-1];
        }
        else
        {
            d_output[tRow * wPixels + tCol] = d_colorPicker[n-1];
        }
    }
}


std::unique_ptr<Propulsion::Matrix<int>> Propulsion::Mandelbrot::calculateMandelCUDA(unsigned int wPixels, unsigned int hPixels, double leftBound, double rightBound, double topBound, double bottomBound, unsigned int maxIterations, std::shared_ptr<Propulsion::Matrix<int> > colorPicker)
{
    // Create a matrix with the dimensions of the window
    std::unique_ptr<Propulsion::Matrix<int>> Mandelset( new Matrix<int>(hPixels, wPixels));


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // Create device array pointers.
    int *dev_output, *dev_colorPicker;

    cudaMalloc((void**) &dev_output, wPixels * sizeof(int) * hPixels);
    cudaMalloc((void**) &dev_colorPicker, maxIterations * sizeof(int));

    cudaMemcpy(dev_colorPicker, colorPicker->getArray(), maxIterations * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block_dim(32,32);

    unsigned blocksX = std::ceil(( (double) hPixels) / ( (double) block_dim.x) );
    unsigned blocksY = std::ceil(( (double) wPixels) / ( (double) block_dim.y) );

    dim3 grid_dim(blocksX, blocksY);

    cudaEventRecord(start);
    mandelbrotCalculate<<<grid_dim, block_dim>>>(dev_output, dev_colorPicker, wPixels,
                                                 hPixels, leftBound, rightBound,
                                                 topBound, bottomBound, maxIterations);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    cudaMemcpy(Mandelset->getArray(), dev_output, wPixels * sizeof(int) * hPixels, cudaMemcpyDeviceToHost);


    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaFree(dev_output);
    cudaFree(dev_colorPicker);


    std::cout << std::left << std::setw(TIME_FORMAT) << " CUDA:  Mandelbrot Calculation took: " <<
              std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
              " ms." << std::setw(TIME_WIDTH) << std::endl;



    /*
     * END OF CUDA STUFF
     */
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    return Mandelset;
}