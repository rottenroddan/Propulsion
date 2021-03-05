//
// Created by steve on 12/18/2020.
//

#include "../Propulsion.cuh"
#include <d2d1.h>
#include <d2d1helper.h>
#include <dwrite.h>
#include <wincodec.h>

#include <memory>

#define MAX_ITER 5000

bool activeApp = false;

unsigned ITER_STEPS[] = {125, 125, 250, 250, 500, 500, 1000, 1000, 1500, 1500, 2000, 2000, 5000, 5000, 10000, 10000, 20000, 20000, 50000, 50000, 100000,100000, 200000, 200000, 500000, 500000, 1000000, 1000000};
unsigned STEP_SIZE = 28;


std::mutex mutexPainting;
std::atomic<bool> windowDestroyed;


Propulsion::Mandelbrot::Mandelbrot(unsigned int width, unsigned int height, double leftBound, double rightBound, double topBound, double bottomBound, double zoomFactor)
{
    // Set window width/height.
    this->windowWidthPixels = width;
    this->windowHeightPixels = height;

    // Set bounds for graphing.
    this->topBound = topBound;
    this->bottomBound = bottomBound;
    this->leftBound = leftBound;
    this->rightBound = rightBound;

    this->zoomFactor = zoomFactor;

    this->iterations = MAX_ITER;

    generateColorScheme(this->iterations);
    this->epoch = std::make_unique<Matrix<unsigned>>(ITER_STEPS , 1, STEP_SIZE);
}


void Propulsion::Mandelbrot::generateColorSchemeV2(unsigned int totalColors)
{
    this->colorPicker = std::unique_ptr<Matrix<int>>(new Matrix<int>(1,totalColors));

    int lastColor = 0x000000;

    // Colors from 0% -> 1.0%     D.Purple  Purple    Blu/Purp
    //std::vector<int> colorPalette{ 0x20006b, 0xd88aff, 0x4e00d6, 0x170099, 0x4f9bff, 0x00d6c1, 0x00d681, 0x17e300, 0xfffb00, 0xffdd00, 0xff8000, 0xff0800, 0xff5ca8, 0xe300d0, 0xbf19cf, 0x6d08cc, 0x140bb8, 0x0096ed, 0x00ede9, 0x0ecf7f, 0x0de00d, 0xffe208, 0xf56a00, 0xffffff};
    //std::vector<double> percents {      0.0,     .005,     .010,    0.015,    0.020,    0.030,    0.040,    0.055,    0.080,    0.100,    0.125,    0.150,    0.165,     .180,    0.210,     .235,     0.25,    0.275,    0.300,    0.320,    0.332,    0.350,    0.370, 1.0};

    std::vector<int> colorPalette{ 0x20006b, 0xd88aff, 0x4e00d6, 0x170099, 0x4f9bff, 0x00d6c1, 0x00d681, 0x17e300, 0xfffb00, 0xffdd00, 0xff8000, 0xff0800, 0xff5ca8, 0xe300d0, 0xbf19cf, 0x6d08cc, 0x140bb8, 0x0096ed, 0x00ede9, 0x0ecf7f, 0x0de00d, 0xffe208, 0xf56a00, 0xffabab, 0xffffff};
    std::vector<double> percents {      0.0,     .005,     .010,    0.015,    0.020,    0.030,    0.040,    0.055,    0.080,    0.100,    0.125,    0.150,    0.190,     .210,    0.240,     .275,     0.31,    0.420,    0.470,    0.520,    0.560,    0.620,    0.720,    0.800,      1.0};


    // For for loop condition initializer
    double percentOfSuccess = 0.0;

    unsigned currentColorItt = 0;

    // For accessing percents vector, firstControl is ith index while
    // second is i+1 index.
    double firstControl;
    double secondControl;

    // Int for storing the colors from colorPalette.
    int firstRed;
    int firstGreen;
    int firstBlue;

    int secondRed;
    int secondGreen;
    int secondBlue;

    for(unsigned i = 0; i < colorPalette.size() - 1; i++)
    {
        firstControl = percents[i];
        secondControl = percents[i+1];

        firstRed   = (int)((unsigned)colorPalette[i] >> 16);
        firstGreen = (int)((unsigned)(colorPalette[i] << 16) >> 24);
        firstBlue  = (int)((unsigned)(colorPalette[i] << 24) >> 24);

        secondRed   = (int)((unsigned)colorPalette[i+1] >> 16);
        secondGreen = (int)((unsigned)(colorPalette[i+1] << 16) >> 24);
        secondBlue  = (int)((unsigned)(colorPalette[i+1] << 24) >> 24);

        // Calculate the diff.
        int redDiff = secondRed - firstRed;
        int greenDiff = secondGreen - firstGreen;
        int blueDiff = secondBlue - firstBlue;

        // Unsigned values for storing calculated color.
        int red;
        int green;
        int blue;
        int finalColor;

        if(i == 0)
        {
            std::cout << "Red: " << std::hex << (int)firstRed << std::endl;
            std::cout << "Green: " << std::hex << (int)firstGreen << std::endl;
            std::cout << "Blue: " << std::hex << (int)firstBlue << std::endl;
        }

        while(percentOfSuccess < secondControl)
        {
            // Check if the value is decreasing from red to less red.
            /*
            if(redDiff < 0)
                red = (int) ( ( ( (1 - 0) * ( percentOfSuccess - secondControl) ) / (firstControl - secondControl) ) * ( std::abs(redDiff) ) + secondRed );
            else
                red = (int) (secondRed - (((1 - 0) * (percentOfSuccess - secondControl) ) / (firstControl - secondControl) ) * ( redDiff ) );

            // Green now...
            if(greenDiff < 0)
                green = (int) ( ( ( (1 - 0) * ( percentOfSuccess - secondControl) ) / (firstControl - secondControl) ) * ( std::abs(greenDiff) ) + secondGreen );
            else
                green = (int) (secondGreen - (((1 - 0) * (percentOfSuccess - secondControl) ) / (firstControl - secondControl) ) * ( greenDiff ) );

            // Lastly blue...
            if(blueDiff < 0)
                blue = (int) ( ( ( (1 - 0) * ( percentOfSuccess - secondControl) ) / (firstControl - secondControl) ) * ( std::abs(blueDiff) ) + secondBlue );
            else
                blue = (int) (secondBlue - (((1 - 0) * (percentOfSuccess - secondControl) ) / (firstControl - secondControl) ) * ( blueDiff ) );
            */
            if(redDiff < 0)
                red = (int) ((percentOfSuccess - secondControl) / (firstControl - secondControl) * (std::abs(redDiff)) + secondRed);
            else
                red = (int) (secondRed - (percentOfSuccess - secondControl) / (firstControl - secondControl) * (std::abs(redDiff)));

            if(greenDiff < 0)
                green = (int) ((percentOfSuccess - secondControl) / (firstControl - secondControl) * (std::abs(greenDiff)) + secondGreen);
            else
                green = (int) (secondGreen - (percentOfSuccess - secondControl) / (firstControl - secondControl) * (std::abs(greenDiff)));

            if(blueDiff < 0)
                blue = (int) ((percentOfSuccess - secondControl) / (firstControl - secondControl) * (std::abs(blueDiff)) + secondBlue);
            else
                blue = (int) (secondBlue - (percentOfSuccess - secondControl) / (firstControl - secondControl) * (std::abs(blueDiff)));

            finalColor = 0x000000;
            finalColor += red << 16;
            finalColor += green << 8;
            finalColor += blue;
            this->colorPicker->at(currentColorItt) = finalColor;

            currentColorItt++;
            percentOfSuccess =  (double)currentColorItt / (double)totalColors;
        }
    }

    this->colorPicker->at(totalColors-1) = lastColor;
}


void Propulsion::Mandelbrot::generateColorScheme(unsigned totalColors)
{
    const double topControl = 1.0;
    const double firstControl = .35;
    const double secondControl = .19;
    const double thirdControl = .12;
    const double fourthControl = .07;
    const double fifthControl = .03;
    const double sixthControl = 0.00;

    // First Color gradient.
    const int firstControlFirstRed      = 0xff;
    const int firstControlFirstGreen    = 0xff;
    const int firstControlFirstBlue     = 0xff;
    const int firstControlSecondRed     = 0xc2;
    const int firstControlSecondGreen   = 0x00;
    const int firstControlSecondBlue    = 0x00;


    // Second Color gradient.
    //const int secondControlFirstRed     = 0x26;
    //const int secondControlFirstGreen   = 0x02;
    //const int secondControlFirstBlue    = 0x4f;

    const int secondControlFirstRed     = 0xc2;
    const int secondControlFirstGreen   = 0x00;
    const int secondControlFirstBlue    = 0x00;

    const int secondControlSecondRed    = 0x56;
    const int secondControlSecondGreen  = 0x00;
    const int secondControlSecondBlue   = 0xbf;
    //const int secondControlSecondRed    = 0x9c;
    //const int secondControlSecondGreen  = 0x00;
    //const int secondControlSecondBlue   = 0xbf;


    // Third Color gradient.
    //const int thirdControlFirstRed      = 0x03;
    //const int thirdControlFirstGreen    = 0x00;
    //const int thirdControlFirstBlue     = 0xcf;

    const int thirdControlFirstRed      = 0x56;
    const int thirdControlFirstGreen    = 0x00;
    const int thirdControlFirstBlue     = 0xbf;

    const int thirdControlSecondRed     = 0x00;
    const int thirdControlSecondGreen   = 0xac;
    const int thirdControlSecondBlue    = 0xcf;


    // Fourth color gradient
    //const int fourthControlFirstRed     = 0x00;
    //const int fourthControlFirstGreen   = 0xe8;
    //const int fourthControlFirstBlue    = 0xba;

    const int fourthControlFirstRed     = 0x00;
    const int fourthControlFirstGreen   = 0xac;
    const int fourthControlFirstBlue    = 0xcf;

    const int fourthControlSecondRed    = 0xe8;
    const int fourthControlSecondGreen  = 0xdc;
    const int fourthControlSecondBlue   = 0x00;


    // Fifth color gradient
    //const int fifthControlFirstRed      = 0xfa;
    //const int fifthControlFirstGreen    = 0x7d;
    //const int fifthControlFirstBlue     = 0x00;

    const int fifthControlFirstRed      = 0xe8;
    const int fifthControlFirstGreen    = 0xdc;
    const int fifthControlFirstBlue     = 0x00;

    const int fifthControlSecondRed     = 0xfa;
    const int fifthControlSecondGreen   = 0x0c;
    const int fifthControlSecondBlue    = 0x00;


    //const int sixthControlFirstRed      = 0xfa;
    //const int sixthControlFirstGreen    = 0x00;
    //const int sixthControlFirstBlue     = 0x00;

    const int sixthControlFirstRed      = 0xfa;
    const int sixthControlFirstGreen    = 0x0c;
    const int sixthControlFirstBlue     = 0x00;

    const int sixthControlSecondRed     = 0x67;
    const int sixthControlSecondGreen   = 0x00;
    const int sixthControlSecondBlue    = 0x8a;

    this->colorPicker = std::unique_ptr<Matrix<int>>(new Matrix<int>(1,totalColors));

    int red = 0xff;
    int green = 0xff;
    int blue = 0xff;
    int color = 0xffffff;
    int redDiff = 0x00;
    int greenDiff = 0x00;
    int blueDiff = 0x00;

    // Starting from unstable to stable.
    for(unsigned i = 0; i < this->colorPicker->getTotalSize() - 1; i++)
    {
        double percentOfSuccess = (double)i / totalColors;
        red = 0x00;
        green = 0x00;
        blue = 0x00;
        redDiff = 0x00;
        greenDiff = 0x00;
        blueDiff = 0x00;


        // If we're on the edge cases of stable to unstable.
        if(percentOfSuccess >= firstControl)
        {
            /*
            red = (int) (0xc2 + 0xff * percentOfSuccess * percentOfSuccess);
            //green = (int) (0x00 + 0xff * percentOfSuccess * percentOfSuccess);
            //blue = (int) (0x00 + 0xff * percentOfSuccess * percentOfSuccess);

            // (b - a) * (x - min)
            // ------------------- + a
            //      (max - min)
            green = (int) ( ( ( percentOfSuccess - firstControl ) / ( topControl - firstControl ) ) * 0xff );
            blue = (int) ( ( ( percentOfSuccess - firstControl ) / ( topControl - firstControl ) ) * 0xff );
            if(red > 0xff)
            {
                red = 0xff;
            }
            if(blue > 0xff)
            {
                blue = 0xff;
            }
            if(green > 0xff)
            {
                green = 0xff;
            }*/

            redDiff = firstControlSecondRed - firstControlFirstRed;
            greenDiff = firstControlSecondGreen - firstControlFirstGreen;
            blueDiff = firstControlSecondBlue - firstControlFirstBlue;


            // If the from first red to second red is a increase in pixel value, then we need to ascend
            if(redDiff < 0)
            {
                red = (int) ( ( ( (1 - 0) * ( percentOfSuccess - firstControl) ) / (topControl - firstControl) ) * ( std::abs(redDiff) ) + firstControlSecondRed );
            }
                // Else we descend
            else
            {
                red = (int) (firstControlSecondRed - (((1 - 0) * (percentOfSuccess - firstControl) ) / (topControl - firstControl) ) * ( redDiff ) );
            }

            // Calculate green color now
            if(greenDiff < 0)
            {
                green = (int) ( ( ( (1 - 0) * ( percentOfSuccess - firstControl) ) / (topControl - firstControl) ) * ( std::abs(greenDiff) ) + firstControlSecondGreen );
            }
            else
            {
                green = (int) (firstControlSecondGreen - (((1 - 0) * (percentOfSuccess - firstControl) ) / (topControl - firstControl) ) * ( greenDiff ) );
            }

            if(blueDiff < 0)
            {
                blue = (int) ( ( ( (1 - 0) * ( percentOfSuccess - firstControl) ) / (topControl - firstControl) ) * ( std::abs(blueDiff) ) + firstControlSecondBlue );
            }
            else
            {
                blue = (int) (firstControlSecondBlue - (((1 - 0) * (percentOfSuccess - firstControl) ) / (topControl - firstControl) ) * ( blueDiff ) );
            }

        }
        else if(percentOfSuccess > secondControl)
        {
            /*
            red = 0x56;
            green = 0x00;
            blue = 0xbf;*/

            // start from
            //red = 0x9c;
            //green = 0x00;
            //blue = 0xbf;

            redDiff = secondControlSecondRed - secondControlFirstRed;
            greenDiff = secondControlSecondGreen - secondControlFirstGreen;
            blueDiff = secondControlSecondBlue - secondControlFirstBlue;


            // If the from first red to second red is a increase in pixel value, then we need to ascend
            if(redDiff < 0)
            {
                red = (int) ( ( ( (1 - 0) * ( percentOfSuccess - secondControl) ) / (firstControl - secondControl) ) * ( std::abs(redDiff) ) + secondControlSecondRed );
            }
            // Else we descend
            else
            {
                red = (int) (secondControlSecondRed - (((1 - 0) * (percentOfSuccess - secondControl) ) / (firstControl - secondControl) ) * ( redDiff ) );
            }

            // Calculate green color now
            if(greenDiff < 0)
            {
                green = (int) ( ( ( (1 - 0) * ( percentOfSuccess - secondControl) ) / (firstControl - secondControl) ) * ( std::abs(greenDiff) ) + secondControlSecondGreen );
            }
            else
            {
                green = (int) (secondControlSecondGreen - (((1 - 0) * (percentOfSuccess - secondControl) ) / (firstControl - secondControl) ) * ( greenDiff ) );
            }

            // Calculate blue color now
            if(blueDiff < 0)
            {
                blue = (int) ( ( ( (1 - 0) * ( percentOfSuccess - secondControl) ) / (firstControl - secondControl) ) * ( std::abs(blueDiff) ) + secondControlSecondBlue );
            }
            else
            {
                blue = (int) (secondControlSecondBlue - (((1 - 0) * (percentOfSuccess - secondControl) ) / (firstControl - secondControl) ) * ( blueDiff ) );
            }


        }
        else if(percentOfSuccess > thirdControl)
        {
            redDiff = thirdControlSecondRed - thirdControlFirstRed;
            greenDiff = thirdControlSecondGreen - thirdControlFirstGreen;
            blueDiff = thirdControlSecondBlue - thirdControlFirstBlue;


            // If the from first red to third red is a increase in pixel value, then we need to ascend
            if(redDiff < 0)
            {
                red = (int) ( ( ( (1 - 0) * ( percentOfSuccess - thirdControl) ) / (secondControl - thirdControl) ) * ( std::abs(redDiff) ) + thirdControlSecondRed );
            }
                // Else we descend
            else
            {
                red = (int) (thirdControlSecondRed - (((1 - 0) * (percentOfSuccess - thirdControl) ) / (secondControl - thirdControl) ) * ( redDiff ) );
            }

            // Calculate green color now
            if(greenDiff < 0)
            {
                green = (int) ( ( ( (1 - 0) * ( percentOfSuccess - thirdControl) ) / (secondControl - thirdControl) ) * ( std::abs(greenDiff) ) + thirdControlSecondGreen );
            }
            else
            {
                green = (int) (thirdControlSecondGreen - (((1 - 0) * (percentOfSuccess - thirdControl) ) / (secondControl - thirdControl) ) * ( greenDiff ) );
            }

            // Calculate blue color now
            if(blueDiff < 0)
            {
                blue = (int) ( ( ( (1 - 0) * ( percentOfSuccess - thirdControl) ) / (secondControl - thirdControl) ) * ( std::abs(blueDiff) ) + thirdControlSecondBlue );
            }
            else
            {
                blue = (int) (thirdControlSecondBlue - (((1 - 0) * (percentOfSuccess - thirdControl) ) / (secondControl - thirdControl) ) * ( blueDiff ) );
            }
        }
        else if(percentOfSuccess > fourthControl)
        {
            redDiff = fourthControlSecondRed - fourthControlFirstRed;
            greenDiff = fourthControlSecondGreen - fourthControlFirstGreen;
            blueDiff = fourthControlSecondBlue - fourthControlFirstBlue;


            // If the from first red to third red is a increase in pixel value, then we need to ascend
            if(redDiff < 0)
            {
                red = (int) ( ( ( (1 - 0) * ( percentOfSuccess - fourthControl) ) / (thirdControl - fourthControl) ) * ( std::abs(redDiff) ) + fourthControlSecondRed );
            }
                // Else we descend
            else
            {
                red = (int) (fourthControlSecondRed - (((1 - 0) * (percentOfSuccess - fourthControl) ) / (thirdControl - fourthControl) ) * ( redDiff ) );
            }

            // Calculate green color now
            if(greenDiff < 0)
            {
                green = (int) ( ( ( (1 - 0) * ( percentOfSuccess - fourthControl) ) / (thirdControl - fourthControl) ) * ( std::abs(greenDiff) ) + fourthControlSecondGreen );
            }
            else
            {
                green = (int) (fourthControlSecondGreen - (((1 - 0) * (percentOfSuccess - fourthControl) ) / (thirdControl - fourthControl) ) * ( greenDiff ) );
            }

            // Calculate blue color now
            if(blueDiff < 0)
            {
                blue = (int) ( ( ( (1 - 0) * ( percentOfSuccess - fourthControl) ) / (thirdControl - fourthControl) ) * ( std::abs(blueDiff) ) + fourthControlSecondBlue );
            }
            else
            {
                blue = (int) (fourthControlSecondBlue - (((1 - 0) * (percentOfSuccess - fourthControl) ) / (thirdControl - fourthControl) ) * ( blueDiff ) );
            }
        }

        else if(percentOfSuccess > fifthControl)
        {

            redDiff = fifthControlSecondRed - fifthControlFirstRed;
            greenDiff = fifthControlSecondGreen - fifthControlFirstGreen;
            blueDiff = fifthControlSecondBlue - fifthControlFirstBlue;


            // If the from first red to third red is a increase in pixel value, then we need to ascend
            if(redDiff < 0)
            {
                red = (int) ( ( ( (1 - 0) * ( percentOfSuccess - fifthControl) ) / (fourthControl - fifthControl) ) * ( std::abs(redDiff) ) + fifthControlSecondRed );
            }
                // Else we descend
            else
            {
                red = (int) (fifthControlSecondRed - (((1 - 0) * (percentOfSuccess - fifthControl) ) / (fourthControl - fifthControl) ) * ( redDiff ) );
            }

            // Calculate green color now
            if(greenDiff < 0)
            {
                green = (int) ( ( ( (1 - 0) * ( percentOfSuccess - fifthControl) ) / (fourthControl - fifthControl) ) * ( std::abs(greenDiff) ) + fifthControlSecondGreen );
            }
            else
            {
                green = (int) (fifthControlSecondGreen - (((1 - 0) * (percentOfSuccess - fifthControl) ) / (fourthControl - fifthControl) ) * ( greenDiff ) );
            }

            // Calculate blue color now
            if(blueDiff < 0)
            {
                blue = (int) ( ( ( (1 - 0) * ( percentOfSuccess - fifthControl) ) / (fourthControl - fifthControl) ) * ( std::abs(blueDiff) ) + fifthControlSecondBlue );
            }
            else
            {
                blue = (int) (fifthControlSecondBlue - (((1 - 0) * (percentOfSuccess - fifthControl) ) / (fourthControl - fifthControl) ) * ( blueDiff ) );
            }

        }
        else
        {
            redDiff = sixthControlSecondRed - sixthControlFirstRed;
            greenDiff = sixthControlSecondGreen - sixthControlFirstGreen;
            blueDiff = sixthControlSecondBlue - sixthControlFirstBlue;


            // If the from first red to third red is a increase in pixel value, then we need to ascend
            if(redDiff < 0)
            {
                red = (int) ( ( ( (1 - 0) * ( percentOfSuccess - 0) ) / (fifthControl - sixthControl) ) * ( std::abs(redDiff) ) + sixthControlSecondRed );
            }
                // Else we descend
            else
            {
                red = (int) (sixthControlSecondRed - (((1 - 0) * (percentOfSuccess - sixthControl) ) / (fifthControl - sixthControl) ) * ( redDiff ) );
            }

            // Calculate green color now
            if(greenDiff < 0)
            {
                green = (int) ( ( ( (1 - 0) * ( percentOfSuccess - sixthControl) ) / (fifthControl - sixthControl) ) * ( std::abs(greenDiff) ) + sixthControlSecondGreen );
            }
            else
            {
                green = (int) (sixthControlSecondGreen - (((1 - 0) * (percentOfSuccess - sixthControl) ) / (fifthControl - sixthControl) ) * ( greenDiff ) );
            }

            // Calculate blue color now
            if(blueDiff < 0)
            {
                blue = (int) ( ( ( (1 - 0) * ( percentOfSuccess - sixthControl) ) / (fifthControl - sixthControl) ) * ( std::abs(blueDiff) ) + sixthControlSecondBlue );
            }
            else
            {
                blue = (int) (sixthControlSecondBlue - (((1 - 0) * (percentOfSuccess - sixthControl) ) / (fifthControl - sixthControl) ) * ( blueDiff ) );
            }
        }

        color = 0x000000;
        color += red << 16;
        color += green << 8;
        color += blue;

        this->colorPicker->at(i) = color;
    }

    // this is stable color for the max iterations, therefore we make black.
    this->colorPicker->at(totalColors - 1) = 0x000000;

}

void Propulsion::Mandelbrot::paintWindow()
{
    //mutexPainting.lock();

    //std::cout << "Starting Paint" << std::endl;
    // Check if window handle exists. If not return and error message.
    if(this->hwnd == nullptr || windowDestroyed)
    {
        std::cout << "Window closed!" << std::endl;
        //mutexPainting.unlock();
        return;
    }

    // Get client area size. That way we know what to draw.
    RECT windowRect;
    RECT clientRect;
    GetWindowRect(this->hwnd, &windowRect);
    GetClientRect(this->hwnd, &clientRect);



    // Check if window is open.
    if(clientRect.bottom > 0 && clientRect.right > 0)
    {
        // Check if we need to recalculate Mandel to fit window.
        if(clientRect.right != this->clientWidthPixels || clientRect.bottom != this->clientHeightPixels || this->redraw)
        {
            // Set new width and height for window.
            this->windowWidthPixels     = windowRect.right;
            this->windowHeightPixels    = windowRect.bottom;

            // Set new width and height for client area.
            this->clientWidthPixels     = clientRect.right;
            this->clientHeightPixels    = clientRect.bottom;


            // Start clock for mandel calculation
            std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();


            // Generate new color scheme as iterations have gone up!
            generateColorSchemeV2(this->iterations);

            /*
            int colorArray[] = {};

            Matrix<int> ()

            generateColorSchemeV2(this->iterations, colors, colorPercentageBounds);*/

            // Calculate Mandel
            /*
            this->Mandel = calculateMandelSingleThreaded(this->clientWidthPixels, this->clientHeightPixels,
                                                         this->leftBound, this->rightBound, this->topBound,
                                                         this->bottomBound, this->iterations,
                                                         this->colorPicker);
            /*
            this->Mandel = calculateMandelAVX256(this->clientWidthPixels, this->clientHeightPixels,
                                                 this->leftBound, this->rightBound, this->topBound,
                                                 this->bottomBound, this->iterations,this->colorPicker);*/


            this->Mandel = calculateMandelCUDA(this->clientWidthPixels, this->clientHeightPixels,
                                               this->leftBound, this->rightBound, this->topBound,
                                               this->bottomBound, this->iterations,this->colorPicker);


            /*
            this->Mandel = calculateMandelMultiThreaded(16,this->clientWidthPixels, this->clientHeightPixels,
                                               this->leftBound, this->rightBound, this->topBound,
                                               this->bottomBound, this->iterations,this->colorPicker); */



            // Calculate total copy constructor time + process time.
            std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
            float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

            // Print about the total time for calculation and copy constructor!
            std::cout << std::left << std::setw(TIME_FORMAT) << " HOST:  Mandel Copy + Calculate Time: " <<
                      std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                      " ms." <<  std::endl;

            this->redraw = false;
        }

        if(Mandel->getColSize() != clientRect.right || Mandel->getRowSize() != clientRect.bottom)
        {
            std::cout << "You should never see this!" << std::endl;
        }

        InvalidateRect(this->hwnd, &clientRect, false);

        // Get Device Context handle.
        PAINTSTRUCT ps;
        //HDC hdc = GetDC(this->hwnd);
        HDC hdc = BeginPaint(this->hwnd, &ps);



        //std::cout << clientRect.top << " + " << clientRect.bottom << " : " << clientRect.left << "+" <<  clientRect.right << std::endl;

        // Create bitmap that way we can draw all at once instead of setpixel.
        HBITMAP map = CreateBitmap(this->clientWidthPixels, this->clientHeightPixels, 1, 8*4, (void*) Mandel->getArray());

        HDC src = CreateCompatibleDC(hdc);

        SelectObject(src, map);

        BitBlt(hdc, 0, 0, this->clientWidthPixels, this->clientHeightPixels, src, 0, 0, SRCCOPY);


        /*


        // Try and lock mutex. If you can't we draw old.
        if(this->mandelMutex.try_lock() )
        {
            SetWindowPos(hwnd, 0, 0, 0, this->windowWidthPixels, this->windowHeightPixels ,  SWP_NOMOVE);
            //AdjustWindowRect((LPRECT) &rect, WS_OVERLAPPEDWINDOW, false);



            // Unlock so next thread can use it.
            mandelMutex.unlock();
        }
        else if(this->lastMandel != nullptr)
        {

        }
        else
        {
            std::cout << "Couldn't draw!" << std::endl;
        }

        */

        // Release the hdc
        EndPaint(hwnd, &ps);

        DeleteObject(map);
        DeleteDC(src);
        ReleaseDC(this->hwnd,hdc);
        DeleteDC(hdc);
    }

    //mutexPainting.unlock();

}

void PaintWindow( HWND hwnd , unsigned pixelWidth, unsigned pixelHeight)
{
}




LRESULT CALLBACK WndProc (HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    RECT rect;
    GetClientRect(hwnd, &rect);


    switch(msg)
    {
    case WM_KEYDOWN:
        if(wParam != VK_ESCAPE)
        {
            // If 'Q'
            if(wParam == 0x51)
            {
                //PaintWindow(hwnd, rect.right, rect.bottom );
            }
            else
            {

            }
            break;
        }
        else
        {
            DestroyWindow(hwnd);
            break;
        }
    case WM_CLOSE:
        DestroyWindow(hwnd);

        break;
    case WM_DESTROY:
        //mutexPainting.lock();
        std::cout << "Closing" << std::endl;
        PostQuitMessage(0);
        windowDestroyed = true;
        //mutexPainting.unlock();
        return 0;
    case WM_ACTIVATEAPP:
        if(activeApp != true)
        {
            activeApp = true;
            //PaintWindow(hwnd, rect.right, rect.bottom);
        }
        break;

    case WM_PAINT:
        return 0;
    default:
        //std::cout << "Itter" <<  ++i<< std::endl;
        break;
    }
    return DefWindowProc( hwnd, msg, wParam, lParam );
}




void Propulsion::Mandelbrot::simulate()
{
    // Register window class

    this->windowClass = new WNDCLASSA
    {
        0, WndProc, 0, 0, 0,
        LoadIcon( nullptr, IDI_APPLICATION ),
        LoadCursor( nullptr, IDC_ARROW ),
        (HBRUSH )GetStockObject(WHITE_BRUSH), // background color == black
        nullptr, // no menu
        "Mandelbrot"
    };

    ATOM wClass = RegisterClassA( this->windowClass );
    if (!wClass)
    {
        fprintf( stderr, "%s\n", "Couldn’t create Window Class" );
        return;
    }

    // Create the window
    this->hwnd = CreateWindowA(
            MAKEINTATOM( wClass ),
            "Mandelbrot",     // window title
            WS_OVERLAPPEDWINDOW, // title bar, thick borders, etc.
            CW_USEDEFAULT, CW_USEDEFAULT, 640, 480,
            NULL, // no parent window
            NULL, // no menu
            GetModuleHandle( NULL ),  // EXE's HINSTANCE
            NULL  // no magic user data
    );
    if (!hwnd)
    {
        fprintf( stderr, "%ld\n", GetLastError() );
        fprintf( stderr, "%s\n", "Failed to create Window" );
        return;
    }

    // Make window visible
    ShowWindow( hwnd, SW_SHOWNORMAL );

    // set window destroyed to false!
    windowDestroyed = false;


    // Start clock for mandel calculation
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();


    // Event loop
    MSG msg;

    bool lookingForInput = true;

    while (GetMessage( &msg, NULL, 0, 0 ) > 0)
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);

        // If Q button is pressed, do something special!
        if (GetKeyState('Q') & 0x8000 && GetActiveWindow() == hwnd && lookingForInput) {
            if(this->currentEpochSelection < this->epoch->getColSize() - 1)
            {
                currentEpochSelection++;
                std::cout << "Setting Iterations to: " << this->epoch->at(this->currentEpochSelection)<< std::endl;
                this->redraw = true;
                this->iterations = this->epoch->at(this->currentEpochSelection);

                lookingForInput = false;
            }
        }
        else if(GetKeyState('E') & 0x8000 && GetActiveWindow() == hwnd && lookingForInput) {
            if(this->currentEpochSelection > 0)
            {
                currentEpochSelection--;
                std::cout << "Setting Iterations to: " << this->epoch->at(this->currentEpochSelection)<< std::endl;
                this->redraw = true;
                this->iterations = this->epoch->at(this->currentEpochSelection);

                lookingForInput = false;
            }

        }
        else if(GetKeyState('T') & 0x8000 && GetActiveWindow() == hwnd && lookingForInput) {
            unsigned desiredIterations;
            std::cout << "Enter Desired Iterations here: ";
            std::cin >> desiredIterations;
            if(desiredIterations != 0) {
                this->redraw = true;
                this->iterations = desiredIterations;

                std::cout << "More stats: " << std::endl
                          << "Iterations: " << this->iterations << std::endl;
            }
            else
            {
                std::cout << "User Provided zero as input, just ignore." << std::endl;
            }
        }

        // Check if we can draw at 100FPS.
        if ((float) (std::chrono::high_resolution_clock::now() - start).count() / 1000 > 10) {
            // Check if left mouse button is pressed, if so and active window, then we ignore the paint as it may be a resize!
            if (GetAsyncKeyState(VK_LBUTTON) & 0x8000 && GetActiveWindow() == hwnd) {

                }
            else {
                paintWindow();
                lookingForInput = true;
            }
            // Reset timer from last frame!
            start = std::chrono::high_resolution_clock::now();
        }

        // Get the user message, as in like a paticular button to create
        // functionality, like zoom.
        switch(msg.message)
        {
            // Check if Mousewheel,if it is, then find if zooming in/out.
            case WM_MOUSEWHEEL:
                // Check if mouse wheel up
                if(GET_WHEEL_DELTA_WPARAM(msg.wParam) > 0)
                {
                    zoomInOnCursor();
                }
                // of if down
                else
                {
                    zoomOutOnCursor();
                }
                break;
            default:
                break;

        }



    }

    std::cout << "See ya: " << msg.wParam << std::endl;
}

std::unique_ptr<Propulsion::Matrix<int>> Propulsion::Mandelbrot::calculateMandelSingleThreaded(unsigned int wPixels, unsigned int hPixels, double leftBound, double rightBound, double topBound, double bottomBound, unsigned maxIterations, std::shared_ptr< Propulsion::Matrix< int>> colorPicker)
{
    // Start clock for mandel calculation
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    // Create a matrix with the dimensions of the window
    std::unique_ptr<Propulsion::Matrix<int>> Mandelset( new Matrix<int>(hPixels, wPixels));


    // Real line is the horizontal line
    double realIncrementer = ( rightBound - leftBound ) / wPixels;
    double complexIncrementer = (topBound - bottomBound) / hPixels;

    for(unsigned i = 0; i < Mandelset->getRowSize(); i++)
    {
        double complexYValue = topBound - complexIncrementer*i;
        for(unsigned j = 0; j < Mandelset->getColSize(); j++)
        {
            // The current values we are calculating. Scaled from the current pixel.
            double realXValue = leftBound + realIncrementer*j;

            double zx = 0;
            double zy = 0;

            unsigned n = 0;
            while(zx * zx + zy * zy <= 4 && n < maxIterations)
            {
                double tempx = zx * zx - zy * zy + realXValue;
                zy = 2 * zx * zy + complexYValue;
                zx = tempx;
                n += 1;
            }

            // find out
            if(n == maxIterations)
            {
                // Set to Black
                Mandelset->at(i,j) = colorPicker->at(n-1);
            }
            else
            {
                // Set to White
                Mandelset->at(i,j) = colorPicker->at(n);
            }
        }
    }

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

    std::cout << std::left << std::setw(TIME_FORMAT) << " HOST:  Mandel Calculate Time: " <<
              std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
              " ms." <<  std::endl;

    /*
    std::cout << "Width: " << wPixels << std::endl;
    std::cout << "Height: " << hPixels << std::endl;
    std::cout << "LeftRange: " << leftBound << std::endl;
    std::cout << "RightRange: " << rightBound << std::endl;*/

    return Mandelset;
}


std::unique_ptr<Propulsion::Matrix<int>> Propulsion::Mandelbrot::calculateMandelAVX256(unsigned int wPixels, unsigned int hPixels, double leftBound, double rightBound, double topBound, double bottomBound, unsigned maxIterations, std::shared_ptr< Propulsion::Matrix< int>> colorPicker)
{
    // Start clock for mandel calculation
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    // Create a matrix with the dimensions of the window
    std::unique_ptr<Propulsion::Matrix<int>> Mandelset( new Matrix<int>(hPixels, wPixels));


    // Real line is the horizontal line
    double realIncrementer = ( rightBound - leftBound ) / wPixels;
    double complexIncrementer = (topBound - bottomBound) / hPixels;

    // Needed for second for loop for any potential pixels outside of factor 4 for double intrin.
    unsigned i, j;
    unsigned boundedRange = Mandelset->getColSize() - (Mandelset->getColSize() % (unsigned)((AVX256BYTES) / (sizeof(double))));

    __m256d _zr, _zi, _cr, _ci, _a, _b, _zr2, _zi2, _two,
            _four, _mask1, _leftBound, _topBound,
            _realIncrementer, _complexIncrementer,
            _i, _j;
    __m256i _n, _maxIterations, _mask2, _c, _one;

    // Below are constants for the SIMD registers to be used.
    _one = _mm256_set1_epi64x(1);
    _two = _mm256_set1_pd(2.0);
    _four = _mm256_set1_pd(4.0);
    _topBound = _mm256_set1_pd(topBound);
    _leftBound = _mm256_set1_pd(leftBound);

    _maxIterations = _mm256_set1_epi64x(maxIterations);
    _realIncrementer = _mm256_set1_pd(realIncrementer);
    _complexIncrementer = _mm256_set1_pd(complexIncrementer);

    for(i = 0; i < Mandelset->getRowSize(); i++)
    {
        // _i = i
        // | 0 | 0 | 0 | 0 |
        //   ^   ^   ^   ^
        //   +---+---+---+
        //   |
        //   i -> 1
        // -----------------
        // | 1 | 1 | 1 | 1 |
        _i = _mm256_set1_pd((double)i);

        // ci = leftBound + realIncrementer*j;
        _ci = _mm256_sub_pd(_topBound, _mm256_mul_pd(_complexIncrementer, _i));

        // Since this is the incrementer for the length of the screen, we
        // reset back to 0,1,2,3 that way we increment by 4 each cycle at the end
        // of each loop in the next for loop.
        _j = _mm256_set_pd(0.0, 1.0, 2.0, 3.0);

        for(j = 0; j < boundedRange; j += 4)
        {
            // cr = leftBound + realIncrementer*j;
            _cr = _mm256_add_pd(_leftBound, _mm256_mul_pd(_realIncrementer, _j));

            // Reset to zero.
            _zr = _mm256_setzero_pd();
            _zi = _mm256_setzero_pd();
            _n = _mm256_setzero_si256();

            // Since formal loops can't be used.
            repeat:
            // The mandelbrot set:
            // z = (z * z) + c
            // broken up since complex:
            // a = zr * zr - zi * zi + cr
            // b = zr * zI * 2.0 + ci
            // zr = a
            // zi = b



            // a = zr * zr - zi * zi + cr
            _zr2 = _mm256_mul_pd(_zr, _zr); // zr * zr
            _zi2 = _mm256_mul_pd(_zi, _zi); // zi * zi
            _a = _mm256_sub_pd(_zr2, _zi2); // zr^2 - zi^2
            _a = _mm256_add_pd(_a, _cr);    // zr^2 - zi^2 + cr

            // b = zr * zI * 2.0 + ci
            _b = _mm256_mul_pd(_zr, _zi);
            _b = _mm256_fmadd_pd(_b, _two, _ci);


            // zr = a
            // zi = b
            _zr = _a;
            _zi = _b;


            // Below is the if statement from Naive solution:
            //  if (a < 4.0 && n < maxIterations)
            // mask2 = | 0..0 | 0..0 | 1..1 | 1..1 | or something of the sort
            _a = _mm256_add_pd(_zr2, _zi2);

            // if _a < 4.0 for all registers
            _mask1 = _mm256_cmp_pd(_a, _four, _CMP_LT_OQ);
            _mask2 = _mm256_cmpgt_epi64(_maxIterations, _n);
            _mask2 = _mm256_and_si256(_mask2, _mm256_castpd_si256(_mask1));

            // Since:
            // mask2 ~= | 1..1 | 0..0 | 1..1 | 1..1 |
            // We want to use this data to add to the _n register, which would be bad unless
            // _n    ~= | 0..1 | 0..0 | 0..1 | 0..1 |
            _c = _mm256_and_si256(_mask2, _one);

            // Now we add _c to _n as this will only increment the correct results
            _n = _mm256_add_epi64(_n, _c);

            if(_mm256_movemask_pd(_mm256_castsi256_pd(_mask2)) > 0)
                goto repeat;

            // Set colors
            Mandelset->at(i,j + 0) = colorPicker->at(_n.m256i_i64[3] - 1);
            Mandelset->at(i,j + 1) = colorPicker->at(_n.m256i_i64[2] - 1);
            Mandelset->at(i,j + 2) = colorPicker->at(_n.m256i_i64[1] - 1);
            Mandelset->at(i,j + 3) = colorPicker->at(_n.m256i_i64[0] - 1);


            // Increment _j by 4 so that
            // | 0 | 1 | 2 | 3 |
            // | 4 | 4 | 4 | 4 | +
            // -----------------
            // | 4 | 5 | 6 | 7 |
            _j = _mm256_add_pd(_j, _four);
        }
        if(boundedRange != Mandelset->getColSize()) {
            double complexYValue = topBound - complexIncrementer*i;
            for (unsigned j2 = boundedRange; j2 < Mandelset->getColSize(); j2++) {
                // The current values we are calculating. Scaled from the current pixel.
                double realXValue = leftBound + realIncrementer * j2;

                double zx = 0;
                double zy = 0;

                unsigned n = 0;
                while (zx * zx + zy * zy <= 4 && n < maxIterations) {
                    double tempx = zx * zx - zy * zy + realXValue;
                    zy = 2 * zx * zy + complexYValue;
                    zx = tempx;
                    n += 1;
                }

                Mandelset->at(i, j2) = colorPicker->at(n);
            }
        }
    }

    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

    std::cout << std::left << std::setw(TIME_FORMAT) << " HOST:  Mandel Calculate Time: " <<
              std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
              " ms." <<  std::endl;

    return Mandelset;
}

std::unique_ptr<Propulsion::Matrix<double>> test(std::shared_ptr<Propulsion::Matrix<int>> s)
{
    auto M = std::make_unique<Propulsion::Matrix<double>>(10,10);
    return M;
}


std::unique_ptr<Propulsion::Matrix<int>> Propulsion::Mandelbrot::calculateMandelMultiThreaded(unsigned int threads, unsigned int wPixels, unsigned int hPixels, double leftBound, double rightBound, double topBound, double bottomBound, unsigned maxIterations, std::shared_ptr< Propulsion::Matrix< int>> colorPicker) {
    // Start clock for mandel calculation
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    // Create a matrix with the dimensions of the window
    std::unique_ptr<Propulsion::Matrix<int>> Mandelset( new Matrix<int>(hPixels, wPixels));

    unsigned rowsPerThread, colsPerThread, bottomThreadRows;
    rowsPerThread = std::floor(hPixels / threads);
    colsPerThread = wPixels;

    bottomThreadRows = hPixels - rowsPerThread * (threads - 1);

    // Real line is the horizontal line
    //double realIncrementer = ( rightBound - leftBound ) / wPixels;
    double complexIncrementer = (topBound - bottomBound) / hPixels;


    std::list< std::future< std::unique_ptr< Propulsion::Matrix< int>>>> mandelPromises;
    std::pair< std::vector< double>, std::vector< double>> promiseTopAndBottomBounds;

    promiseTopAndBottomBounds.first.resize(threads);
    promiseTopAndBottomBounds.second.resize(threads);

    for(unsigned i = 0; i < threads; i++)
    {


        if(i == (threads - 1))
        {
            promiseTopAndBottomBounds.first[i] = topBound - rowsPerThread * i * complexIncrementer;
            promiseTopAndBottomBounds.second[i] = bottomBound;
        }
        else
        {
            promiseTopAndBottomBounds.first[i] = topBound - rowsPerThread * i * complexIncrementer;
            promiseTopAndBottomBounds.second[i] = topBound - rowsPerThread * (i + 1) * complexIncrementer + complexIncrementer;
        }
    }



    // We skip last thread as last thread starts with special condition.
    for(unsigned i = 0; i < threads - 1; i++)
    {
        // Lambda wrapper for
        mandelPromises.push_back(std::async( [=](){
            return Propulsion::Mandelbrot::calculateMandelAVX256(colsPerThread, rowsPerThread, leftBound, rightBound, promiseTopAndBottomBounds.first[i], promiseTopAndBottomBounds.second[i], maxIterations, colorPicker);
        }));
    }
    mandelPromises.push_back(std::async( [=]() {
        return Propulsion::Mandelbrot::calculateMandelAVX256(colsPerThread, bottomThreadRows, leftBound, rightBound, promiseTopAndBottomBounds.first[threads - 1], promiseTopAndBottomBounds.second[threads - 1], maxIterations, colorPicker);
    }));

    std::list<std::future< std::unique_ptr< Propulsion::Matrix< int>>>>::iterator it;

    for(it = mandelPromises.begin(); it != mandelPromises.end(); it++)
    {
        if(it == mandelPromises.begin())
        {
            Mandelset = std::move(it->get());
            std::cout << Mandelset->getRowSize() << " : " << Mandelset->getColSize() << std::endl;
        }
        else
        {
            // Get from Promise the unique pointer, dereference it to pass to mergeBelow
            Mandelset->operator=(Mandelset->mergeBelow(*it->get()));
        }
    }


    // End Stats
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

    std::cout << std::left << std::setw(TIME_FORMAT) << " HOST:  Mandel Calculate Time: " <<
              std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
              " ms." << std::endl;


    return Mandelset;
}

void Propulsion::Mandelbrot::zoomInOnCursor()
{
    // Firstly, we ned to find cursor position.
    POINT cursor;

    if(GetCursorPos(&cursor))
    {
        if(ScreenToClient(this->hwnd, &cursor))
        {
            // Now that we have position, check if the client area is not negative.
            // Negative meaning that
            if(cursor.x >= 0 && cursor.y >= 0)
            {
                // Get x-range and y-range of current window.
                double xRange = this->rightBound - this->leftBound;
                double yRange = this->topBound   - this->bottomBound;

                // Get graph x and y positions on screen from cursor information.
                double xPos = this->leftBound   + (xRange) * (      (double)cursor.x    / this->clientWidthPixels );
                double yPos = this->bottomBound + (yRange) * ( 1 -  (double)cursor.y    / this->clientHeightPixels);


                //
                double newXRange = (xRange - xRange * this->zoomFactor) / 2;
                double newYRange = (yRange - yRange * this->zoomFactor) / 2;


                // Now calculate and set new bounds!
                this->leftBound     = xPos - newXRange;
                this->rightBound    = xPos + newXRange;
                this->bottomBound   = yPos - newYRange;
                this->topBound      = yPos + newYRange;


                // Change to zoomed, that way a recalculation is performed!
                this->redraw = true;
            }
            else
            {
                std::cout << "Can't zoom in on nothing" << std::endl;
            }
        }
        else
        {
            std::cout << "Well this is awkward" << std::endl;
        }
    }




}

void Propulsion::Mandelbrot::zoomOutOnCursor()
{
    // Firstly, we ned to find cursor position.
    POINT cursor;

    if(GetCursorPos(&cursor)) {
        // Get the cursor position on screen.
        if (ScreenToClient(this->hwnd, &cursor)) {
            // Now that we have position, check if the client area is not negative.
            // Which means we're in a proper region.
            if (cursor.x >= 0 && cursor.y >= 0) {
                // Get x-range and y-range of current window.
                double xRange = this->rightBound - this->leftBound;
                double yRange = this->topBound - this->bottomBound;

                // Get graph x and y positions on screen from cursor information.
                double xPos =(double) this->leftBound + (xRange) * ((double) cursor.x / (double) this->clientWidthPixels);
                double yPos = (double) this->bottomBound +(yRange) * (1.0 - (double) cursor.y / (double) this->clientHeightPixels);


                // Get new Ranges
                double newXRange = (xRange + xRange * this->zoomFactor) / 2;
                double newYRange = (yRange + yRange * this->zoomFactor) / 2;


                // Now calculate and set new bounds!
                this->leftBound = xPos - newXRange;
                this->rightBound = xPos + newXRange;
                this->bottomBound = yPos - newYRange;
                this->topBound = yPos + newYRange;


                // Change to zoomed, that way a recalculation is performed!
                this->redraw = true;
            } else {
                std::cout << "Can't zoom out on nothing" << std::endl;
            }
        } else {
            // Interesting...
        }
    }
}
