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

unsigned ITER_STEPS[] = {125,2000,5000, 100000, 200000, 500000, 1000000};
unsigned stepSize = 7;


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
    this->epoch = std::make_unique<Matrix<unsigned>>(ITER_STEPS , 1, stepSize);
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
    const int secondControlFirstRed     = 0x26;
    const int secondControlFirstGreen   = 0x02;
    const int secondControlFirstBlue    = 0x4f;
    const int secondControlSecondRed    = 0x56;
    const int secondControlSecondGreen  = 0x00;
    const int secondControlSecondBlue   = 0xbf;
    //const int secondControlSecondRed    = 0x9c;
    //const int secondControlSecondGreen  = 0x00;
    //const int secondControlSecondBlue   = 0xbf;


    // Third Color gradient.
    const int thirdControlFirstRed      = 0x03;
    const int thirdControlFirstGreen    = 0x00;
    const int thirdControlFirstBlue     = 0xcf;
    const int thirdControlSecondRed     = 0x00;
    const int thirdControlSecondGreen   = 0xac;
    const int thirdControlSecondBlue    = 0xcf;


    // Fourth color gradient
    const int fourthControlFirstRed     = 0x00;
    const int fourthControlFirstGreen   = 0xe8;
    const int fourthControlFirstBlue    = 0xba;
    const int fourthControlSecondRed    = 0xe8;
    const int fourthControlSecondGreen  = 0xdc;
    const int fourthControlSecondBlue   = 0x00;


    // Fifth color gradient
    const int fifthControlFirstRed      = 0xfa;
    const int fifthControlFirstGreen    = 0x7d;
    const int fifthControlFirstBlue     = 0x00;
    const int fifthControlSecondRed     = 0xfa;
    const int fifthControlSecondGreen   = 0x0c;
    const int fifthControlSecondBlue    = 0x00;


    const int sixthControlFirstRed      = 0xfa;
    const int sixthControlFirstGreen    = 0x00;
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
    this->colorPicker->at(totalColors - 1);

}

void Propulsion::Mandelbrot::paintWindow()
{
    mutexPainting.lock();

    //std::cout << "Starting Paint" << std::endl;
    // Check if window handle exists. If not return and error message.
    if(this->hwnd == nullptr || windowDestroyed)
    {
        std::cout << "Window closed!" << std::endl;
        mutexPainting.unlock();
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

            // Calculate Mandel
            /*
            this->Mandel = calculateMandelSingleThreaded(this->clientWidthPixels, this->clientHeightPixels,
                                                         this->leftBound, this->rightBound, this->topBound,
                                                         this->bottomBound, MAX_ITER,
                                                         this->colorPicker);*/

            this->Mandel = calculateMandelCUDA(this->clientWidthPixels, this->clientHeightPixels,
                                               this->leftBound, this->rightBound, this->topBound,
                                               this->bottomBound, this->iterations,this->colorPicker);


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
            std::cout << "You should never see this the fuck!" << std::endl;
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

    mutexPainting.unlock();

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
        mutexPainting.lock();
        std::cout << "Closing" << std::endl;
        PostQuitMessage(0);
        windowDestroyed = true;
        mutexPainting.unlock();
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

    // Calculate total copy constructor time + process time.
    std::chrono::high_resolution_clock::time_point end;


    // Event loop
    MSG msg;

    POINT p;

    while (GetMessage( &msg, NULL, 0, 0 ) > 0)
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);

        // If Q button is pressed, do something special!
        if (GetKeyState('Q') & 0x8000 && GetActiveWindow() == hwnd) {
            if(this->currentEpochSelection < this->epoch->getColSize())
            {
                currentEpochSelection++;
                std::cout << "Setting Iterations to: " << this->epoch->at(this->currentEpochSelection)<< std::endl;
                this->redraw = true;
                this->iterations = this->epoch->at(this->currentEpochSelection);
                Sleep(50);
            }
        }
        else if(GetKeyState('E') & 0x8000 && GetActiveWindow() == hwnd) {
            if(this->currentEpochSelection > 0)
            {
                currentEpochSelection--;
                std::cout << "Setting Iterations to: " << this->epoch->at(this->currentEpochSelection)<< std::endl;
                this->redraw = true;
                this->iterations = this->epoch->at(this->currentEpochSelection);
                Sleep(50);
            }

        }

        // Check if we can draw at 100FPS.
        if ((float) (std::chrono::high_resolution_clock::now() - start).count() / 1000 > 10) {
            // Check if left mouse button is pressed, if so and active window, then we ignore the paint as it may be a resize!
            if (GetAsyncKeyState(VK_LBUTTON) & 0x8000 && GetActiveWindow() == hwnd) {

                }
            else {
                paintWindow();
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
        for(unsigned j = 0; j < Mandelset->getColSize(); j++)
        {
            // The current values we are calculating. Scaled from the current pixel.
            double realXValue = leftBound + realIncrementer*j;
            double complexYValue = topBound - complexIncrementer*i;

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
                Mandelset->at(i,j) = 0x000000;
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

    // Get the cursor position on screen.
    if(GetCursorPos(&cursor))
    {
        if(ScreenToClient(this->hwnd, &cursor))
        {
            std::cout << cursor.x << " : " << cursor.y << std::endl;
        }
        else
        {
            std::cout << "Well this is awkward" << std::endl;
        }
    }

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
        double newXRange = (xRange + xRange * this->zoomFactor) / 2;
        double newYRange = (yRange + yRange * this->zoomFactor) / 2;


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


