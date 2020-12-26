//
// Created by steve on 12/18/2020.
//

#include "../Propulsion.cuh"
#include <d2d1.h>
#include <d2d1helper.h>
#include <dwrite.h>
#include <wincodec.h>

#define MAX_ITER 500

bool activeApp = false;

int i = 640;

std::mutex mutexPainting;
std::atomic<bool> windowDestroyed;


Propulsion::Mandelbrot::Mandelbrot(unsigned int width, unsigned int height, double leftBound, double rightBound, double topBound, double bottomBound)
{
    // Set window width/height.
    this->windowWidthPixels = width;
    this->windowHeightPixels = height;

    // Set bounds for graphing.
    this->topBound = topBound;
    this->bottomBound = bottomBound;
    this->leftBound = leftBound;
    this->rightBound = rightBound;
}

void Propulsion::Mandelbrot::paintWindow()
{
    mutexPainting.lock();

    std::cout << "Starting Paint" << std::endl;
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

    if(clientRect.bottom > 0 && clientRect.right > 0)
    {
        // Check if we need to recalculate!
        if(clientRect.right != this->clientWidthPixels || clientRect.bottom != this->clientHeightPixels)
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
            this->Mandel = calculateMandelCPU(this->clientWidthPixels, this->clientHeightPixels, this->leftBound, this->rightBound, this->topBound, this->bottomBound, MAX_ITER);

            // Calculate total copy constructor time + process time.
            std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
            float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

            // Print about the total time for calculation and copy constructor!
            std::cout << std::left << std::setw(TIME_FORMAT) << " HOST:  Mandel Copy + Calculate Time: " <<
                      std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
                      " ms." <<  std::endl;
        }

        if(Mandel->getColSize() != clientRect.right || Mandel->getRowSize() != clientRect.bottom)
        {
            std::cout << "You should never see this the fuck!" << std::endl;
        }

        // Get Device Context handle.
        HDC hdc = GetDC(this->hwnd);


        //std::cout << clientRect.top << " + " << clientRect.bottom << " : " << clientRect.left << "+" <<  clientRect.right << std::endl;

        // Create bitmap that way we can draw all at once instead of setpixel.
        HBITMAP map = CreateBitmap(this->clientWidthPixels, this->clientHeightPixels, 1, 8*4, (void*) Mandel->getArray());

        HDC src = CreateCompatibleDC(hdc);

        SelectObject(src, map);

        BitBlt(hdc, 0, 0, this->clientWidthPixels, this->clientHeightPixels, src, 0, 0, SRCCOPY);


        DeleteDC(src);


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

        ReleaseDC(this->hwnd,hdc);
    }
    std::cout << "Painting" << std::endl;
    mutexPainting.unlock();

}

void PaintWindow( HWND hwnd , unsigned pixelWidth, unsigned pixelHeight)
{
    /*
    if(hwnd == nullptr)
    {
        std::cout << "Fuck Off" << std::endl;
    }

    HDC hdc = GetDC( hwnd);

    int c = 0xf3f200;


    // Start clock for mandel calculation
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    Propulsion::Matrix<int> Mandel = Propulsion::Mandelbrot::calculateMandelCPU(pixelWidth, pixelHeight, -2.0, 1.0, 1.5, -1.5, MAX_ITER);

    // Calculate total copy constructor time + process time.
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

    std::cout << std::left << std::setw(TIME_FORMAT) << " HOST:  Mandel Copy + Calculate Time: " <<
              std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
              " ms." <<  std::endl;


    // Start clock for mandel calculation
    std::chrono::high_resolution_clock::time_point startDraw = std::chrono::high_resolution_clock::now();


    int *arr = (int*) calloc(pixelWidth * pixelHeight, sizeof(int));

    HBITMAP map = CreateBitmap(pixelWidth, pixelHeight, 1, 8*4, (void*) Mandel.getArray());

    HDC src = CreateCompatibleDC(hdc);

    SelectObject(src, map);

    BitBlt(hdc, 0, 0, pixelWidth, pixelHeight, src, 0, 0, SRCCOPY);


    DeleteDC(src);



    for(unsigned i = 0; i < pixelHeight; i++)
    {
        for(unsigned j = 0; j < pixelWidth; j++)
        {
            SetPixel(hdc, j, i, Mandel(i,j));
        }
    }

    // Calculate total copy constructor time + process time.
    std::chrono::high_resolution_clock::time_point endDraw = std::chrono::high_resolution_clock::now();
    milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(endDraw - startDraw).count() / 1000;

    std::cout << std::left << std::setw(TIME_FORMAT) << " HOST:  Mandel Draw Time: " <<
              std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
              " ms." <<  std::endl;


    ReleaseDC(hwnd,hdc);*/
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
    case WM_MOUSEWHEEL:
        if(GET_WHEEL_DELTA_WPARAM(wParam) > 0)
        {
            std::cout << "Mouse Wheel Up" <<
            std::endl;
        }
        else if(GET_WHEEL_DELTA_WPARAM(wParam) < 0)
        {
            std::cout << "Mouse Wheel Down" << std::endl;
        }
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
            "Window sample",     // window title
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

    if(IsWindow(hwnd))
    {
        std::cout << "So its a window somehow!" << std::endl;
    }

    // set window destroyed to false!
    windowDestroyed = false;


    // Start clock for mandel calculation
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    // Calculate total copy constructor time + process time.
    std::chrono::high_resolution_clock::time_point end;


    // Event loop
    MSG msg;

    while (GetMessage( &msg, NULL, 0, 0 ) > 0)
    {
        TranslateMessage( &msg );
        DispatchMessage( &msg );

        // If Q button is pressed, do something special!
        if(GetKeyState('Q') & 0x8000 && GetActiveWindow() == hwnd)
        {
            std::cout << "Q" << std::endl;
        }

        // Check if we can draw at 100FPS.
        if((float) (std::chrono::high_resolution_clock::now() -  start).count() / 1000 > 10 )
        {
            // Check if left mouse button is pressed, if so and active window, then we ignore the paint as it may be a resize!
            if(GetAsyncKeyState(VK_LBUTTON) & 0x8000 && GetActiveWindow() == hwnd)
            {
                std::cout << "Pressing" << std::endl;
            }
            else
            {
                paintWindow();
            }

            // Reset timer from last frame!
            start = std::chrono::high_resolution_clock::now();
        }
    }

    std::cout << "See ya: " << msg.wParam << std::endl;
}

std::unique_ptr<Propulsion::Matrix<int>> Propulsion::Mandelbrot::calculateMandelCPU(unsigned int wPixels, unsigned int hPixels, double leftBound, double rightBound, double topBound, double bottomBound, unsigned maxIterations)
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

            if(n == maxIterations)
            {
                // Set to Black
                Mandelset->at(i,j) = 0x000000;
            }
            else
            {
                int color = 0x000000;
                double difference = (double)n/(double)maxIterations;

                /*
                if(difference < .25)
                {
                    c = 0xFF0000;
                }
                else if(difference < .50)
                {
                    c = 0x601475;
                }
                else if(difference < .75)
                {
                    c = 0xFF0000;
                }*/

                double red = 0xff;
                double green = 0xff;
                double blue = 0xff;

                double compliment = 1 - difference;

                // Alter towards the compliment, meaning a small difference -> white.
                red     = red   * compliment;
                green   = green * compliment;
                blue    = blue  * compliment;

                color += ((int)blue) << 16;
                color += ((int)green) << 8;
                color += ((int)red);




                // Set to White
                Mandelset->at(i,j) = color;
                //std::cout << "Failed at: " << n << " " << realXValue << " " << complexYValue << std::endl;
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

