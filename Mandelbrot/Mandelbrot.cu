//
// Created by steve on 12/18/2020.
//

#include "../Propulsion.cuh"

#define MAX_ITER 1000

bool activeApp = false;


Propulsion::Mandelbrot::Mandelbrot(unsigned int width, unsigned int height)
{
    this->widthPixels = width;
    this->heightPixels = height;
}

void PaintWindow( HWND hwnd , unsigned pixelWidth, unsigned pixelHeight)
{
    if(hwnd == nullptr)
    {
        std::cout << "Fuck Off" << std::endl;
    }

    HDC hdc = GetDC( hwnd);

    int c = 0xf3f200;


    // Start clock for mandel calculation
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    Propulsion::Matrix<int> Mandel = Propulsion::Mandelbrot::calculateMandelCPU(pixelWidth, pixelHeight, -3.0, 3.0, 2, -2, MAX_ITER);

    // Calculate total copy constructor time + process time.
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    float milliseconds = (float) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;

    std::cout << std::left << std::setw(TIME_FORMAT) << " HOST:  Mandel Copy + Calculate Time: " <<
              std::right << std::setw(TIME_WIDTH) << std::fixed << std::setprecision(TIME_PREC) << milliseconds <<
              " ms." <<  std::endl;


    // Start clock for mandel calculation
    std::chrono::high_resolution_clock::time_point startDraw = std::chrono::high_resolution_clock::now();

    int arr[] = {0x000000,0x888888, 0x888888, 0x888888};

    auto x = CreateBitmap(2, 2, 1, 32,  arr);

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


    ReleaseDC(hwnd,hdc);
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
                PaintWindow(hwnd, rect.right, rect.bottom );
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
        PostQuitMessage(0);
        return 0;

    case WM_ACTIVATEAPP:
        if(activeApp != true)
        {
            activeApp = true;
            PaintWindow(hwnd, rect.right, rect.bottom);
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
    HWND hwnd = CreateWindowA(
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

    // Event loop
    MSG msg;
    while (GetMessage( &msg, NULL, 0, 0 ) > 0)
    {
        TranslateMessage( &msg );
        DispatchMessage( &msg );
    }

    std::cout << "See ya: " << msg.wParam << std::endl;
}

Propulsion::Matrix<int> Propulsion::Mandelbrot::calculateMandelCPU(unsigned int wPixels, unsigned int hPixels, double leftBound, double rightBound, double topBound, double bottomBound, unsigned maxIterations)
{
    // Start clock for mandel calculation
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    // Create a matrix with the dimensions of the window
    Propulsion::Matrix<int> Mandelset(hPixels, wPixels);


    // Real line is the horizontal line
    double realIncrementer = ( rightBound - leftBound ) / wPixels;
    double complexIncrementer = (topBound - bottomBound) / hPixels;

    for(unsigned i = 0; i < Mandelset.getRowSize(); i++)
    {
        for(unsigned j = 0; j < Mandelset.getColSize(); j++)
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
                Mandelset(i,j) = 0x000000;
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
                Mandelset(i,j) = color;
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

