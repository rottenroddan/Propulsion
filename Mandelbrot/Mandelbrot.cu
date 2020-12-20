//
// Created by steve on 12/18/2020.
//

int x = 0;
int y = 0;


Propulsion::Mandelbrot::Mandelbrot(unsigned int width, unsigned int height)
{
    this->widthPixels = width;
    this->heightPixels = height;
}

void PaintWindow( HWND hwnd )
{
    if(hwnd == nullptr)
    {
        std::cout << "Fuck Off" << std::endl;
    }

    HDC hdc = GetDC( hwnd);


    if(SetPixel( hdc, x, y, 0x0000FF ) == -1)
    {
        std::cout << "What " << &hwnd << " " << &hdc << std::endl;
    }// green

    x++;
    y++;

    if(SetPixel( hdc, x, y, 0x0000FF ) == -1)
    {
        std::cout << "What " << &hwnd << " " << &hdc << std::endl;
    }// green

    ReleaseDC(hwnd,hdc);
}


LRESULT CALLBACK WndProc (HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch(msg)
    {
    case WM_KEYDOWN:
        if(wParam != VK_ESCAPE)
        {
            if(wParam == 0x51)
            {

            }
            else
            {
                PaintWindow(hwnd);
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
    /*
    case WM_PAINT:
        std::cout << "The Fuck" << std::endl;
        PaintWindow(hwnd);
        return 0;*/
    default:
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
        (HBRUSH )GetStockObject(BLACK_BRUSH), // background color == black
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

    // Event loop
    MSG msg;
    while (GetMessage( &msg, NULL, 0, 0 ) > 0)
    {
        TranslateMessage( &msg );
        DispatchMessage( &msg );
    }

    std::cout << "See ya: " << msg.wParam << std::endl;
}

