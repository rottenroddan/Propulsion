#pragma section("__nv_managed_data__")
static char __nv_inited_managed_rt = 0; static void **__nv_fatbinhandle_for_managed_rt; static void __nv_save_fatbinhandle_for_managed_rt(void **in){__nv_fatbinhandle_for_managed_rt = in;} static char __nv_init_managed_rt_with_module(void **); static inline void __nv_init_managed_rt(void) { __nv_inited_managed_rt = (__nv_inited_managed_rt ? __nv_inited_managed_rt                 : __nv_init_managed_rt_with_module(__nv_fatbinhandle_for_managed_rt));}
#line 1 "CMakeCUDACompilerId.cu"
#define __nv_is_extended_device_lambda_closure_type(X) false
#define __nv_is_extended_host_device_lambda_closure_type(X) false
#if defined(__nv_is_extended_device_lambda_closure_type) && defined(__nv_is_extended_host_device_lambda_closure_type)
#endif

#line 1
#line 67 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1/bin/../include\\cuda_runtime.h"
#pragma warning(push)
#pragma warning(disable: 4820)
#line 708 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\sal.h"
#pragma region Input Buffer SAL 1 compatibility macros
#line 1472
#pragma endregion Input Buffer SAL 1 compatibility macros
#line 2361 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\sal.h"
extern "C" {
#line 2967 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\sal.h"
}
#line 22 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\concurrencysal.h"
extern "C" {
#line 391 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\concurrencysal.h"
}
#line 15 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\vadefs.h"
#pragma pack ( push, 8 )
#line 18
extern "C" {
#line 28 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\vadefs.h"
typedef unsigned __int64 uintptr_t; 
#line 39 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\vadefs.h"
typedef char *va_list; 
#line 112 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\vadefs.h"
void __cdecl __va_start(va_list *, ...); 
#line 124 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\vadefs.h"
}
#line 128 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\vadefs.h"
extern "C++" {
#line 130
template< class _Ty> 
#line 131
struct __vcrt_va_list_is_reference { 
#line 133
enum: bool { __the_value}; 
#line 134
}; 
#line 136
template< class _Ty> 
#line 137
struct __vcrt_va_list_is_reference< _Ty &>  { 
#line 139
enum: bool { __the_value = '\001'}; 
#line 140
}; 
#line 142
template< class _Ty> 
#line 143
struct __vcrt_va_list_is_reference< _Ty &&>  { 
#line 145
enum: bool { __the_value = '\001'}; 
#line 146
}; 
#line 148
template< class _Ty> 
#line 149
struct __vcrt_assert_va_start_is_not_reference { 
#line 151
static_assert((!__vcrt_va_list_is_reference< _Ty> ::__the_value), "va_start argument must not have reference type and must not be parenthesized");
#line 153
}; 
#line 154
}
#line 164 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\vadefs.h"
#pragma pack ( pop )
#line 83 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\vcruntime.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 180 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\vcruntime.h"
typedef unsigned __int64 size_t; 
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#line 181
typedef __int64 ptrdiff_t; 
#line 182
typedef __int64 intptr_t; 
#line 190 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\vcruntime.h"
typedef bool __vcrt_bool; 
#line 233 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\vcruntime.h"
extern "C++" {
#line 235
template< class _CountofType, size_t _SizeOfArray> char (*__countof_helper(__unaligned _CountofType (& _Array)[_SizeOfArray]))[_SizeOfArray]; 
#line 239
}
#line 319 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\vcruntime.h"
void __cdecl __security_init_cookie(); 
#line 325
void __cdecl __security_check_cookie(uintptr_t _StackCookie); 
#line 326
__declspec(noreturn) void __cdecl __report_gsfailure(uintptr_t _StackCookie); 
#line 330 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\vcruntime.h"
extern uintptr_t __security_cookie; 
#line 338 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\vcruntime.h"
}__pragma( pack ( pop )) 
#line 12 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 136 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt.h"
extern "C++" {
#line 138
template< bool _Enable, class _Ty> struct _CrtEnableIf; 
#line 141
template< class _Ty> 
#line 142
struct _CrtEnableIf< true, _Ty>  { 
#line 144
typedef _Ty _Type; 
#line 145
}; 
#line 146
}
#line 150 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt.h"
typedef bool __crt_bool; 
#line 278 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt.h"
void __cdecl _invalid_parameter_noinfo(); 
#line 279
__declspec(noreturn) void __cdecl _invalid_parameter_noinfo_noreturn(); 
#line 281
__declspec(noreturn) void __cdecl 
#line 282
_invoke_watson(const __wchar_t * _Expression, const __wchar_t * _FunctionName, const __wchar_t * _FileName, unsigned _LineNo, uintptr_t _Reserved); 
#line 511 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt.h"
typedef int errno_t; 
#line 512
typedef unsigned short wint_t; 
#line 513
typedef unsigned short wctype_t; 
#line 514
typedef long __time32_t; 
#line 515
typedef __int64 __time64_t; 
#line 522
typedef 
#line 517
struct __crt_locale_data_public { 
#line 519
const unsigned short *_locale_pctype; 
#line 520
int _locale_mb_cur_max; 
#line 521
unsigned _locale_lc_codepage; 
#line 522
} __crt_locale_data_public; 
#line 528
typedef 
#line 524
struct __crt_locale_pointers { 
#line 526
struct __crt_locale_data *locinfo; 
#line 527
struct __crt_multibyte_data *mbcinfo; 
#line 528
} __crt_locale_pointers; 
#line 530
typedef __crt_locale_pointers *_locale_t; 
#line 536
typedef 
#line 532
struct _Mbstatet { 
#line 534
unsigned long _Wchar; 
#line 535
unsigned short _Byte, _State; 
#line 536
} _Mbstatet; 
#line 538
typedef _Mbstatet mbstate_t; 
#line 552 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt.h"
typedef __time64_t time_t; 
#line 562 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt.h"
typedef size_t rsize_t; 
#line 2011 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt.h"
}__pragma( pack ( pop )) 
#line 66 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\device_types.h"
#if 0
#line 66
enum cudaRoundMode { 
#line 68
cudaRoundNearest, 
#line 69
cudaRoundZero, 
#line 70
cudaRoundPosInf, 
#line 71
cudaRoundMinInf
#line 72
}; 
#endif
#line 98 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 98
struct char1 { 
#line 100
signed char x; 
#line 101
}; 
#endif
#line 103 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 103
struct uchar1 { 
#line 105
unsigned char x; 
#line 106
}; 
#endif
#line 109 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 109
struct __declspec(align(2)) char2 { 
#line 111
signed char x, y; 
#line 112
}; 
#endif
#line 114 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 114
struct __declspec(align(2)) uchar2 { 
#line 116
unsigned char x, y; 
#line 117
}; 
#endif
#line 119 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 119
struct char3 { 
#line 121
signed char x, y, z; 
#line 122
}; 
#endif
#line 124 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 124
struct uchar3 { 
#line 126
unsigned char x, y, z; 
#line 127
}; 
#endif
#line 129 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 129
struct __declspec(align(4)) char4 { 
#line 131
signed char x, y, z, w; 
#line 132
}; 
#endif
#line 134 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 134
struct __declspec(align(4)) uchar4 { 
#line 136
unsigned char x, y, z, w; 
#line 137
}; 
#endif
#line 139 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 139
struct short1 { 
#line 141
short x; 
#line 142
}; 
#endif
#line 144 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 144
struct ushort1 { 
#line 146
unsigned short x; 
#line 147
}; 
#endif
#line 149 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 149
struct __declspec(align(4)) short2 { 
#line 151
short x, y; 
#line 152
}; 
#endif
#line 154 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 154
struct __declspec(align(4)) ushort2 { 
#line 156
unsigned short x, y; 
#line 157
}; 
#endif
#line 159 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 159
struct short3 { 
#line 161
short x, y, z; 
#line 162
}; 
#endif
#line 164 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 164
struct ushort3 { 
#line 166
unsigned short x, y, z; 
#line 167
}; 
#endif
#line 169 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 169
struct __declspec(align(8)) short4 { short x; short y; short z; short w; }; 
#endif
#line 170 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 170
struct __declspec(align(8)) ushort4 { unsigned short x; unsigned short y; unsigned short z; unsigned short w; }; 
#endif
#line 172 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 172
struct int1 { 
#line 174
int x; 
#line 175
}; 
#endif
#line 177 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 177
struct uint1 { 
#line 179
unsigned x; 
#line 180
}; 
#endif
#line 182 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 182
struct __declspec(align(8)) int2 { int x; int y; }; 
#endif
#line 183 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 183
struct __declspec(align(8)) uint2 { unsigned x; unsigned y; }; 
#endif
#line 185 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 185
struct int3 { 
#line 187
int x, y, z; 
#line 188
}; 
#endif
#line 190 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 190
struct uint3 { 
#line 192
unsigned x, y, z; 
#line 193
}; 
#endif
#line 195 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 195
struct __declspec(align(16)) int4 { 
#line 197
int x, y, z, w; 
#line 198
}; 
#endif
#line 200 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 200
struct __declspec(align(16)) uint4 { 
#line 202
unsigned x, y, z, w; 
#line 203
}; 
#endif
#line 205 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 205
struct long1 { 
#line 207
long x; 
#line 208
}; 
#endif
#line 210 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 210
struct ulong1 { 
#line 212
unsigned long x; 
#line 213
}; 
#endif
#line 216 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 216
struct __declspec(align(8)) long2 { long x; long y; }; 
#endif
#line 217 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 217
struct __declspec(align(8)) ulong2 { unsigned long x; unsigned long y; }; 
#endif
#line 232 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 232
struct long3 { 
#line 234
long x, y, z; 
#line 235
}; 
#endif
#line 237 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 237
struct ulong3 { 
#line 239
unsigned long x, y, z; 
#line 240
}; 
#endif
#line 242 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 242
struct __declspec(align(16)) long4 { 
#line 244
long x, y, z, w; 
#line 245
}; 
#endif
#line 247 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 247
struct __declspec(align(16)) ulong4 { 
#line 249
unsigned long x, y, z, w; 
#line 250
}; 
#endif
#line 252 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 252
struct float1 { 
#line 254
float x; 
#line 255
}; 
#endif
#line 274 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 274
struct __declspec(align(8)) float2 { float x; float y; }; 
#endif
#line 279 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 279
struct float3 { 
#line 281
float x, y, z; 
#line 282
}; 
#endif
#line 284 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 284
struct __declspec(align(16)) float4 { 
#line 286
float x, y, z, w; 
#line 287
}; 
#endif
#line 289 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 289
struct longlong1 { 
#line 291
__int64 x; 
#line 292
}; 
#endif
#line 294 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 294
struct ulonglong1 { 
#line 296
unsigned __int64 x; 
#line 297
}; 
#endif
#line 299 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 299
struct __declspec(align(16)) longlong2 { 
#line 301
__int64 x, y; 
#line 302
}; 
#endif
#line 304 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 304
struct __declspec(align(16)) ulonglong2 { 
#line 306
unsigned __int64 x, y; 
#line 307
}; 
#endif
#line 309 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 309
struct longlong3 { 
#line 311
__int64 x, y, z; 
#line 312
}; 
#endif
#line 314 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 314
struct ulonglong3 { 
#line 316
unsigned __int64 x, y, z; 
#line 317
}; 
#endif
#line 319 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 319
struct __declspec(align(16)) longlong4 { 
#line 321
__int64 x, y, z, w; 
#line 322
}; 
#endif
#line 324 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 324
struct __declspec(align(16)) ulonglong4 { 
#line 326
unsigned __int64 x, y, z, w; 
#line 327
}; 
#endif
#line 329 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 329
struct double1 { 
#line 331
double x; 
#line 332
}; 
#endif
#line 334 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 334
struct __declspec(align(16)) double2 { 
#line 336
double x, y; 
#line 337
}; 
#endif
#line 339 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 339
struct double3 { 
#line 341
double x, y, z; 
#line 342
}; 
#endif
#line 344 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 344
struct __declspec(align(16)) double4 { 
#line 346
double x, y, z, w; 
#line 347
}; 
#endif
#line 361 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef char1 
#line 361
char1; 
#endif
#line 362 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef uchar1 
#line 362
uchar1; 
#endif
#line 363 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef char2 
#line 363
char2; 
#endif
#line 364 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef uchar2 
#line 364
uchar2; 
#endif
#line 365 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef char3 
#line 365
char3; 
#endif
#line 366 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef uchar3 
#line 366
uchar3; 
#endif
#line 367 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef char4 
#line 367
char4; 
#endif
#line 368 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef uchar4 
#line 368
uchar4; 
#endif
#line 369 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef short1 
#line 369
short1; 
#endif
#line 370 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef ushort1 
#line 370
ushort1; 
#endif
#line 371 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef short2 
#line 371
short2; 
#endif
#line 372 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef ushort2 
#line 372
ushort2; 
#endif
#line 373 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef short3 
#line 373
short3; 
#endif
#line 374 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef ushort3 
#line 374
ushort3; 
#endif
#line 375 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef short4 
#line 375
short4; 
#endif
#line 376 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef ushort4 
#line 376
ushort4; 
#endif
#line 377 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef int1 
#line 377
int1; 
#endif
#line 378 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef uint1 
#line 378
uint1; 
#endif
#line 379 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef int2 
#line 379
int2; 
#endif
#line 380 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef uint2 
#line 380
uint2; 
#endif
#line 381 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef int3 
#line 381
int3; 
#endif
#line 382 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef uint3 
#line 382
uint3; 
#endif
#line 383 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef int4 
#line 383
int4; 
#endif
#line 384 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef uint4 
#line 384
uint4; 
#endif
#line 385 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef long1 
#line 385
long1; 
#endif
#line 386 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef ulong1 
#line 386
ulong1; 
#endif
#line 387 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef long2 
#line 387
long2; 
#endif
#line 388 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef ulong2 
#line 388
ulong2; 
#endif
#line 389 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef long3 
#line 389
long3; 
#endif
#line 390 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef ulong3 
#line 390
ulong3; 
#endif
#line 391 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef long4 
#line 391
long4; 
#endif
#line 392 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef ulong4 
#line 392
ulong4; 
#endif
#line 393 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef float1 
#line 393
float1; 
#endif
#line 394 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef float2 
#line 394
float2; 
#endif
#line 395 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef float3 
#line 395
float3; 
#endif
#line 396 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef float4 
#line 396
float4; 
#endif
#line 397 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef longlong1 
#line 397
longlong1; 
#endif
#line 398 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef ulonglong1 
#line 398
ulonglong1; 
#endif
#line 399 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef longlong2 
#line 399
longlong2; 
#endif
#line 400 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef ulonglong2 
#line 400
ulonglong2; 
#endif
#line 401 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef longlong3 
#line 401
longlong3; 
#endif
#line 402 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef ulonglong3 
#line 402
ulonglong3; 
#endif
#line 403 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef longlong4 
#line 403
longlong4; 
#endif
#line 404 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef ulonglong4 
#line 404
ulonglong4; 
#endif
#line 405 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef double1 
#line 405
double1; 
#endif
#line 406 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef double2 
#line 406
double2; 
#endif
#line 407 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef double3 
#line 407
double3; 
#endif
#line 408 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef double4 
#line 408
double4; 
#endif
#line 416 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
#line 416
struct dim3 { 
#line 418
unsigned x, y, z; 
#line 430 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
}; 
#endif
#line 432 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_types.h"
#if 0
typedef dim3 
#line 432
dim3; 
#endif
#line 13 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\limits.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 88 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\limits.h"
}__pragma( pack ( pop )) 
#line 14 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stddef.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 19
namespace std { 
#line 21
typedef decltype((nullptr)) nullptr_t; 
#line 22
}
#line 24
using std::nullptr_t;
#line 31 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stddef.h"
int *__cdecl _errno(); 
#line 34
errno_t __cdecl _set_errno(int _Value); 
#line 35
errno_t __cdecl _get_errno(int * _Value); 
#line 51 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stddef.h"
extern unsigned long __cdecl __threadid(); 
#line 53
extern uintptr_t __cdecl __threadhandle(); 
#line 57
}__pragma( pack ( pop )) 
#line 197 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 197
enum cudaError { 
#line 204
cudaSuccess, 
#line 210
cudaErrorInvalidValue, 
#line 216
cudaErrorMemoryAllocation, 
#line 222
cudaErrorInitializationError, 
#line 229
cudaErrorCudartUnloading, 
#line 236
cudaErrorProfilerDisabled, 
#line 244
cudaErrorProfilerNotInitialized, 
#line 251
cudaErrorProfilerAlreadyStarted, 
#line 258
cudaErrorProfilerAlreadyStopped, 
#line 267
cudaErrorInvalidConfiguration, 
#line 273
cudaErrorInvalidPitchValue = 12, 
#line 279
cudaErrorInvalidSymbol, 
#line 287
cudaErrorInvalidHostPointer = 16, 
#line 295
cudaErrorInvalidDevicePointer, 
#line 301
cudaErrorInvalidTexture, 
#line 307
cudaErrorInvalidTextureBinding, 
#line 314
cudaErrorInvalidChannelDescriptor, 
#line 320
cudaErrorInvalidMemcpyDirection, 
#line 330
cudaErrorAddressOfConstant, 
#line 339
cudaErrorTextureFetchFailed, 
#line 348
cudaErrorTextureNotBound, 
#line 357
cudaErrorSynchronizationError, 
#line 363
cudaErrorInvalidFilterSetting, 
#line 369
cudaErrorInvalidNormSetting, 
#line 377
cudaErrorMixedDeviceExecution, 
#line 385
cudaErrorNotYetImplemented = 31, 
#line 394
cudaErrorMemoryValueTooLarge, 
#line 401
cudaErrorStubLibrary = 34, 
#line 408
cudaErrorInsufficientDriver, 
#line 415
cudaErrorCallRequiresNewerDriver, 
#line 421
cudaErrorInvalidSurface, 
#line 427
cudaErrorDuplicateVariableName = 43, 
#line 433
cudaErrorDuplicateTextureName, 
#line 439
cudaErrorDuplicateSurfaceName, 
#line 449
cudaErrorDevicesUnavailable, 
#line 462
cudaErrorIncompatibleDriverContext = 49, 
#line 468
cudaErrorMissingConfiguration = 52, 
#line 477
cudaErrorPriorLaunchFailure, 
#line 484
cudaErrorLaunchMaxDepthExceeded = 65, 
#line 492
cudaErrorLaunchFileScopedTex, 
#line 500
cudaErrorLaunchFileScopedSurf, 
#line 515
cudaErrorSyncDepthExceeded, 
#line 527
cudaErrorLaunchPendingCountExceeded, 
#line 533
cudaErrorInvalidDeviceFunction = 98, 
#line 539
cudaErrorNoDevice = 100, 
#line 545
cudaErrorInvalidDevice, 
#line 550
cudaErrorDeviceNotLicensed, 
#line 555
cudaErrorStartupFailure = 127, 
#line 560
cudaErrorInvalidKernelImage = 200, 
#line 570
cudaErrorDeviceUninitialized, 
#line 575
cudaErrorMapBufferObjectFailed = 205, 
#line 580
cudaErrorUnmapBufferObjectFailed, 
#line 586
cudaErrorArrayIsMapped, 
#line 591
cudaErrorAlreadyMapped, 
#line 599
cudaErrorNoKernelImageForDevice, 
#line 604
cudaErrorAlreadyAcquired, 
#line 609
cudaErrorNotMapped, 
#line 615
cudaErrorNotMappedAsArray, 
#line 621
cudaErrorNotMappedAsPointer, 
#line 627
cudaErrorECCUncorrectable, 
#line 633
cudaErrorUnsupportedLimit, 
#line 639
cudaErrorDeviceAlreadyInUse, 
#line 645
cudaErrorPeerAccessUnsupported, 
#line 651
cudaErrorInvalidPtx, 
#line 656
cudaErrorInvalidGraphicsContext, 
#line 662
cudaErrorNvlinkUncorrectable, 
#line 669
cudaErrorJitCompilerNotFound, 
#line 676
cudaErrorUnsupportedPtxVersion, 
#line 681
cudaErrorInvalidSource = 300, 
#line 686
cudaErrorFileNotFound, 
#line 691
cudaErrorSharedObjectSymbolNotFound, 
#line 696
cudaErrorSharedObjectInitFailed, 
#line 701
cudaErrorOperatingSystem, 
#line 708
cudaErrorInvalidResourceHandle = 400, 
#line 714
cudaErrorIllegalState, 
#line 720
cudaErrorSymbolNotFound = 500, 
#line 728
cudaErrorNotReady = 600, 
#line 736
cudaErrorIllegalAddress = 700, 
#line 745
cudaErrorLaunchOutOfResources, 
#line 756
cudaErrorLaunchTimeout, 
#line 762
cudaErrorLaunchIncompatibleTexturing, 
#line 769
cudaErrorPeerAccessAlreadyEnabled, 
#line 776
cudaErrorPeerAccessNotEnabled, 
#line 789
cudaErrorSetOnActiveProcess = 708, 
#line 796
cudaErrorContextIsDestroyed, 
#line 803
cudaErrorAssert, 
#line 810
cudaErrorTooManyPeers, 
#line 816
cudaErrorHostMemoryAlreadyRegistered, 
#line 822
cudaErrorHostMemoryNotRegistered, 
#line 831
cudaErrorHardwareStackError, 
#line 839
cudaErrorIllegalInstruction, 
#line 848
cudaErrorMisalignedAddress, 
#line 859
cudaErrorInvalidAddressSpace, 
#line 867
cudaErrorInvalidPc, 
#line 878
cudaErrorLaunchFailure, 
#line 887
cudaErrorCooperativeLaunchTooLarge, 
#line 892
cudaErrorNotPermitted = 800, 
#line 898
cudaErrorNotSupported, 
#line 907
cudaErrorSystemNotReady, 
#line 914
cudaErrorSystemDriverMismatch, 
#line 923
cudaErrorCompatNotSupportedOnDevice, 
#line 928
cudaErrorStreamCaptureUnsupported = 900, 
#line 934
cudaErrorStreamCaptureInvalidated, 
#line 940
cudaErrorStreamCaptureMerge, 
#line 945
cudaErrorStreamCaptureUnmatched, 
#line 951
cudaErrorStreamCaptureUnjoined, 
#line 958
cudaErrorStreamCaptureIsolation, 
#line 964
cudaErrorStreamCaptureImplicit, 
#line 970
cudaErrorCapturedEvent, 
#line 977
cudaErrorStreamCaptureWrongThread, 
#line 982
cudaErrorTimeout, 
#line 988
cudaErrorGraphExecUpdateFailure, 
#line 993
cudaErrorUnknown = 999, 
#line 1001
cudaErrorApiFailureBase = 10000
#line 1002
}; 
#endif
#line 1007 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1007
enum cudaChannelFormatKind { 
#line 1009
cudaChannelFormatKindSigned, 
#line 1010
cudaChannelFormatKindUnsigned, 
#line 1011
cudaChannelFormatKindFloat, 
#line 1012
cudaChannelFormatKindNone
#line 1013
}; 
#endif
#line 1018 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1018
struct cudaChannelFormatDesc { 
#line 1020
int x; 
#line 1021
int y; 
#line 1022
int z; 
#line 1023
int w; 
#line 1024
cudaChannelFormatKind f; 
#line 1025
}; 
#endif
#line 1030 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
typedef struct cudaArray *cudaArray_t; 
#line 1035
typedef const cudaArray *cudaArray_const_t; 
#line 1037
struct cudaArray; 
#line 1042
typedef struct cudaMipmappedArray *cudaMipmappedArray_t; 
#line 1047
typedef const cudaMipmappedArray *cudaMipmappedArray_const_t; 
#line 1049
struct cudaMipmappedArray; 
#line 1059
#if 0
#line 1059
struct cudaArraySparseProperties { 
#line 1060
struct { 
#line 1061
unsigned width; 
#line 1062
unsigned height; 
#line 1063
unsigned depth; 
#line 1064
} tileExtent; 
#line 1065
unsigned miptailFirstLevel; 
#line 1066
unsigned __int64 miptailSize; 
#line 1067
unsigned flags; 
#line 1068
unsigned reserved[4]; 
#line 1069
}; 
#endif
#line 1074 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1074
enum cudaMemoryType { 
#line 1076
cudaMemoryTypeUnregistered, 
#line 1077
cudaMemoryTypeHost, 
#line 1078
cudaMemoryTypeDevice, 
#line 1079
cudaMemoryTypeManaged
#line 1080
}; 
#endif
#line 1085 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1085
enum cudaMemcpyKind { 
#line 1087
cudaMemcpyHostToHost, 
#line 1088
cudaMemcpyHostToDevice, 
#line 1089
cudaMemcpyDeviceToHost, 
#line 1090
cudaMemcpyDeviceToDevice, 
#line 1091
cudaMemcpyDefault
#line 1092
}; 
#endif
#line 1099 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1099
struct cudaPitchedPtr { 
#line 1101
void *ptr; 
#line 1102
size_t pitch; 
#line 1103
size_t xsize; 
#line 1104
size_t ysize; 
#line 1105
}; 
#endif
#line 1112 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1112
struct cudaExtent { 
#line 1114
size_t width; 
#line 1115
size_t height; 
#line 1116
size_t depth; 
#line 1117
}; 
#endif
#line 1124 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1124
struct cudaPos { 
#line 1126
size_t x; 
#line 1127
size_t y; 
#line 1128
size_t z; 
#line 1129
}; 
#endif
#line 1134 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1134
struct cudaMemcpy3DParms { 
#line 1136
cudaArray_t srcArray; 
#line 1137
cudaPos srcPos; 
#line 1138
cudaPitchedPtr srcPtr; 
#line 1140
cudaArray_t dstArray; 
#line 1141
cudaPos dstPos; 
#line 1142
cudaPitchedPtr dstPtr; 
#line 1144
cudaExtent extent; 
#line 1145
cudaMemcpyKind kind; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 1146
}; 
#endif
#line 1151 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1151
struct cudaMemcpy3DPeerParms { 
#line 1153
cudaArray_t srcArray; 
#line 1154
cudaPos srcPos; 
#line 1155
cudaPitchedPtr srcPtr; 
#line 1156
int srcDevice; 
#line 1158
cudaArray_t dstArray; 
#line 1159
cudaPos dstPos; 
#line 1160
cudaPitchedPtr dstPtr; 
#line 1161
int dstDevice; 
#line 1163
cudaExtent extent; 
#line 1164
}; 
#endif
#line 1169 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1169
struct cudaMemsetParams { 
#line 1170
void *dst; 
#line 1171
size_t pitch; 
#line 1172
unsigned value; 
#line 1173
unsigned elementSize; 
#line 1174
size_t width; 
#line 1175
size_t height; 
#line 1176
}; 
#endif
#line 1181 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1181
enum cudaAccessProperty { 
#line 1182
cudaAccessPropertyNormal, 
#line 1183
cudaAccessPropertyStreaming, 
#line 1184
cudaAccessPropertyPersisting
#line 1185
}; 
#endif
#line 1198 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1198
struct cudaAccessPolicyWindow { 
#line 1199
void *base_ptr; 
#line 1200
size_t num_bytes; 
#line 1201
float hitRatio; 
#line 1202
cudaAccessProperty hitProp; 
#line 1203
cudaAccessProperty missProp; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 1204
}; 
#endif
#line 1216 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
typedef void (__stdcall *cudaHostFn_t)(void * userData); 
#line 1221
#if 0
#line 1221
struct cudaHostNodeParams { 
#line 1222
cudaHostFn_t fn; 
#line 1223
void *userData; 
#line 1224
}; 
#endif
#line 1229 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1229
enum cudaStreamCaptureStatus { 
#line 1230
cudaStreamCaptureStatusNone, 
#line 1231
cudaStreamCaptureStatusActive, 
#line 1232
cudaStreamCaptureStatusInvalidated
#line 1234
}; 
#endif
#line 1240 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1240
enum cudaStreamCaptureMode { 
#line 1241
cudaStreamCaptureModeGlobal, 
#line 1242
cudaStreamCaptureModeThreadLocal, 
#line 1243
cudaStreamCaptureModeRelaxed
#line 1244
}; 
#endif
#line 1246 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1246
enum cudaSynchronizationPolicy { 
#line 1247
cudaSyncPolicyAuto = 1, 
#line 1248
cudaSyncPolicySpin, 
#line 1249
cudaSyncPolicyYield, 
#line 1250
cudaSyncPolicyBlockingSync
#line 1251
}; 
#endif
#line 1256 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1256
enum cudaStreamAttrID { 
#line 1257
cudaStreamAttributeAccessPolicyWindow = 1, 
#line 1258
cudaStreamAttributeSynchronizationPolicy = 3
#line 1259
}; 
#endif
#line 1264 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1264
union cudaStreamAttrValue { 
#line 1265
cudaAccessPolicyWindow accessPolicyWindow; 
#line 1266
cudaSynchronizationPolicy syncPolicy; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 1267
}; 
#endif
#line 1272 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
struct cudaGraphicsResource; 
#line 1277
#if 0
#line 1277
enum cudaGraphicsRegisterFlags { 
#line 1279
cudaGraphicsRegisterFlagsNone, 
#line 1280
cudaGraphicsRegisterFlagsReadOnly, 
#line 1281
cudaGraphicsRegisterFlagsWriteDiscard, 
#line 1282
cudaGraphicsRegisterFlagsSurfaceLoadStore = 4, 
#line 1283
cudaGraphicsRegisterFlagsTextureGather = 8
#line 1284
}; 
#endif
#line 1289 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1289
enum cudaGraphicsMapFlags { 
#line 1291
cudaGraphicsMapFlagsNone, 
#line 1292
cudaGraphicsMapFlagsReadOnly, 
#line 1293
cudaGraphicsMapFlagsWriteDiscard
#line 1294
}; 
#endif
#line 1299 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1299
enum cudaGraphicsCubeFace { 
#line 1301
cudaGraphicsCubeFacePositiveX, 
#line 1302
cudaGraphicsCubeFaceNegativeX, 
#line 1303
cudaGraphicsCubeFacePositiveY, 
#line 1304
cudaGraphicsCubeFaceNegativeY, 
#line 1305
cudaGraphicsCubeFacePositiveZ, 
#line 1306
cudaGraphicsCubeFaceNegativeZ
#line 1307
}; 
#endif
#line 1312 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1312
enum cudaKernelNodeAttrID { 
#line 1313
cudaKernelNodeAttributeAccessPolicyWindow = 1, 
#line 1314
cudaKernelNodeAttributeCooperative
#line 1315
}; 
#endif
#line 1320 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1320
union cudaKernelNodeAttrValue { 
#line 1321
cudaAccessPolicyWindow accessPolicyWindow; 
#line 1322
int cooperative; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 1323
}; 
#endif
#line 1328 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1328
enum cudaResourceType { 
#line 1330
cudaResourceTypeArray, 
#line 1331
cudaResourceTypeMipmappedArray, 
#line 1332
cudaResourceTypeLinear, 
#line 1333
cudaResourceTypePitch2D
#line 1334
}; 
#endif
#line 1339 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1339
enum cudaResourceViewFormat { 
#line 1341
cudaResViewFormatNone, 
#line 1342
cudaResViewFormatUnsignedChar1, 
#line 1343
cudaResViewFormatUnsignedChar2, 
#line 1344
cudaResViewFormatUnsignedChar4, 
#line 1345
cudaResViewFormatSignedChar1, 
#line 1346
cudaResViewFormatSignedChar2, 
#line 1347
cudaResViewFormatSignedChar4, 
#line 1348
cudaResViewFormatUnsignedShort1, 
#line 1349
cudaResViewFormatUnsignedShort2, 
#line 1350
cudaResViewFormatUnsignedShort4, 
#line 1351
cudaResViewFormatSignedShort1, 
#line 1352
cudaResViewFormatSignedShort2, 
#line 1353
cudaResViewFormatSignedShort4, 
#line 1354
cudaResViewFormatUnsignedInt1, 
#line 1355
cudaResViewFormatUnsignedInt2, 
#line 1356
cudaResViewFormatUnsignedInt4, 
#line 1357
cudaResViewFormatSignedInt1, 
#line 1358
cudaResViewFormatSignedInt2, 
#line 1359
cudaResViewFormatSignedInt4, 
#line 1360
cudaResViewFormatHalf1, 
#line 1361
cudaResViewFormatHalf2, 
#line 1362
cudaResViewFormatHalf4, 
#line 1363
cudaResViewFormatFloat1, 
#line 1364
cudaResViewFormatFloat2, 
#line 1365
cudaResViewFormatFloat4, 
#line 1366
cudaResViewFormatUnsignedBlockCompressed1, 
#line 1367
cudaResViewFormatUnsignedBlockCompressed2, 
#line 1368
cudaResViewFormatUnsignedBlockCompressed3, 
#line 1369
cudaResViewFormatUnsignedBlockCompressed4, 
#line 1370
cudaResViewFormatSignedBlockCompressed4, 
#line 1371
cudaResViewFormatUnsignedBlockCompressed5, 
#line 1372
cudaResViewFormatSignedBlockCompressed5, 
#line 1373
cudaResViewFormatUnsignedBlockCompressed6H, 
#line 1374
cudaResViewFormatSignedBlockCompressed6H, 
#line 1375
cudaResViewFormatUnsignedBlockCompressed7
#line 1376
}; 
#endif
#line 1381 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1381
struct cudaResourceDesc { 
#line 1382
cudaResourceType resType; 
#line 1384
union { 
#line 1385
struct { 
#line 1386
cudaArray_t array; 
#line 1387
} array; 
#line 1388
struct { 
#line 1389
cudaMipmappedArray_t mipmap; 
#line 1390
} mipmap; 
#line 1391
struct { 
#line 1392
void *devPtr; 
#line 1393
cudaChannelFormatDesc desc; 
#line 1394
size_t sizeInBytes; 
#line 1395
} linear; 
#line 1396
struct { 
#line 1397
void *devPtr; 
#line 1398
cudaChannelFormatDesc desc; 
#line 1399
size_t width; 
#line 1400
size_t height; 
#line 1401
size_t pitchInBytes; 
#line 1402
} pitch2D; 
#line 1403
} res; 
#line 1404
}; 
#endif
#line 1409 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1409
struct cudaResourceViewDesc { 
#line 1411
cudaResourceViewFormat format; 
#line 1412
size_t width; 
#line 1413
size_t height; 
#line 1414
size_t depth; 
#line 1415
unsigned firstMipmapLevel; 
#line 1416
unsigned lastMipmapLevel; 
#line 1417
unsigned firstLayer; 
#line 1418
unsigned lastLayer; 
#line 1419
}; 
#endif
#line 1424 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1424
struct cudaPointerAttributes { 
#line 1430
cudaMemoryType type; 
#line 1441
int device; 
#line 1447
void *devicePointer; 
#line 1456
void *hostPointer; 
#line 1457
}; 
#endif
#line 1462 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1462
struct cudaFuncAttributes { 
#line 1469
size_t sharedSizeBytes; 
#line 1475
size_t constSizeBytes; 
#line 1480
size_t localSizeBytes; 
#line 1487
int maxThreadsPerBlock; 
#line 1492
int numRegs; 
#line 1499
int ptxVersion; 
#line 1506
int binaryVersion; 
#line 1512
int cacheModeCA; 
#line 1519
int maxDynamicSharedSizeBytes; 
#line 1528
int preferredShmemCarveout; 
#line 1529
}; 
#endif
#line 1534 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1534
enum cudaFuncAttribute { 
#line 1536
cudaFuncAttributeMaxDynamicSharedMemorySize = 8, 
#line 1537
cudaFuncAttributePreferredSharedMemoryCarveout, 
#line 1538
cudaFuncAttributeMax
#line 1539
}; 
#endif
#line 1544 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1544
enum cudaFuncCache { 
#line 1546
cudaFuncCachePreferNone, 
#line 1547
cudaFuncCachePreferShared, 
#line 1548
cudaFuncCachePreferL1, 
#line 1549
cudaFuncCachePreferEqual
#line 1550
}; 
#endif
#line 1556 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1556
enum cudaSharedMemConfig { 
#line 1558
cudaSharedMemBankSizeDefault, 
#line 1559
cudaSharedMemBankSizeFourByte, 
#line 1560
cudaSharedMemBankSizeEightByte
#line 1561
}; 
#endif
#line 1566 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1566
enum cudaSharedCarveout { 
#line 1567
cudaSharedmemCarveoutDefault = (-1), 
#line 1568
cudaSharedmemCarveoutMaxShared = 100, 
#line 1569
cudaSharedmemCarveoutMaxL1 = 0
#line 1570
}; 
#endif
#line 1575 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1575
enum cudaComputeMode { 
#line 1577
cudaComputeModeDefault, 
#line 1578
cudaComputeModeExclusive, 
#line 1579
cudaComputeModeProhibited, 
#line 1580
cudaComputeModeExclusiveProcess
#line 1581
}; 
#endif
#line 1586 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1586
enum cudaLimit { 
#line 1588
cudaLimitStackSize, 
#line 1589
cudaLimitPrintfFifoSize, 
#line 1590
cudaLimitMallocHeapSize, 
#line 1591
cudaLimitDevRuntimeSyncDepth, 
#line 1592
cudaLimitDevRuntimePendingLaunchCount, 
#line 1593
cudaLimitMaxL2FetchGranularity, 
#line 1594
cudaLimitPersistingL2CacheSize
#line 1595
}; 
#endif
#line 1600 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1600
enum cudaMemoryAdvise { 
#line 1602
cudaMemAdviseSetReadMostly = 1, 
#line 1603
cudaMemAdviseUnsetReadMostly, 
#line 1604
cudaMemAdviseSetPreferredLocation, 
#line 1605
cudaMemAdviseUnsetPreferredLocation, 
#line 1606
cudaMemAdviseSetAccessedBy, 
#line 1607
cudaMemAdviseUnsetAccessedBy
#line 1608
}; 
#endif
#line 1613 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1613
enum cudaMemRangeAttribute { 
#line 1615
cudaMemRangeAttributeReadMostly = 1, 
#line 1616
cudaMemRangeAttributePreferredLocation, 
#line 1617
cudaMemRangeAttributeAccessedBy, 
#line 1618
cudaMemRangeAttributeLastPrefetchLocation
#line 1619
}; 
#endif
#line 1624 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1624
enum cudaOutputMode { 
#line 1626
cudaKeyValuePair, 
#line 1627
cudaCSV
#line 1628
}; 
#endif
#line 1633 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1633
enum cudaDeviceAttr { 
#line 1635
cudaDevAttrMaxThreadsPerBlock = 1, 
#line 1636
cudaDevAttrMaxBlockDimX, 
#line 1637
cudaDevAttrMaxBlockDimY, 
#line 1638
cudaDevAttrMaxBlockDimZ, 
#line 1639
cudaDevAttrMaxGridDimX, 
#line 1640
cudaDevAttrMaxGridDimY, 
#line 1641
cudaDevAttrMaxGridDimZ, 
#line 1642
cudaDevAttrMaxSharedMemoryPerBlock, 
#line 1643
cudaDevAttrTotalConstantMemory, 
#line 1644
cudaDevAttrWarpSize, 
#line 1645
cudaDevAttrMaxPitch, 
#line 1646
cudaDevAttrMaxRegistersPerBlock, 
#line 1647
cudaDevAttrClockRate, 
#line 1648
cudaDevAttrTextureAlignment, 
#line 1649
cudaDevAttrGpuOverlap, 
#line 1650
cudaDevAttrMultiProcessorCount, 
#line 1651
cudaDevAttrKernelExecTimeout, 
#line 1652
cudaDevAttrIntegrated, 
#line 1653
cudaDevAttrCanMapHostMemory, 
#line 1654
cudaDevAttrComputeMode, 
#line 1655
cudaDevAttrMaxTexture1DWidth, 
#line 1656
cudaDevAttrMaxTexture2DWidth, 
#line 1657
cudaDevAttrMaxTexture2DHeight, 
#line 1658
cudaDevAttrMaxTexture3DWidth, 
#line 1659
cudaDevAttrMaxTexture3DHeight, 
#line 1660
cudaDevAttrMaxTexture3DDepth, 
#line 1661
cudaDevAttrMaxTexture2DLayeredWidth, 
#line 1662
cudaDevAttrMaxTexture2DLayeredHeight, 
#line 1663
cudaDevAttrMaxTexture2DLayeredLayers, 
#line 1664
cudaDevAttrSurfaceAlignment, 
#line 1665
cudaDevAttrConcurrentKernels, 
#line 1666
cudaDevAttrEccEnabled, 
#line 1667
cudaDevAttrPciBusId, 
#line 1668
cudaDevAttrPciDeviceId, 
#line 1669
cudaDevAttrTccDriver, 
#line 1670
cudaDevAttrMemoryClockRate, 
#line 1671
cudaDevAttrGlobalMemoryBusWidth, 
#line 1672
cudaDevAttrL2CacheSize, 
#line 1673
cudaDevAttrMaxThreadsPerMultiProcessor, 
#line 1674
cudaDevAttrAsyncEngineCount, 
#line 1675
cudaDevAttrUnifiedAddressing, 
#line 1676
cudaDevAttrMaxTexture1DLayeredWidth, 
#line 1677
cudaDevAttrMaxTexture1DLayeredLayers, 
#line 1678
cudaDevAttrMaxTexture2DGatherWidth = 45, 
#line 1679
cudaDevAttrMaxTexture2DGatherHeight, 
#line 1680
cudaDevAttrMaxTexture3DWidthAlt, 
#line 1681
cudaDevAttrMaxTexture3DHeightAlt, 
#line 1682
cudaDevAttrMaxTexture3DDepthAlt, 
#line 1683
cudaDevAttrPciDomainId, 
#line 1684
cudaDevAttrTexturePitchAlignment, 
#line 1685
cudaDevAttrMaxTextureCubemapWidth, 
#line 1686
cudaDevAttrMaxTextureCubemapLayeredWidth, 
#line 1687
cudaDevAttrMaxTextureCubemapLayeredLayers, 
#line 1688
cudaDevAttrMaxSurface1DWidth, 
#line 1689
cudaDevAttrMaxSurface2DWidth, 
#line 1690
cudaDevAttrMaxSurface2DHeight, 
#line 1691
cudaDevAttrMaxSurface3DWidth, 
#line 1692
cudaDevAttrMaxSurface3DHeight, 
#line 1693
cudaDevAttrMaxSurface3DDepth, 
#line 1694
cudaDevAttrMaxSurface1DLayeredWidth, 
#line 1695
cudaDevAttrMaxSurface1DLayeredLayers, 
#line 1696
cudaDevAttrMaxSurface2DLayeredWidth, 
#line 1697
cudaDevAttrMaxSurface2DLayeredHeight, 
#line 1698
cudaDevAttrMaxSurface2DLayeredLayers, 
#line 1699
cudaDevAttrMaxSurfaceCubemapWidth, 
#line 1700
cudaDevAttrMaxSurfaceCubemapLayeredWidth, 
#line 1701
cudaDevAttrMaxSurfaceCubemapLayeredLayers, 
#line 1702
cudaDevAttrMaxTexture1DLinearWidth, 
#line 1703
cudaDevAttrMaxTexture2DLinearWidth, 
#line 1704
cudaDevAttrMaxTexture2DLinearHeight, 
#line 1705
cudaDevAttrMaxTexture2DLinearPitch, 
#line 1706
cudaDevAttrMaxTexture2DMipmappedWidth, 
#line 1707
cudaDevAttrMaxTexture2DMipmappedHeight, 
#line 1708
cudaDevAttrComputeCapabilityMajor, 
#line 1709
cudaDevAttrComputeCapabilityMinor, 
#line 1710
cudaDevAttrMaxTexture1DMipmappedWidth, 
#line 1711
cudaDevAttrStreamPrioritiesSupported, 
#line 1712
cudaDevAttrGlobalL1CacheSupported, 
#line 1713
cudaDevAttrLocalL1CacheSupported, 
#line 1714
cudaDevAttrMaxSharedMemoryPerMultiprocessor, 
#line 1715
cudaDevAttrMaxRegistersPerMultiprocessor, 
#line 1716
cudaDevAttrManagedMemory, 
#line 1717
cudaDevAttrIsMultiGpuBoard, 
#line 1718
cudaDevAttrMultiGpuBoardGroupID, 
#line 1719
cudaDevAttrHostNativeAtomicSupported, 
#line 1720
cudaDevAttrSingleToDoublePrecisionPerfRatio, 
#line 1721
cudaDevAttrPageableMemoryAccess, 
#line 1722
cudaDevAttrConcurrentManagedAccess, 
#line 1723
cudaDevAttrComputePreemptionSupported, 
#line 1724
cudaDevAttrCanUseHostPointerForRegisteredMem, 
#line 1725
cudaDevAttrReserved92, 
#line 1726
cudaDevAttrReserved93, 
#line 1727
cudaDevAttrReserved94, 
#line 1728
cudaDevAttrCooperativeLaunch, 
#line 1729
cudaDevAttrCooperativeMultiDeviceLaunch, 
#line 1730
cudaDevAttrMaxSharedMemoryPerBlockOptin, 
#line 1731
cudaDevAttrCanFlushRemoteWrites, 
#line 1732
cudaDevAttrHostRegisterSupported, 
#line 1733
cudaDevAttrPageableMemoryAccessUsesHostPageTables, 
#line 1734
cudaDevAttrDirectManagedMemAccessFromHost, 
#line 1735
cudaDevAttrMaxBlocksPerMultiprocessor = 106, 
#line 1736
cudaDevAttrReservedSharedMemoryPerBlock = 111, 
#line 1737
cudaDevAttrSparseCudaArraySupported, 
#line 1738
cudaDevAttrHostRegisterReadOnlySupported
#line 1739
}; 
#endif
#line 1745 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1745
enum cudaDeviceP2PAttr { 
#line 1746
cudaDevP2PAttrPerformanceRank = 1, 
#line 1747
cudaDevP2PAttrAccessSupported, 
#line 1748
cudaDevP2PAttrNativeAtomicSupported, 
#line 1749
cudaDevP2PAttrCudaArrayAccessSupported
#line 1750
}; 
#endif
#line 1757 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1757
struct CUuuid_st { 
#line 1758
char bytes[16]; 
#line 1759
}; 
#endif
#line 1760 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
typedef CUuuid_st 
#line 1760
CUuuid; 
#endif
#line 1762 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
typedef CUuuid_st 
#line 1762
cudaUUID_t; 
#endif
#line 1767 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1767
struct cudaDeviceProp { 
#line 1769
char name[256]; 
#line 1770
cudaUUID_t uuid; 
#line 1771
char luid[8]; 
#line 1772
unsigned luidDeviceNodeMask; 
#line 1773
size_t totalGlobalMem; 
#line 1774
size_t sharedMemPerBlock; 
#line 1775
int regsPerBlock; 
#line 1776
int warpSize; 
#line 1777
size_t memPitch; 
#line 1778
int maxThreadsPerBlock; 
#line 1779
int maxThreadsDim[3]; 
#line 1780
int maxGridSize[3]; 
#line 1781
int clockRate; 
#line 1782
size_t totalConstMem; 
#line 1783
int major; 
#line 1784
int minor; 
#line 1785
size_t textureAlignment; 
#line 1786
size_t texturePitchAlignment; 
#line 1787
int deviceOverlap; 
#line 1788
int multiProcessorCount; 
#line 1789
int kernelExecTimeoutEnabled; 
#line 1790
int integrated; 
#line 1791
int canMapHostMemory; 
#line 1792
int computeMode; 
#line 1793
int maxTexture1D; 
#line 1794
int maxTexture1DMipmap; 
#line 1795
int maxTexture1DLinear; 
#line 1796
int maxTexture2D[2]; 
#line 1797
int maxTexture2DMipmap[2]; 
#line 1798
int maxTexture2DLinear[3]; 
#line 1799
int maxTexture2DGather[2]; 
#line 1800
int maxTexture3D[3]; 
#line 1801
int maxTexture3DAlt[3]; 
#line 1802
int maxTextureCubemap; 
#line 1803
int maxTexture1DLayered[2]; 
#line 1804
int maxTexture2DLayered[3]; 
#line 1805
int maxTextureCubemapLayered[2]; 
#line 1806
int maxSurface1D; 
#line 1807
int maxSurface2D[2]; 
#line 1808
int maxSurface3D[3]; 
#line 1809
int maxSurface1DLayered[2]; 
#line 1810
int maxSurface2DLayered[3]; 
#line 1811
int maxSurfaceCubemap; 
#line 1812
int maxSurfaceCubemapLayered[2]; 
#line 1813
size_t surfaceAlignment; 
#line 1814
int concurrentKernels; 
#line 1815
int ECCEnabled; 
#line 1816
int pciBusID; 
#line 1817
int pciDeviceID; 
#line 1818
int pciDomainID; 
#line 1819
int tccDriver; 
#line 1820
int asyncEngineCount; 
#line 1821
int unifiedAddressing; 
#line 1822
int memoryClockRate; 
#line 1823
int memoryBusWidth; 
#line 1824
int l2CacheSize; 
#line 1825
int persistingL2CacheMaxSize; 
#line 1826
int maxThreadsPerMultiProcessor; 
#line 1827
int streamPrioritiesSupported; 
#line 1828
int globalL1CacheSupported; 
#line 1829
int localL1CacheSupported; 
#line 1830
size_t sharedMemPerMultiprocessor; 
#line 1831
int regsPerMultiprocessor; 
#line 1832
int managedMemory; 
#line 1833
int isMultiGpuBoard; 
#line 1834
int multiGpuBoardGroupID; 
#line 1835
int hostNativeAtomicSupported; 
#line 1836
int singleToDoublePrecisionPerfRatio; 
#line 1837
int pageableMemoryAccess; 
#line 1838
int concurrentManagedAccess; 
#line 1839
int computePreemptionSupported; 
#line 1840
int canUseHostPointerForRegisteredMem; 
#line 1841
int cooperativeLaunch; 
#line 1842
int cooperativeMultiDeviceLaunch; 
#line 1843
size_t sharedMemPerBlockOptin; 
#line 1844
int pageableMemoryAccessUsesHostPageTables; 
#line 1845
int directManagedMemAccessFromHost; 
#line 1846
int maxBlocksPerMultiProcessor; 
#line 1847
int accessPolicyMaxWindowSize; 
#line 1848
size_t reservedSharedMemPerBlock; 
#line 1849
}; 
#endif
#line 1945 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
typedef 
#line 1942
struct cudaIpcEventHandle_st { 
#line 1944
char reserved[64]; 
#line 1945
} cudaIpcEventHandle_t; 
#endif
#line 1953 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
typedef 
#line 1950
struct cudaIpcMemHandle_st { 
#line 1952
char reserved[64]; 
#line 1953
} cudaIpcMemHandle_t; 
#endif
#line 1958 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 1958
enum cudaExternalMemoryHandleType { 
#line 1962
cudaExternalMemoryHandleTypeOpaqueFd = 1, 
#line 1966
cudaExternalMemoryHandleTypeOpaqueWin32, 
#line 1970
cudaExternalMemoryHandleTypeOpaqueWin32Kmt, 
#line 1974
cudaExternalMemoryHandleTypeD3D12Heap, 
#line 1978
cudaExternalMemoryHandleTypeD3D12Resource, 
#line 1982
cudaExternalMemoryHandleTypeD3D11Resource, 
#line 1986
cudaExternalMemoryHandleTypeD3D11ResourceKmt, 
#line 1990
cudaExternalMemoryHandleTypeNvSciBuf
#line 1991
}; 
#endif
#line 2033 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 2033
struct cudaExternalMemoryHandleDesc { 
#line 2037
cudaExternalMemoryHandleType type; 
#line 2038
union { 
#line 2044
int fd; 
#line 2060
struct { 
#line 2064
void *handle; 
#line 2069
const void *name; 
#line 2070
} win32; 
#line 2075
const void *nvSciBufObject; 
#line 2076
} handle; 
#line 2080
unsigned __int64 size; 
#line 2084
unsigned flags; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 2085
}; 
#endif
#line 2090 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 2090
struct cudaExternalMemoryBufferDesc { 
#line 2094
unsigned __int64 offset; 
#line 2098
unsigned __int64 size; 
#line 2102
unsigned flags; 
#line 2103
}; 
#endif
#line 2108 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 2108
struct cudaExternalMemoryMipmappedArrayDesc { 
#line 2113
unsigned __int64 offset; 
#line 2117
cudaChannelFormatDesc formatDesc; 
#line 2121
cudaExtent extent; 
#line 2126
unsigned flags; 
#line 2130
unsigned numLevels; 
#line 2131
}; 
#endif
#line 2136 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 2136
enum cudaExternalSemaphoreHandleType { 
#line 2140
cudaExternalSemaphoreHandleTypeOpaqueFd = 1, 
#line 2144
cudaExternalSemaphoreHandleTypeOpaqueWin32, 
#line 2148
cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt, 
#line 2152
cudaExternalSemaphoreHandleTypeD3D12Fence, 
#line 2156
cudaExternalSemaphoreHandleTypeD3D11Fence, 
#line 2160
cudaExternalSemaphoreHandleTypeNvSciSync, 
#line 2164
cudaExternalSemaphoreHandleTypeKeyedMutex, 
#line 2168
cudaExternalSemaphoreHandleTypeKeyedMutexKmt
#line 2169
}; 
#endif
#line 2174 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 2174
struct cudaExternalSemaphoreHandleDesc { 
#line 2178
cudaExternalSemaphoreHandleType type; 
#line 2179
union { 
#line 2184
int fd; 
#line 2199
struct { 
#line 2203
void *handle; 
#line 2208
const void *name; 
#line 2209
} win32; 
#line 2213
const void *nvSciSyncObj; 
#line 2214
} handle; 
#line 2218
unsigned flags; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
#line 2219
}; 
#endif
#line 2224 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 2224
struct cudaExternalSemaphoreSignalParams { 
#line 2225
struct { 
#line 2229
struct { 
#line 2233
unsigned __int64 value; 
#line 2234
} fence; 
#line 2235
union { 
#line 2240
void *fence; 
#line 2241
unsigned __int64 reserved; 
#line 2242
} nvSciSync; 
#line 2246
struct { 
#line 2250
unsigned __int64 key; 
#line 2251
} keyedMutex; 
#line 2252
} params; 
#line 2263
unsigned flags; 
#line 2264
}; 
#endif
#line 2269 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 2269
struct cudaExternalSemaphoreWaitParams { 
#line 2270
struct { 
#line 2274
struct { 
#line 2278
unsigned __int64 value; 
#line 2279
} fence; 
#line 2280
union { 
#line 2285
void *fence; 
#line 2286
unsigned __int64 reserved; 
#line 2287
} nvSciSync; 
#line 2291
struct { 
#line 2295
unsigned __int64 key; 
#line 2299
unsigned timeoutMs; 
#line 2300
} keyedMutex; 
#line 2301
} params; 
#line 2312
unsigned flags; 
#line 2313
}; 
#endif
#line 2325 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
typedef cudaError 
#line 2325
cudaError_t; 
#endif
#line 2330 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
typedef struct CUstream_st *
#line 2330
cudaStream_t; 
#endif
#line 2335 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
typedef struct CUevent_st *
#line 2335
cudaEvent_t; 
#endif
#line 2340 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
typedef cudaGraphicsResource *
#line 2340
cudaGraphicsResource_t; 
#endif
#line 2345 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
typedef cudaOutputMode 
#line 2345
cudaOutputMode_t; 
#endif
#line 2350 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
typedef struct CUexternalMemory_st *
#line 2350
cudaExternalMemory_t; 
#endif
#line 2355 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
typedef struct CUexternalSemaphore_st *
#line 2355
cudaExternalSemaphore_t; 
#endif
#line 2360 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
typedef struct CUgraph_st *
#line 2360
cudaGraph_t; 
#endif
#line 2365 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
typedef struct CUgraphNode_st *
#line 2365
cudaGraphNode_t; 
#endif
#line 2370 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
typedef struct CUfunc_st *
#line 2370
cudaFunction_t; 
#endif
#line 2375 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 2375
enum cudaCGScope { 
#line 2376
cudaCGScopeInvalid, 
#line 2377
cudaCGScopeGrid, 
#line 2378
cudaCGScopeMultiGrid
#line 2379
}; 
#endif
#line 2384 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 2384
struct cudaLaunchParams { 
#line 2386
void *func; 
#line 2387
dim3 gridDim; 
#line 2388
dim3 blockDim; 
#line 2389
void **args; 
#line 2390
size_t sharedMem; 
#line 2391
cudaStream_t stream; 
#line 2392
}; 
#endif
#line 2397 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 2397
struct cudaKernelNodeParams { 
#line 2398
void *func; 
#line 2399
dim3 gridDim; 
#line 2400
dim3 blockDim; 
#line 2401
unsigned sharedMemBytes; 
#line 2402
void **kernelParams; 
#line 2403
void **extra; 
#line 2404
}; 
#endif
#line 2409 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
#if 0
#line 2409
enum cudaGraphNodeType { 
#line 2410
cudaGraphNodeTypeKernel, 
#line 2411
cudaGraphNodeTypeMemcpy, 
#line 2412
cudaGraphNodeTypeMemset, 
#line 2413
cudaGraphNodeTypeHost, 
#line 2414
cudaGraphNodeTypeGraph, 
#line 2415
cudaGraphNodeTypeEmpty, 
#line 2416
cudaGraphNodeTypeWaitEvent, 
#line 2417
cudaGraphNodeTypeEventRecord, 
#line 2418
cudaGraphNodeTypeCount
#line 2419
}; 
#endif
#line 2424 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_types.h"
typedef struct CUgraphExec_st *cudaGraphExec_t; 
#line 2429
#if 0
#line 2429
enum cudaGraphExecUpdateResult { 
#line 2430
cudaGraphExecUpdateSuccess, 
#line 2431
cudaGraphExecUpdateError, 
#line 2432
cudaGraphExecUpdateErrorTopologyChanged, 
#line 2433
cudaGraphExecUpdateErrorNodeTypeChanged, 
#line 2434
cudaGraphExecUpdateErrorFunctionChanged, 
#line 2435
cudaGraphExecUpdateErrorParametersChanged, 
#line 2436
cudaGraphExecUpdateErrorNotSupported
#line 2437
}; 
#endif
#line 84 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_types.h"
#if 0
#line 84
enum cudaSurfaceBoundaryMode { 
#line 86
cudaBoundaryModeZero, 
#line 87
cudaBoundaryModeClamp, 
#line 88
cudaBoundaryModeTrap
#line 89
}; 
#endif
#line 94 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_types.h"
#if 0
#line 94
enum cudaSurfaceFormatMode { 
#line 96
cudaFormatModeForced, 
#line 97
cudaFormatModeAuto
#line 98
}; 
#endif
#line 103 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_types.h"
#if 0
#line 103
struct surfaceReference { 
#line 108
cudaChannelFormatDesc channelDesc; 
#line 109
}; 
#endif
#line 114 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_types.h"
#if 0
typedef unsigned __int64 
#line 114
cudaSurfaceObject_t; 
#endif
#line 84 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_types.h"
#if 0
#line 84
enum cudaTextureAddressMode { 
#line 86
cudaAddressModeWrap, 
#line 87
cudaAddressModeClamp, 
#line 88
cudaAddressModeMirror, 
#line 89
cudaAddressModeBorder
#line 90
}; 
#endif
#line 95 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_types.h"
#if 0
#line 95
enum cudaTextureFilterMode { 
#line 97
cudaFilterModePoint, 
#line 98
cudaFilterModeLinear
#line 99
}; 
#endif
#line 104 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_types.h"
#if 0
#line 104
enum cudaTextureReadMode { 
#line 106
cudaReadModeElementType, 
#line 107
cudaReadModeNormalizedFloat
#line 108
}; 
#endif
#line 113 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_types.h"
#if 0
#line 113
struct textureReference { 
#line 118
int normalized; 
#line 122
cudaTextureFilterMode filterMode; 
#line 126
cudaTextureAddressMode addressMode[3]; 
#line 130
cudaChannelFormatDesc channelDesc; 
#line 134
int sRGB; 
#line 138
unsigned maxAnisotropy; 
#line 142
cudaTextureFilterMode mipmapFilterMode; 
#line 146
float mipmapLevelBias; 
#line 150
float minMipmapLevelClamp; 
#line 154
float maxMipmapLevelClamp; 
#line 158
int disableTrilinearOptimization; 
#line 159
int __cudaReserved[14]; 
#line 160
}; 
#endif
#line 165 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_types.h"
#if 0
#line 165
struct cudaTextureDesc { 
#line 170
cudaTextureAddressMode addressMode[3]; 
#line 174
cudaTextureFilterMode filterMode; 
#line 178
cudaTextureReadMode readMode; 
#line 182
int sRGB; 
#line 186
float borderColor[4]; 
#line 190
int normalizedCoords; 
#line 194
unsigned maxAnisotropy; 
#line 198
cudaTextureFilterMode mipmapFilterMode; 
#line 202
float mipmapLevelBias; 
#line 206
float minMipmapLevelClamp; 
#line 210
float maxMipmapLevelClamp; 
#line 214
int disableTrilinearOptimization; 
#line 215
}; 
#endif
#line 220 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_types.h"
#if 0
typedef unsigned __int64 
#line 220
cudaTextureObject_t; 
#endif
#line 84 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\library_types.h"
typedef 
#line 54
enum cudaDataType_t { 
#line 56
CUDA_R_16F = 2, 
#line 57
CUDA_C_16F = 6, 
#line 58
CUDA_R_16BF = 14, 
#line 59
CUDA_C_16BF, 
#line 60
CUDA_R_32F = 0, 
#line 61
CUDA_C_32F = 4, 
#line 62
CUDA_R_64F = 1, 
#line 63
CUDA_C_64F = 5, 
#line 64
CUDA_R_4I = 16, 
#line 65
CUDA_C_4I, 
#line 66
CUDA_R_4U, 
#line 67
CUDA_C_4U, 
#line 68
CUDA_R_8I = 3, 
#line 69
CUDA_C_8I = 7, 
#line 70
CUDA_R_8U, 
#line 71
CUDA_C_8U, 
#line 72
CUDA_R_16I = 20, 
#line 73
CUDA_C_16I, 
#line 74
CUDA_R_16U, 
#line 75
CUDA_C_16U, 
#line 76
CUDA_R_32I = 10, 
#line 77
CUDA_C_32I, 
#line 78
CUDA_R_32U, 
#line 79
CUDA_C_32U, 
#line 80
CUDA_R_64I = 24, 
#line 81
CUDA_C_64I, 
#line 82
CUDA_R_64U, 
#line 83
CUDA_C_64U
#line 84
} cudaDataType; 
#line 92
typedef 
#line 87
enum libraryPropertyType_t { 
#line 89
MAJOR_VERSION, 
#line 90
MINOR_VERSION, 
#line 91
PATCH_LEVEL
#line 92
} libraryPropertyType; 
#line 121 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_device_runtime_api.h"
extern "C" {
#line 123
extern cudaError_t __stdcall cudaDeviceGetAttribute(int * value, cudaDeviceAttr attr, int device); 
#line 124
extern cudaError_t __stdcall cudaDeviceGetLimit(size_t * pValue, cudaLimit limit); 
#line 125
extern cudaError_t __stdcall cudaDeviceGetCacheConfig(cudaFuncCache * pCacheConfig); 
#line 126
extern cudaError_t __stdcall cudaDeviceGetSharedMemConfig(cudaSharedMemConfig * pConfig); 
#line 127
extern cudaError_t __stdcall cudaDeviceSynchronize(); 
#line 128
extern cudaError_t __stdcall cudaGetLastError(); 
#line 129
extern cudaError_t __stdcall cudaPeekAtLastError(); 
#line 130
extern const char *__stdcall cudaGetErrorString(cudaError_t error); 
#line 131
extern const char *__stdcall cudaGetErrorName(cudaError_t error); 
#line 132
extern cudaError_t __stdcall cudaGetDeviceCount(int * count); 
#line 133
extern cudaError_t __stdcall cudaGetDevice(int * device); 
#line 134
extern cudaError_t __stdcall cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned flags); 
#line 135
extern cudaError_t __stdcall cudaStreamDestroy(cudaStream_t stream); 
#line 136
extern cudaError_t __stdcall cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
#line 137
extern cudaError_t __stdcall cudaStreamWaitEvent_ptsz(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
#line 138
extern cudaError_t __stdcall cudaEventCreateWithFlags(cudaEvent_t * event, unsigned flags); 
#line 139
extern cudaError_t __stdcall cudaEventRecord(cudaEvent_t event, cudaStream_t stream); 
#line 140
extern cudaError_t __stdcall cudaEventRecord_ptsz(cudaEvent_t event, cudaStream_t stream); 
#line 141
extern cudaError_t __stdcall cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned flags); 
#line 142
extern cudaError_t __stdcall cudaEventRecordWithFlags_ptsz(cudaEvent_t event, cudaStream_t stream, unsigned flags); 
#line 143
extern cudaError_t __stdcall cudaEventDestroy(cudaEvent_t event); 
#line 144
extern cudaError_t __stdcall cudaFuncGetAttributes(cudaFuncAttributes * attr, const void * func); 
#line 145
extern cudaError_t __stdcall cudaFree(void * devPtr); 
#line 146
extern cudaError_t __stdcall cudaMalloc(void ** devPtr, size_t size); 
#line 147
extern cudaError_t __stdcall cudaMemcpyAsync(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream); 
#line 148
extern cudaError_t __stdcall cudaMemcpyAsync_ptsz(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream); 
#line 149
extern cudaError_t __stdcall cudaMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream); 
#line 150
extern cudaError_t __stdcall cudaMemcpy2DAsync_ptsz(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream); 
#line 151
extern cudaError_t __stdcall cudaMemcpy3DAsync(const cudaMemcpy3DParms * p, cudaStream_t stream); 
#line 152
extern cudaError_t __stdcall cudaMemcpy3DAsync_ptsz(const cudaMemcpy3DParms * p, cudaStream_t stream); 
#line 153
extern cudaError_t __stdcall cudaMemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream); 
#line 154
extern cudaError_t __stdcall cudaMemsetAsync_ptsz(void * devPtr, int value, size_t count, cudaStream_t stream); 
#line 155
extern cudaError_t __stdcall cudaMemset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream); 
#line 156
extern cudaError_t __stdcall cudaMemset2DAsync_ptsz(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream); 
#line 157
extern cudaError_t __stdcall cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream); 
#line 158
extern cudaError_t __stdcall cudaMemset3DAsync_ptsz(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream); 
#line 159
extern cudaError_t __stdcall cudaRuntimeGetVersion(int * runtimeVersion); 
#line 180
extern void *__stdcall cudaGetParameterBuffer(size_t alignment, size_t size); 
#line 208
extern void *__stdcall cudaGetParameterBufferV2(void * func, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize); 
#line 209
extern cudaError_t __stdcall cudaLaunchDevice_ptsz(void * func, void * parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream); 
#line 210
extern cudaError_t __stdcall cudaLaunchDeviceV2_ptsz(void * parameterBuffer, cudaStream_t stream); 
#line 228 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_device_runtime_api.h"
extern cudaError_t __stdcall cudaLaunchDevice(void * func, void * parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream); 
#line 229
extern cudaError_t __stdcall cudaLaunchDeviceV2(void * parameterBuffer, cudaStream_t stream); 
#line 232 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_device_runtime_api.h"
extern cudaError_t __stdcall cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * func, int blockSize, size_t dynamicSmemSize); 
#line 233
extern cudaError_t __stdcall cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int blockSize, size_t dynamicSmemSize, unsigned flags); 
#line 235
extern unsigned __int64 __stdcall cudaCGGetIntrinsicHandle(cudaCGScope scope); 
#line 236
extern cudaError_t __stdcall cudaCGSynchronize(unsigned __int64 handle, unsigned flags); 
#line 237
extern cudaError_t __stdcall cudaCGSynchronizeGrid(unsigned __int64 handle, unsigned flags); 
#line 238
extern cudaError_t __stdcall cudaCGGetSize(unsigned * numThreads, unsigned * numGrids, unsigned __int64 handle); 
#line 239
extern cudaError_t __stdcall cudaCGGetRank(unsigned * threadRank, unsigned * gridRank, unsigned __int64 handle); 
#line 240
}
#line 242
template< class T> static __inline cudaError_t cudaMalloc(T ** devPtr, size_t size); 
#line 243
template< class T> static __inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes * attr, T * entry); 
#line 244
template< class T> static __inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, T func, int blockSize, size_t dynamicSmemSize); 
#line 245
template< class T> static __inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, T func, int blockSize, size_t dynamicSmemSize, unsigned flags); 
#line 256 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_runtime_api.h"
extern "C" {
#line 291 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaDeviceReset(); 
#line 312
extern cudaError_t __stdcall cudaDeviceSynchronize(); 
#line 399
extern cudaError_t __stdcall cudaDeviceSetLimit(cudaLimit limit, size_t value); 
#line 434
extern cudaError_t __stdcall cudaDeviceGetLimit(size_t * pValue, cudaLimit limit); 
#line 457
extern cudaError_t __stdcall cudaDeviceGetTexture1DLinearMaxWidth(size_t * maxWidthInElements, const cudaChannelFormatDesc * fmtDesc, int device); 
#line 491 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaDeviceGetCacheConfig(cudaFuncCache * pCacheConfig); 
#line 528
extern cudaError_t __stdcall cudaDeviceGetStreamPriorityRange(int * leastPriority, int * greatestPriority); 
#line 572
extern cudaError_t __stdcall cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig); 
#line 603
extern cudaError_t __stdcall cudaDeviceGetSharedMemConfig(cudaSharedMemConfig * pConfig); 
#line 647
extern cudaError_t __stdcall cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config); 
#line 674
extern cudaError_t __stdcall cudaDeviceGetByPCIBusId(int * device, const char * pciBusId); 
#line 704
extern cudaError_t __stdcall cudaDeviceGetPCIBusId(char * pciBusId, int len, int device); 
#line 752
extern cudaError_t __stdcall cudaIpcGetEventHandle(cudaIpcEventHandle_t * handle, cudaEvent_t event); 
#line 793
extern cudaError_t __stdcall cudaIpcOpenEventHandle(cudaEvent_t * event, cudaIpcEventHandle_t handle); 
#line 836
extern cudaError_t __stdcall cudaIpcGetMemHandle(cudaIpcMemHandle_t * handle, void * devPtr); 
#line 900
extern cudaError_t __stdcall cudaIpcOpenMemHandle(void ** devPtr, cudaIpcMemHandle_t handle, unsigned flags); 
#line 936
extern cudaError_t __stdcall cudaIpcCloseMemHandle(void * devPtr); 
#line 978
__declspec(deprecated) extern cudaError_t __stdcall cudaThreadExit(); 
#line 1004
__declspec(deprecated) extern cudaError_t __stdcall cudaThreadSynchronize(); 
#line 1053
__declspec(deprecated) extern cudaError_t __stdcall cudaThreadSetLimit(cudaLimit limit, size_t value); 
#line 1086
__declspec(deprecated) extern cudaError_t __stdcall cudaThreadGetLimit(size_t * pValue, cudaLimit limit); 
#line 1122
__declspec(deprecated) extern cudaError_t __stdcall cudaThreadGetCacheConfig(cudaFuncCache * pCacheConfig); 
#line 1169
__declspec(deprecated) extern cudaError_t __stdcall cudaThreadSetCacheConfig(cudaFuncCache cacheConfig); 
#line 1229
extern cudaError_t __stdcall cudaGetLastError(); 
#line 1276
extern cudaError_t __stdcall cudaPeekAtLastError(); 
#line 1292
extern const char *__stdcall cudaGetErrorName(cudaError_t error); 
#line 1308
extern const char *__stdcall cudaGetErrorString(cudaError_t error); 
#line 1336
extern cudaError_t __stdcall cudaGetDeviceCount(int * count); 
#line 1614
extern cudaError_t __stdcall cudaGetDeviceProperties(cudaDeviceProp * prop, int device); 
#line 1807
extern cudaError_t __stdcall cudaDeviceGetAttribute(int * value, cudaDeviceAttr attr, int device); 
#line 1856
extern cudaError_t __stdcall cudaDeviceGetNvSciSyncAttributes(void * nvSciSyncAttrList, int device, int flags); 
#line 1896
extern cudaError_t __stdcall cudaDeviceGetP2PAttribute(int * value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice); 
#line 1917
extern cudaError_t __stdcall cudaChooseDevice(int * device, const cudaDeviceProp * prop); 
#line 1954
extern cudaError_t __stdcall cudaSetDevice(int device); 
#line 1975
extern cudaError_t __stdcall cudaGetDevice(int * device); 
#line 2006
extern cudaError_t __stdcall cudaSetValidDevices(int * device_arr, int len); 
#line 2075
extern cudaError_t __stdcall cudaSetDeviceFlags(unsigned flags); 
#line 2121
extern cudaError_t __stdcall cudaGetDeviceFlags(unsigned * flags); 
#line 2161
extern cudaError_t __stdcall cudaStreamCreate(cudaStream_t * pStream); 
#line 2193
extern cudaError_t __stdcall cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned flags); 
#line 2239
extern cudaError_t __stdcall cudaStreamCreateWithPriority(cudaStream_t * pStream, unsigned flags, int priority); 
#line 2266
extern cudaError_t __stdcall cudaStreamGetPriority(cudaStream_t hStream, int * priority); 
#line 2291
extern cudaError_t __stdcall cudaStreamGetFlags(cudaStream_t hStream, unsigned * flags); 
#line 2306
extern cudaError_t __stdcall cudaCtxResetPersistingL2Cache(); 
#line 2326
extern cudaError_t __stdcall cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src); 
#line 2347
extern cudaError_t __stdcall cudaStreamGetAttribute(cudaStream_t hStream, cudaStreamAttrID attr, cudaStreamAttrValue * value_out); 
#line 2371
extern cudaError_t __stdcall cudaStreamSetAttribute(cudaStream_t hStream, cudaStreamAttrID attr, const cudaStreamAttrValue * value); 
#line 2405
extern cudaError_t __stdcall cudaStreamDestroy(cudaStream_t stream); 
#line 2436
extern cudaError_t __stdcall cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags = 0); 
#line 2444
typedef void (__stdcall *cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void * userData); 
#line 2511
extern cudaError_t __stdcall cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void * userData, unsigned flags); 
#line 2535
extern cudaError_t __stdcall cudaStreamSynchronize(cudaStream_t stream); 
#line 2560
extern cudaError_t __stdcall cudaStreamQuery(cudaStream_t stream); 
#line 2643
extern cudaError_t __stdcall cudaStreamAttachMemAsync(cudaStream_t stream, void * devPtr, size_t length = 0, unsigned flags = 4); 
#line 2679
extern cudaError_t __stdcall cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode); 
#line 2730
extern cudaError_t __stdcall cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode * mode); 
#line 2758
extern cudaError_t __stdcall cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t * pGraph); 
#line 2796
extern cudaError_t __stdcall cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus * pCaptureStatus); 
#line 2824
extern cudaError_t __stdcall cudaStreamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus * pCaptureStatus, unsigned __int64 * pId); 
#line 2861
extern cudaError_t __stdcall cudaEventCreate(cudaEvent_t * event); 
#line 2898
extern cudaError_t __stdcall cudaEventCreateWithFlags(cudaEvent_t * event, unsigned flags); 
#line 2938
extern cudaError_t __stdcall cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0); 
#line 2985
extern cudaError_t __stdcall cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream = 0, unsigned flags = 0); 
#line 3017 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaEventQuery(cudaEvent_t event); 
#line 3047
extern cudaError_t __stdcall cudaEventSynchronize(cudaEvent_t event); 
#line 3076
extern cudaError_t __stdcall cudaEventDestroy(cudaEvent_t event); 
#line 3119
extern cudaError_t __stdcall cudaEventElapsedTime(float * ms, cudaEvent_t start, cudaEvent_t end); 
#line 3296
extern cudaError_t __stdcall cudaImportExternalMemory(cudaExternalMemory_t * extMem_out, const cudaExternalMemoryHandleDesc * memHandleDesc); 
#line 3350
extern cudaError_t __stdcall cudaExternalMemoryGetMappedBuffer(void ** devPtr, cudaExternalMemory_t extMem, const cudaExternalMemoryBufferDesc * bufferDesc); 
#line 3409
extern cudaError_t __stdcall cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t * mipmap, cudaExternalMemory_t extMem, const cudaExternalMemoryMipmappedArrayDesc * mipmapDesc); 
#line 3433
extern cudaError_t __stdcall cudaDestroyExternalMemory(cudaExternalMemory_t extMem); 
#line 3564
extern cudaError_t __stdcall cudaImportExternalSemaphore(cudaExternalSemaphore_t * extSem_out, const cudaExternalSemaphoreHandleDesc * semHandleDesc); 
#line 3629
extern cudaError_t __stdcall cudaSignalExternalSemaphoresAsync(const cudaExternalSemaphore_t * extSemArray, const cudaExternalSemaphoreSignalParams * paramsArray, unsigned numExtSems, cudaStream_t stream = 0); 
#line 3703
extern cudaError_t __stdcall cudaWaitExternalSemaphoresAsync(const cudaExternalSemaphore_t * extSemArray, const cudaExternalSemaphoreWaitParams * paramsArray, unsigned numExtSems, cudaStream_t stream = 0); 
#line 3726
extern cudaError_t __stdcall cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem); 
#line 3792
extern cudaError_t __stdcall cudaLaunchKernel(const void * func, dim3 gridDim, dim3 blockDim, void ** args, size_t sharedMem, cudaStream_t stream); 
#line 3849
extern cudaError_t __stdcall cudaLaunchCooperativeKernel(const void * func, dim3 gridDim, dim3 blockDim, void ** args, size_t sharedMem, cudaStream_t stream); 
#line 3948
extern cudaError_t __stdcall cudaLaunchCooperativeKernelMultiDevice(cudaLaunchParams * launchParamsList, unsigned numDevices, unsigned flags = 0); 
#line 3997
extern cudaError_t __stdcall cudaFuncSetCacheConfig(const void * func, cudaFuncCache cacheConfig); 
#line 4052
extern cudaError_t __stdcall cudaFuncSetSharedMemConfig(const void * func, cudaSharedMemConfig config); 
#line 4087
extern cudaError_t __stdcall cudaFuncGetAttributes(cudaFuncAttributes * attr, const void * func); 
#line 4126
extern cudaError_t __stdcall cudaFuncSetAttribute(const void * func, cudaFuncAttribute attr, int value); 
#line 4150
__declspec(deprecated) extern cudaError_t __stdcall cudaSetDoubleForDevice(double * d); 
#line 4174
__declspec(deprecated) extern cudaError_t __stdcall cudaSetDoubleForHost(double * d); 
#line 4240
extern cudaError_t __stdcall cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void * userData); 
#line 4297
extern cudaError_t __stdcall cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * func, int blockSize, size_t dynamicSMemSize); 
#line 4326
extern cudaError_t __stdcall cudaOccupancyAvailableDynamicSMemPerBlock(size_t * dynamicSmemSize, const void * func, int numBlocks, int blockSize); 
#line 4371
extern cudaError_t __stdcall cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int blockSize, size_t dynamicSMemSize, unsigned flags); 
#line 4491
extern cudaError_t __stdcall cudaMallocManaged(void ** devPtr, size_t size, unsigned flags = 1); 
#line 4522
extern cudaError_t __stdcall cudaMalloc(void ** devPtr, size_t size); 
#line 4555
extern cudaError_t __stdcall cudaMallocHost(void ** ptr, size_t size); 
#line 4598
extern cudaError_t __stdcall cudaMallocPitch(void ** devPtr, size_t * pitch, size_t width, size_t height); 
#line 4647
extern cudaError_t __stdcall cudaMallocArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, size_t width, size_t height = 0, unsigned flags = 0); 
#line 4676
extern cudaError_t __stdcall cudaFree(void * devPtr); 
#line 4699
extern cudaError_t __stdcall cudaFreeHost(void * ptr); 
#line 4722
extern cudaError_t __stdcall cudaFreeArray(cudaArray_t array); 
#line 4745
extern cudaError_t __stdcall cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray); 
#line 4811
extern cudaError_t __stdcall cudaHostAlloc(void ** pHost, size_t size, unsigned flags); 
#line 4904
extern cudaError_t __stdcall cudaHostRegister(void * ptr, size_t size, unsigned flags); 
#line 4927
extern cudaError_t __stdcall cudaHostUnregister(void * ptr); 
#line 4972
extern cudaError_t __stdcall cudaHostGetDevicePointer(void ** pDevice, void * pHost, unsigned flags); 
#line 4994
extern cudaError_t __stdcall cudaHostGetFlags(unsigned * pFlags, void * pHost); 
#line 5033
extern cudaError_t __stdcall cudaMalloc3D(cudaPitchedPtr * pitchedDevPtr, cudaExtent extent); 
#line 5175
extern cudaError_t __stdcall cudaMalloc3DArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, cudaExtent extent, unsigned flags = 0); 
#line 5317
extern cudaError_t __stdcall cudaMallocMipmappedArray(cudaMipmappedArray_t * mipmappedArray, const cudaChannelFormatDesc * desc, cudaExtent extent, unsigned numLevels, unsigned flags = 0); 
#line 5350
extern cudaError_t __stdcall cudaGetMipmappedArrayLevel(cudaArray_t * levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned level); 
#line 5455
extern cudaError_t __stdcall cudaMemcpy3D(const cudaMemcpy3DParms * p); 
#line 5486
extern cudaError_t __stdcall cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms * p); 
#line 5604
extern cudaError_t __stdcall cudaMemcpy3DAsync(const cudaMemcpy3DParms * p, cudaStream_t stream = 0); 
#line 5630
extern cudaError_t __stdcall cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms * p, cudaStream_t stream = 0); 
#line 5652
extern cudaError_t __stdcall cudaMemGetInfo(size_t * free, size_t * total); 
#line 5678
extern cudaError_t __stdcall cudaArrayGetInfo(cudaChannelFormatDesc * desc, cudaExtent * extent, unsigned * flags, cudaArray_t array); 
#line 5706
extern cudaError_t __stdcall cudaArrayGetSparseProperties(cudaArraySparseProperties * sparseProperties, cudaArray_t array); 
#line 5736 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaMipmappedArrayGetSparseProperties(cudaArraySparseProperties * sparseProperties, cudaMipmappedArray_t mipmap); 
#line 5781 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaMemcpy(void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
#line 5816
extern cudaError_t __stdcall cudaMemcpyPeer(void * dst, int dstDevice, const void * src, int srcDevice, size_t count); 
#line 5865
extern cudaError_t __stdcall cudaMemcpy2D(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind); 
#line 5915
extern cudaError_t __stdcall cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind); 
#line 5965
extern cudaError_t __stdcall cudaMemcpy2DFromArray(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind); 
#line 6012
extern cudaError_t __stdcall cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice); 
#line 6055
extern cudaError_t __stdcall cudaMemcpyToSymbol(const void * symbol, const void * src, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyHostToDevice); 
#line 6098
extern cudaError_t __stdcall cudaMemcpyFromSymbol(void * dst, const void * symbol, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyDeviceToHost); 
#line 6155
extern cudaError_t __stdcall cudaMemcpyAsync(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 6190
extern cudaError_t __stdcall cudaMemcpyPeerAsync(void * dst, int dstDevice, const void * src, int srcDevice, size_t count, cudaStream_t stream = 0); 
#line 6253
extern cudaError_t __stdcall cudaMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 6311
extern cudaError_t __stdcall cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 6368
extern cudaError_t __stdcall cudaMemcpy2DFromArrayAsync(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 6419
extern cudaError_t __stdcall cudaMemcpyToSymbolAsync(const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 6470
extern cudaError_t __stdcall cudaMemcpyFromSymbolAsync(void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 6499
extern cudaError_t __stdcall cudaMemset(void * devPtr, int value, size_t count); 
#line 6533
extern cudaError_t __stdcall cudaMemset2D(void * devPtr, size_t pitch, int value, size_t width, size_t height); 
#line 6579
extern cudaError_t __stdcall cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent); 
#line 6615
extern cudaError_t __stdcall cudaMemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream = 0); 
#line 6656
extern cudaError_t __stdcall cudaMemset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream = 0); 
#line 6709
extern cudaError_t __stdcall cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream = 0); 
#line 6737
extern cudaError_t __stdcall cudaGetSymbolAddress(void ** devPtr, const void * symbol); 
#line 6764
extern cudaError_t __stdcall cudaGetSymbolSize(size_t * size, const void * symbol); 
#line 6834
extern cudaError_t __stdcall cudaMemPrefetchAsync(const void * devPtr, size_t count, int dstDevice, cudaStream_t stream = 0); 
#line 6950
extern cudaError_t __stdcall cudaMemAdvise(const void * devPtr, size_t count, cudaMemoryAdvise advice, int device); 
#line 7009
extern cudaError_t __stdcall cudaMemRangeGetAttribute(void * data, size_t dataSize, cudaMemRangeAttribute attribute, const void * devPtr, size_t count); 
#line 7048
extern cudaError_t __stdcall cudaMemRangeGetAttributes(void ** data, size_t * dataSizes, cudaMemRangeAttribute * attributes, size_t numAttributes, const void * devPtr, size_t count); 
#line 7108
__declspec(deprecated) extern cudaError_t __stdcall cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, cudaMemcpyKind kind); 
#line 7150
__declspec(deprecated) extern cudaError_t __stdcall cudaMemcpyFromArray(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind); 
#line 7193
__declspec(deprecated) extern cudaError_t __stdcall cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice); 
#line 7244
__declspec(deprecated) extern cudaError_t __stdcall cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 7294
__declspec(deprecated) extern cudaError_t __stdcall cudaMemcpyFromArrayAsync(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
#line 7449
extern cudaError_t __stdcall cudaPointerGetAttributes(cudaPointerAttributes * attributes, const void * ptr); 
#line 7490
extern cudaError_t __stdcall cudaDeviceCanAccessPeer(int * canAccessPeer, int device, int peerDevice); 
#line 7532
extern cudaError_t __stdcall cudaDeviceEnablePeerAccess(int peerDevice, unsigned flags); 
#line 7554
extern cudaError_t __stdcall cudaDeviceDisablePeerAccess(int peerDevice); 
#line 7618
extern cudaError_t __stdcall cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource); 
#line 7653
extern cudaError_t __stdcall cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned flags); 
#line 7692
extern cudaError_t __stdcall cudaGraphicsMapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream = 0); 
#line 7727
extern cudaError_t __stdcall cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream = 0); 
#line 7759
extern cudaError_t __stdcall cudaGraphicsResourceGetMappedPointer(void ** devPtr, size_t * size, cudaGraphicsResource_t resource); 
#line 7797
extern cudaError_t __stdcall cudaGraphicsSubResourceGetMappedArray(cudaArray_t * array, cudaGraphicsResource_t resource, unsigned arrayIndex, unsigned mipLevel); 
#line 7826
extern cudaError_t __stdcall cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t * mipmappedArray, cudaGraphicsResource_t resource); 
#line 7897
__declspec(deprecated) extern cudaError_t __stdcall cudaBindTexture(size_t * offset, const textureReference * texref, const void * devPtr, const cudaChannelFormatDesc * desc, size_t size = 4294967295U); 
#line 7956
__declspec(deprecated) extern cudaError_t __stdcall cudaBindTexture2D(size_t * offset, const textureReference * texref, const void * devPtr, const cudaChannelFormatDesc * desc, size_t width, size_t height, size_t pitch); 
#line 7994
__declspec(deprecated) extern cudaError_t __stdcall cudaBindTextureToArray(const textureReference * texref, cudaArray_const_t array, const cudaChannelFormatDesc * desc); 
#line 8034
__declspec(deprecated) extern cudaError_t __stdcall cudaBindTextureToMipmappedArray(const textureReference * texref, cudaMipmappedArray_const_t mipmappedArray, const cudaChannelFormatDesc * desc); 
#line 8060
__declspec(deprecated) extern cudaError_t __stdcall cudaUnbindTexture(const textureReference * texref); 
#line 8089
__declspec(deprecated) extern cudaError_t __stdcall cudaGetTextureAlignmentOffset(size_t * offset, const textureReference * texref); 
#line 8119
__declspec(deprecated) extern cudaError_t __stdcall cudaGetTextureReference(const textureReference ** texref, const void * symbol); 
#line 8164
__declspec(deprecated) extern cudaError_t __stdcall cudaBindSurfaceToArray(const surfaceReference * surfref, cudaArray_const_t array, const cudaChannelFormatDesc * desc); 
#line 8189
__declspec(deprecated) extern cudaError_t __stdcall cudaGetSurfaceReference(const surfaceReference ** surfref, const void * symbol); 
#line 8224
extern cudaError_t __stdcall cudaGetChannelDesc(cudaChannelFormatDesc * desc, cudaArray_const_t array); 
#line 8254
extern cudaChannelFormatDesc __stdcall cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f); 
#line 8472
extern cudaError_t __stdcall cudaCreateTextureObject(cudaTextureObject_t * pTexObject, const cudaResourceDesc * pResDesc, const cudaTextureDesc * pTexDesc, const cudaResourceViewDesc * pResViewDesc); 
#line 8492
extern cudaError_t __stdcall cudaDestroyTextureObject(cudaTextureObject_t texObject); 
#line 8512
extern cudaError_t __stdcall cudaGetTextureObjectResourceDesc(cudaResourceDesc * pResDesc, cudaTextureObject_t texObject); 
#line 8532
extern cudaError_t __stdcall cudaGetTextureObjectTextureDesc(cudaTextureDesc * pTexDesc, cudaTextureObject_t texObject); 
#line 8553
extern cudaError_t __stdcall cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc * pResViewDesc, cudaTextureObject_t texObject); 
#line 8598
extern cudaError_t __stdcall cudaCreateSurfaceObject(cudaSurfaceObject_t * pSurfObject, const cudaResourceDesc * pResDesc); 
#line 8618
extern cudaError_t __stdcall cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject); 
#line 8637
extern cudaError_t __stdcall cudaGetSurfaceObjectResourceDesc(cudaResourceDesc * pResDesc, cudaSurfaceObject_t surfObject); 
#line 8671
extern cudaError_t __stdcall cudaDriverGetVersion(int * driverVersion); 
#line 8696
extern cudaError_t __stdcall cudaRuntimeGetVersion(int * runtimeVersion); 
#line 8743
extern cudaError_t __stdcall cudaGraphCreate(cudaGraph_t * pGraph, unsigned flags); 
#line 8840
extern cudaError_t __stdcall cudaGraphAddKernelNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaKernelNodeParams * pNodeParams); 
#line 8873
extern cudaError_t __stdcall cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams * pNodeParams); 
#line 8898
extern cudaError_t __stdcall cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const cudaKernelNodeParams * pNodeParams); 
#line 8918
extern cudaError_t __stdcall cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hSrc, cudaGraphNode_t hDst); 
#line 8941
extern cudaError_t __stdcall cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, cudaKernelNodeAttrValue * value_out); 
#line 8965
extern cudaError_t __stdcall cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode, cudaKernelNodeAttrID attr, const cudaKernelNodeAttrValue * value); 
#line 9015
extern cudaError_t __stdcall cudaGraphAddMemcpyNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaMemcpy3DParms * pCopyParams); 
#line 9074
extern cudaError_t __stdcall cudaGraphAddMemcpyNodeToSymbol(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind); 
#line 9143 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphAddMemcpyNodeFromSymbol(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind); 
#line 9211 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphAddMemcpyNode1D(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
#line 9243 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, cudaMemcpy3DParms * pNodeParams); 
#line 9269
extern cudaError_t __stdcall cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const cudaMemcpy3DParms * pNodeParams); 
#line 9308
extern cudaError_t __stdcall cudaGraphMemcpyNodeSetParamsToSymbol(cudaGraphNode_t node, const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind); 
#line 9354 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphMemcpyNodeSetParamsFromSymbol(cudaGraphNode_t node, void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind); 
#line 9400 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
#line 9447 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphAddMemsetNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaMemsetParams * pMemsetParams); 
#line 9470
extern cudaError_t __stdcall cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, cudaMemsetParams * pNodeParams); 
#line 9493
extern cudaError_t __stdcall cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const cudaMemsetParams * pNodeParams); 
#line 9534
extern cudaError_t __stdcall cudaGraphAddHostNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, const cudaHostNodeParams * pNodeParams); 
#line 9557
extern cudaError_t __stdcall cudaGraphHostNodeGetParams(cudaGraphNode_t node, cudaHostNodeParams * pNodeParams); 
#line 9580
extern cudaError_t __stdcall cudaGraphHostNodeSetParams(cudaGraphNode_t node, const cudaHostNodeParams * pNodeParams); 
#line 9618
extern cudaError_t __stdcall cudaGraphAddChildGraphNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaGraph_t childGraph); 
#line 9642
extern cudaError_t __stdcall cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t * pGraph); 
#line 9679
extern cudaError_t __stdcall cudaGraphAddEmptyNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies); 
#line 9723
extern cudaError_t __stdcall cudaGraphAddEventRecordNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaEvent_t event); 
#line 9750 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t * event_out); 
#line 9777 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event); 
#line 9824 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphAddEventWaitNode(cudaGraphNode_t * pGraphNode, cudaGraph_t graph, const cudaGraphNode_t * pDependencies, size_t numDependencies, cudaEvent_t event); 
#line 9851 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t * event_out); 
#line 9878 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event); 
#line 9906 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphClone(cudaGraph_t * pGraphClone, cudaGraph_t originalGraph); 
#line 9934
extern cudaError_t __stdcall cudaGraphNodeFindInClone(cudaGraphNode_t * pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph); 
#line 9965
extern cudaError_t __stdcall cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType * pType); 
#line 9996
extern cudaError_t __stdcall cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t * nodes, size_t * numNodes); 
#line 10027
extern cudaError_t __stdcall cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t * pRootNodes, size_t * pNumRootNodes); 
#line 10061
extern cudaError_t __stdcall cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t * from, cudaGraphNode_t * to, size_t * numEdges); 
#line 10092
extern cudaError_t __stdcall cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t * pDependencies, size_t * pNumDependencies); 
#line 10124
extern cudaError_t __stdcall cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t * pDependentNodes, size_t * pNumDependentNodes); 
#line 10155
extern cudaError_t __stdcall cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, size_t numDependencies); 
#line 10186
extern cudaError_t __stdcall cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t * from, const cudaGraphNode_t * to, size_t numDependencies); 
#line 10213
extern cudaError_t __stdcall cudaGraphDestroyNode(cudaGraphNode_t node); 
#line 10250
extern cudaError_t __stdcall cudaGraphInstantiate(cudaGraphExec_t * pGraphExec, cudaGraph_t graph, cudaGraphNode_t * pErrorNode, char * pLogBuffer, size_t bufferSize); 
#line 10284
extern cudaError_t __stdcall cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaKernelNodeParams * pNodeParams); 
#line 10328
extern cudaError_t __stdcall cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemcpy3DParms * pNodeParams); 
#line 10377
extern cudaError_t __stdcall cudaGraphExecMemcpyNodeSetParamsToSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind); 
#line 10434 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExecMemcpyNodeSetParamsFromSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind); 
#line 10489 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
#line 10537 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaMemsetParams * pNodeParams); 
#line 10570
extern cudaError_t __stdcall cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaHostNodeParams * pNodeParams); 
#line 10612
extern cudaError_t __stdcall cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph); 
#line 10647 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event); 
#line 10682 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event); 
#line 10758 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphNode_t * hErrorNode_out, cudaGraphExecUpdateResult * updateResult_out); 
#line 10782
extern cudaError_t __stdcall cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream); 
#line 10809 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_runtime_api.h"
extern cudaError_t __stdcall cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream); 
#line 10832
extern cudaError_t __stdcall cudaGraphExecDestroy(cudaGraphExec_t graphExec); 
#line 10853
extern cudaError_t __stdcall cudaGraphDestroy(cudaGraph_t graph); 
#line 10858
extern cudaError_t __stdcall cudaGetExportTable(const void ** ppExportTable, const cudaUUID_t * pExportTableId); 
#line 11034
extern cudaError_t cudaGetFuncBySymbol(cudaFunction_t * functionPtr, const void * symbolPtr); 
#line 11159 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_runtime_api.h"
}
#line 104 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\channel_descriptor.h"
template< class T> __inline ::cudaChannelFormatDesc cudaCreateChannelDesc() 
#line 105
{ 
#line 106
return cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone); 
#line 107
} 
#line 109
static __inline cudaChannelFormatDesc cudaCreateChannelDescHalf() 
#line 110
{ 
#line 111
int e = (((int)sizeof(unsigned short)) * 8); 
#line 113
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
#line 114
} 
#line 116
static __inline cudaChannelFormatDesc cudaCreateChannelDescHalf1() 
#line 117
{ 
#line 118
int e = (((int)sizeof(unsigned short)) * 8); 
#line 120
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
#line 121
} 
#line 123
static __inline cudaChannelFormatDesc cudaCreateChannelDescHalf2() 
#line 124
{ 
#line 125
int e = (((int)sizeof(unsigned short)) * 8); 
#line 127
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat); 
#line 128
} 
#line 130
static __inline cudaChannelFormatDesc cudaCreateChannelDescHalf4() 
#line 131
{ 
#line 132
int e = (((int)sizeof(unsigned short)) * 8); 
#line 134
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat); 
#line 135
} 
#line 137
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< char> () 
#line 138
{ 
#line 139
int e = (((int)sizeof(char)) * 8); 
#line 144 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\channel_descriptor.h"
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 146 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\channel_descriptor.h"
} 
#line 148
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< signed char> () 
#line 149
{ 
#line 150
int e = (((int)sizeof(signed char)) * 8); 
#line 152
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 153
} 
#line 155
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned char> () 
#line 156
{ 
#line 157
int e = (((int)sizeof(unsigned char)) * 8); 
#line 159
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 160
} 
#line 162
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< char1> () 
#line 163
{ 
#line 164
int e = (((int)sizeof(signed char)) * 8); 
#line 166
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 167
} 
#line 169
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar1> () 
#line 170
{ 
#line 171
int e = (((int)sizeof(unsigned char)) * 8); 
#line 173
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 174
} 
#line 176
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< char2> () 
#line 177
{ 
#line 178
int e = (((int)sizeof(signed char)) * 8); 
#line 180
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
#line 181
} 
#line 183
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar2> () 
#line 184
{ 
#line 185
int e = (((int)sizeof(unsigned char)) * 8); 
#line 187
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
#line 188
} 
#line 190
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< char4> () 
#line 191
{ 
#line 192
int e = (((int)sizeof(signed char)) * 8); 
#line 194
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
#line 195
} 
#line 197
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar4> () 
#line 198
{ 
#line 199
int e = (((int)sizeof(unsigned char)) * 8); 
#line 201
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
#line 202
} 
#line 204
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< short> () 
#line 205
{ 
#line 206
int e = (((int)sizeof(short)) * 8); 
#line 208
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 209
} 
#line 211
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned short> () 
#line 212
{ 
#line 213
int e = (((int)sizeof(unsigned short)) * 8); 
#line 215
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 216
} 
#line 218
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< short1> () 
#line 219
{ 
#line 220
int e = (((int)sizeof(short)) * 8); 
#line 222
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 223
} 
#line 225
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort1> () 
#line 226
{ 
#line 227
int e = (((int)sizeof(unsigned short)) * 8); 
#line 229
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 230
} 
#line 232
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< short2> () 
#line 233
{ 
#line 234
int e = (((int)sizeof(short)) * 8); 
#line 236
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
#line 237
} 
#line 239
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort2> () 
#line 240
{ 
#line 241
int e = (((int)sizeof(unsigned short)) * 8); 
#line 243
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
#line 244
} 
#line 246
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< short4> () 
#line 247
{ 
#line 248
int e = (((int)sizeof(short)) * 8); 
#line 250
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
#line 251
} 
#line 253
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort4> () 
#line 254
{ 
#line 255
int e = (((int)sizeof(unsigned short)) * 8); 
#line 257
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
#line 258
} 
#line 260
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< int> () 
#line 261
{ 
#line 262
int e = (((int)sizeof(int)) * 8); 
#line 264
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 265
} 
#line 267
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned> () 
#line 268
{ 
#line 269
int e = (((int)sizeof(unsigned)) * 8); 
#line 271
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 272
} 
#line 274
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< int1> () 
#line 275
{ 
#line 276
int e = (((int)sizeof(int)) * 8); 
#line 278
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 279
} 
#line 281
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uint1> () 
#line 282
{ 
#line 283
int e = (((int)sizeof(unsigned)) * 8); 
#line 285
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 286
} 
#line 288
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< int2> () 
#line 289
{ 
#line 290
int e = (((int)sizeof(int)) * 8); 
#line 292
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
#line 293
} 
#line 295
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uint2> () 
#line 296
{ 
#line 297
int e = (((int)sizeof(unsigned)) * 8); 
#line 299
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
#line 300
} 
#line 302
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< int4> () 
#line 303
{ 
#line 304
int e = (((int)sizeof(int)) * 8); 
#line 306
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
#line 307
} 
#line 309
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< uint4> () 
#line 310
{ 
#line 311
int e = (((int)sizeof(unsigned)) * 8); 
#line 313
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
#line 314
} 
#line 318
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< long> () 
#line 319
{ 
#line 320
int e = (((int)sizeof(long)) * 8); 
#line 322
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 323
} 
#line 325
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned long> () 
#line 326
{ 
#line 327
int e = (((int)sizeof(unsigned long)) * 8); 
#line 329
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 330
} 
#line 332
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< long1> () 
#line 333
{ 
#line 334
int e = (((int)sizeof(long)) * 8); 
#line 336
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
#line 337
} 
#line 339
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ulong1> () 
#line 340
{ 
#line 341
int e = (((int)sizeof(unsigned long)) * 8); 
#line 343
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
#line 344
} 
#line 346
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< long2> () 
#line 347
{ 
#line 348
int e = (((int)sizeof(long)) * 8); 
#line 350
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
#line 351
} 
#line 353
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ulong2> () 
#line 354
{ 
#line 355
int e = (((int)sizeof(unsigned long)) * 8); 
#line 357
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
#line 358
} 
#line 360
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< long4> () 
#line 361
{ 
#line 362
int e = (((int)sizeof(long)) * 8); 
#line 364
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
#line 365
} 
#line 367
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< ulong4> () 
#line 368
{ 
#line 369
int e = (((int)sizeof(unsigned long)) * 8); 
#line 371
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
#line 372
} 
#line 376 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\channel_descriptor.h"
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< float> () 
#line 377
{ 
#line 378
int e = (((int)sizeof(float)) * 8); 
#line 380
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
#line 381
} 
#line 383
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< float1> () 
#line 384
{ 
#line 385
int e = (((int)sizeof(float)) * 8); 
#line 387
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
#line 388
} 
#line 390
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< float2> () 
#line 391
{ 
#line 392
int e = (((int)sizeof(float)) * 8); 
#line 394
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat); 
#line 395
} 
#line 397
template<> __inline cudaChannelFormatDesc cudaCreateChannelDesc< float4> () 
#line 398
{ 
#line 399
int e = (((int)sizeof(float)) * 8); 
#line 401
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat); 
#line 402
} 
#line 79 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\driver_functions.h"
static __inline cudaPitchedPtr make_cudaPitchedPtr(void *d, size_t p, size_t xsz, size_t ysz) 
#line 80
{ 
#line 81
cudaPitchedPtr s; 
#line 83
(s.ptr) = d; 
#line 84
(s.pitch) = p; 
#line 85
(s.xsize) = xsz; 
#line 86
(s.ysize) = ysz; 
#line 88
return s; 
#line 89
} 
#line 106
static __inline cudaPos make_cudaPos(size_t x, size_t y, size_t z) 
#line 107
{ 
#line 108
cudaPos p; 
#line 110
(p.x) = x; 
#line 111
(p.y) = y; 
#line 112
(p.z) = z; 
#line 114
return p; 
#line 115
} 
#line 132
static __inline cudaExtent make_cudaExtent(size_t w, size_t h, size_t d) 
#line 133
{ 
#line 134
cudaExtent e; 
#line 136
(e.width) = w; 
#line 137
(e.height) = h; 
#line 138
(e.depth) = d; 
#line 140
return e; 
#line 141
} 
#line 73 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_functions.h"
static __inline char1 make_char1(signed char x); 
#line 75
static __inline uchar1 make_uchar1(unsigned char x); 
#line 77
static __inline char2 make_char2(signed char x, signed char y); 
#line 79
static __inline uchar2 make_uchar2(unsigned char x, unsigned char y); 
#line 81
static __inline char3 make_char3(signed char x, signed char y, signed char z); 
#line 83
static __inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z); 
#line 85
static __inline char4 make_char4(signed char x, signed char y, signed char z, signed char w); 
#line 87
static __inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w); 
#line 89
static __inline short1 make_short1(short x); 
#line 91
static __inline ushort1 make_ushort1(unsigned short x); 
#line 93
static __inline short2 make_short2(short x, short y); 
#line 95
static __inline ushort2 make_ushort2(unsigned short x, unsigned short y); 
#line 97
static __inline short3 make_short3(short x, short y, short z); 
#line 99
static __inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z); 
#line 101
static __inline short4 make_short4(short x, short y, short z, short w); 
#line 103
static __inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w); 
#line 105
static __inline int1 make_int1(int x); 
#line 107
static __inline uint1 make_uint1(unsigned x); 
#line 109
static __inline int2 make_int2(int x, int y); 
#line 111
static __inline uint2 make_uint2(unsigned x, unsigned y); 
#line 113
static __inline int3 make_int3(int x, int y, int z); 
#line 115
static __inline uint3 make_uint3(unsigned x, unsigned y, unsigned z); 
#line 117
static __inline int4 make_int4(int x, int y, int z, int w); 
#line 119
static __inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w); 
#line 121
static __inline long1 make_long1(long x); 
#line 123
static __inline ulong1 make_ulong1(unsigned long x); 
#line 125
static __inline long2 make_long2(long x, long y); 
#line 127
static __inline ulong2 make_ulong2(unsigned long x, unsigned long y); 
#line 129
static __inline long3 make_long3(long x, long y, long z); 
#line 131
static __inline ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z); 
#line 133
static __inline long4 make_long4(long x, long y, long z, long w); 
#line 135
static __inline ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w); 
#line 137
static __inline float1 make_float1(float x); 
#line 139
static __inline float2 make_float2(float x, float y); 
#line 141
static __inline float3 make_float3(float x, float y, float z); 
#line 143
static __inline float4 make_float4(float x, float y, float z, float w); 
#line 145
static __inline longlong1 make_longlong1(__int64 x); 
#line 147
static __inline ulonglong1 make_ulonglong1(unsigned __int64 x); 
#line 149
static __inline longlong2 make_longlong2(__int64 x, __int64 y); 
#line 151
static __inline ulonglong2 make_ulonglong2(unsigned __int64 x, unsigned __int64 y); 
#line 153
static __inline longlong3 make_longlong3(__int64 x, __int64 y, __int64 z); 
#line 155
static __inline ulonglong3 make_ulonglong3(unsigned __int64 x, unsigned __int64 y, unsigned __int64 z); 
#line 157
static __inline longlong4 make_longlong4(__int64 x, __int64 y, __int64 z, __int64 w); 
#line 159
static __inline ulonglong4 make_ulonglong4(unsigned __int64 x, unsigned __int64 y, unsigned __int64 z, unsigned __int64 w); 
#line 161
static __inline double1 make_double1(double x); 
#line 163
static __inline double2 make_double2(double x, double y); 
#line 165
static __inline double3 make_double3(double x, double y, double z); 
#line 167
static __inline double4 make_double4(double x, double y, double z, double w); 
#line 73 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\vector_functions.hpp"
static __inline char1 make_char1(signed char x) 
#line 74
{ 
#line 75
char1 t; (t.x) = x; return t; 
#line 76
} 
#line 78
static __inline uchar1 make_uchar1(unsigned char x) 
#line 79
{ 
#line 80
uchar1 t; (t.x) = x; return t; 
#line 81
} 
#line 83
static __inline char2 make_char2(signed char x, signed char y) 
#line 84
{ 
#line 85
char2 t; (t.x) = x; (t.y) = y; return t; 
#line 86
} 
#line 88
static __inline uchar2 make_uchar2(unsigned char x, unsigned char y) 
#line 89
{ 
#line 90
uchar2 t; (t.x) = x; (t.y) = y; return t; 
#line 91
} 
#line 93
static __inline char3 make_char3(signed char x, signed char y, signed char z) 
#line 94
{ 
#line 95
char3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 96
} 
#line 98
static __inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z) 
#line 99
{ 
#line 100
uchar3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 101
} 
#line 103
static __inline char4 make_char4(signed char x, signed char y, signed char z, signed char w) 
#line 104
{ 
#line 105
char4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 106
} 
#line 108
static __inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w) 
#line 109
{ 
#line 110
uchar4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 111
} 
#line 113
static __inline short1 make_short1(short x) 
#line 114
{ 
#line 115
short1 t; (t.x) = x; return t; 
#line 116
} 
#line 118
static __inline ushort1 make_ushort1(unsigned short x) 
#line 119
{ 
#line 120
ushort1 t; (t.x) = x; return t; 
#line 121
} 
#line 123
static __inline short2 make_short2(short x, short y) 
#line 124
{ 
#line 125
short2 t; (t.x) = x; (t.y) = y; return t; 
#line 126
} 
#line 128
static __inline ushort2 make_ushort2(unsigned short x, unsigned short y) 
#line 129
{ 
#line 130
ushort2 t; (t.x) = x; (t.y) = y; return t; 
#line 131
} 
#line 133
static __inline short3 make_short3(short x, short y, short z) 
#line 134
{ 
#line 135
short3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 136
} 
#line 138
static __inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z) 
#line 139
{ 
#line 140
ushort3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 141
} 
#line 143
static __inline short4 make_short4(short x, short y, short z, short w) 
#line 144
{ 
#line 145
short4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 146
} 
#line 148
static __inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w) 
#line 149
{ 
#line 150
ushort4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 151
} 
#line 153
static __inline int1 make_int1(int x) 
#line 154
{ 
#line 155
int1 t; (t.x) = x; return t; 
#line 156
} 
#line 158
static __inline uint1 make_uint1(unsigned x) 
#line 159
{ 
#line 160
uint1 t; (t.x) = x; return t; 
#line 161
} 
#line 163
static __inline int2 make_int2(int x, int y) 
#line 164
{ 
#line 165
int2 t; (t.x) = x; (t.y) = y; return t; 
#line 166
} 
#line 168
static __inline uint2 make_uint2(unsigned x, unsigned y) 
#line 169
{ 
#line 170
uint2 t; (t.x) = x; (t.y) = y; return t; 
#line 171
} 
#line 173
static __inline int3 make_int3(int x, int y, int z) 
#line 174
{ 
#line 175
int3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 176
} 
#line 178
static __inline uint3 make_uint3(unsigned x, unsigned y, unsigned z) 
#line 179
{ 
#line 180
uint3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 181
} 
#line 183
static __inline int4 make_int4(int x, int y, int z, int w) 
#line 184
{ 
#line 185
int4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 186
} 
#line 188
static __inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w) 
#line 189
{ 
#line 190
uint4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 191
} 
#line 193
static __inline long1 make_long1(long x) 
#line 194
{ 
#line 195
long1 t; (t.x) = x; return t; 
#line 196
} 
#line 198
static __inline ulong1 make_ulong1(unsigned long x) 
#line 199
{ 
#line 200
ulong1 t; (t.x) = x; return t; 
#line 201
} 
#line 203
static __inline long2 make_long2(long x, long y) 
#line 204
{ 
#line 205
long2 t; (t.x) = x; (t.y) = y; return t; 
#line 206
} 
#line 208
static __inline ulong2 make_ulong2(unsigned long x, unsigned long y) 
#line 209
{ 
#line 210
ulong2 t; (t.x) = x; (t.y) = y; return t; 
#line 211
} 
#line 213
static __inline long3 make_long3(long x, long y, long z) 
#line 214
{ 
#line 215
long3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 216
} 
#line 218
static __inline ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z) 
#line 219
{ 
#line 220
ulong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 221
} 
#line 223
static __inline long4 make_long4(long x, long y, long z, long w) 
#line 224
{ 
#line 225
long4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 226
} 
#line 228
static __inline ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w) 
#line 229
{ 
#line 230
ulong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 231
} 
#line 233
static __inline float1 make_float1(float x) 
#line 234
{ 
#line 235
float1 t; (t.x) = x; return t; 
#line 236
} 
#line 238
static __inline float2 make_float2(float x, float y) 
#line 239
{ 
#line 240
float2 t; (t.x) = x; (t.y) = y; return t; 
#line 241
} 
#line 243
static __inline float3 make_float3(float x, float y, float z) 
#line 244
{ 
#line 245
float3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 246
} 
#line 248
static __inline float4 make_float4(float x, float y, float z, float w) 
#line 249
{ 
#line 250
float4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 251
} 
#line 253
static __inline longlong1 make_longlong1(__int64 x) 
#line 254
{ 
#line 255
longlong1 t; (t.x) = x; return t; 
#line 256
} 
#line 258
static __inline ulonglong1 make_ulonglong1(unsigned __int64 x) 
#line 259
{ 
#line 260
ulonglong1 t; (t.x) = x; return t; 
#line 261
} 
#line 263
static __inline longlong2 make_longlong2(__int64 x, __int64 y) 
#line 264
{ 
#line 265
longlong2 t; (t.x) = x; (t.y) = y; return t; 
#line 266
} 
#line 268
static __inline ulonglong2 make_ulonglong2(unsigned __int64 x, unsigned __int64 y) 
#line 269
{ 
#line 270
ulonglong2 t; (t.x) = x; (t.y) = y; return t; 
#line 271
} 
#line 273
static __inline longlong3 make_longlong3(__int64 x, __int64 y, __int64 z) 
#line 274
{ 
#line 275
longlong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 276
} 
#line 278
static __inline ulonglong3 make_ulonglong3(unsigned __int64 x, unsigned __int64 y, unsigned __int64 z) 
#line 279
{ 
#line 280
ulonglong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 281
} 
#line 283
static __inline longlong4 make_longlong4(__int64 x, __int64 y, __int64 z, __int64 w) 
#line 284
{ 
#line 285
longlong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 286
} 
#line 288
static __inline ulonglong4 make_ulonglong4(unsigned __int64 x, unsigned __int64 y, unsigned __int64 z, unsigned __int64 w) 
#line 289
{ 
#line 290
ulonglong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 291
} 
#line 293
static __inline double1 make_double1(double x) 
#line 294
{ 
#line 295
double1 t; (t.x) = x; return t; 
#line 296
} 
#line 298
static __inline double2 make_double2(double x, double y) 
#line 299
{ 
#line 300
double2 t; (t.x) = x; (t.y) = y; return t; 
#line 301
} 
#line 303
static __inline double3 make_double3(double x, double y, double z) 
#line 304
{ 
#line 305
double3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
#line 306
} 
#line 308
static __inline double4 make_double4(double x, double y, double z, double w) 
#line 309
{ 
#line 310
double4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
#line 311
} 
#line 14 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\errno.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 19
int *__cdecl _errno(); 
#line 22
errno_t __cdecl _set_errno(int _Value); 
#line 23
errno_t __cdecl _get_errno(int * _Value); 
#line 25
unsigned long *__cdecl __doserrno(); 
#line 28
errno_t __cdecl _set_doserrno(unsigned long _Value); 
#line 29
errno_t __cdecl _get_doserrno(unsigned long * _Value); 
#line 130 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\errno.h"
}__pragma( pack ( pop )) 
#line 14 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\vcruntime_string.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 19
const void *__cdecl memchr(const void * _Buf, int _Val, size_t _MaxCount); 
#line 26
int __cdecl memcmp(const void * _Buf1, const void * _Buf2, size_t _Size); 
#line 41 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\vcruntime_string.h"
void *__cdecl memcpy(void * _Dst, const void * _Src, size_t _Size); 
#line 48
void *__cdecl memmove(void * _Dst, const void * _Src, size_t _Size); 
#line 61 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\vcruntime_string.h"
void *__cdecl memset(void * _Dst, int _Val, size_t _Size); 
#line 68
const char *__cdecl strchr(const char * _Str, int _Val); 
#line 74
const char *__cdecl strrchr(const char * _Str, int _Ch); 
#line 80
const char *__cdecl strstr(const char * _Str, const char * _SubStr); 
#line 87
const __wchar_t *__cdecl wcschr(const __wchar_t * _Str, __wchar_t _Ch); 
#line 93
const __wchar_t *__cdecl wcsrchr(const __wchar_t * _Str, __wchar_t _Ch); 
#line 100
const __wchar_t *__cdecl wcsstr(const __wchar_t * _Str, const __wchar_t * _SubStr); 
#line 107
}__pragma( pack ( pop )) 
#line 14 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_memcpy_s.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 35 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_memcpy_s.h"
static __inline errno_t __cdecl memcpy_s(void *const 
#line 36
_Destination, const rsize_t 
#line 37
_DestinationSize, const void *const 
#line 38
_Source, const rsize_t 
#line 39
_SourceSize) 
#line 41
{ 
#line 42
if (_SourceSize == (0)) 
#line 43
{ 
#line 44
return 0; 
#line 45
}  
#line 47
{ int _Expr_val = !(!(_Destination != (0))); if (!_Expr_val) { (*_errno()) = 22; _invalid_parameter_noinfo(); return 22; }  } ; 
#line 48
if ((_Source == (0)) || (_DestinationSize < _SourceSize)) 
#line 49
{ 
#line 50
memset(_Destination, 0, _DestinationSize); 
#line 52
{ int _Expr_val = !(!(_Source != (0))); if (!_Expr_val) { (*_errno()) = 22; _invalid_parameter_noinfo(); return 22; }  } ; 
#line 53
{ int _Expr_val = !(!(_DestinationSize >= _SourceSize)); if (!_Expr_val) { (*_errno()) = 34; _invalid_parameter_noinfo(); return 34; }  } ; 
#line 56
return 22; 
#line 57
}  
#pragma warning(suppress:4996)
memcpy(_Destination, _Source, _SourceSize); 
#line 60
return 0; 
#line 61
} 
#line 64
static __inline errno_t __cdecl memmove_s(void *const 
#line 65
_Destination, const rsize_t 
#line 66
_DestinationSize, const void *const 
#line 67
_Source, const rsize_t 
#line 68
_SourceSize) 
#line 70
{ 
#line 71
if (_SourceSize == (0)) 
#line 72
{ 
#line 73
return 0; 
#line 74
}  
#line 76
{ int _Expr_val = !(!(_Destination != (0))); if (!_Expr_val) { (*_errno()) = 22; _invalid_parameter_noinfo(); return 22; }  } ; 
#line 77
{ int _Expr_val = !(!(_Source != (0))); if (!_Expr_val) { (*_errno()) = 22; _invalid_parameter_noinfo(); return 22; }  } ; 
#line 78
{ int _Expr_val = !(!(_DestinationSize >= _SourceSize)); if (!_Expr_val) { (*_errno()) = 34; _invalid_parameter_noinfo(); return 34; }  } ; 
#line 80
#pragma warning(suppress:4996)
memmove(_Destination, _Source, _SourceSize); 
#line 82
return 0; 
#line 83
} 
#line 89 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_memcpy_s.h"
}__pragma( pack ( pop )) 
#line 19 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_memory.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 24
int __cdecl _memicmp(const void * _Buf1, const void * _Buf2, size_t _Size); 
#line 31
int __cdecl _memicmp_l(const void * _Buf1, const void * _Buf2, size_t _Size, _locale_t _Locale); 
#line 79 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_memory.h"
void *__cdecl memccpy(void * _Dst, const void * _Src, int _Val, size_t _Size); 
#line 87
int __cdecl memicmp(const void * _Buf1, const void * _Buf2, size_t _Size); 
#line 100 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_memory.h"
extern "C++" inline void *__cdecl memchr(void *
#line 101
_Pv, int 
#line 102
_C, size_t 
#line 103
_N) 
#line 105
{ 
#line 106
const void *const _Pvc = _Pv; 
#line 107
return const_cast< void *>(memchr(_Pvc, _C, _N)); 
#line 108
} 
#line 114 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_memory.h"
}__pragma( pack ( pop )) 
#line 16 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 28
errno_t __cdecl wcscat_s(__wchar_t * _Destination, rsize_t _SizeInWords, const __wchar_t * _Source); 
#line 35
errno_t __cdecl wcscpy_s(__wchar_t * _Destination, rsize_t _SizeInWords, const __wchar_t * _Source); 
#line 42
errno_t __cdecl wcsncat_s(__wchar_t * _Destination, rsize_t _SizeInWords, const __wchar_t * _Source, rsize_t _MaxCount); 
#line 50
errno_t __cdecl wcsncpy_s(__wchar_t * _Destination, rsize_t _SizeInWords, const __wchar_t * _Source, rsize_t _MaxCount); 
#line 58
__wchar_t *__cdecl wcstok_s(__wchar_t * _String, const __wchar_t * _Delimiter, __wchar_t ** _Context); 
#line 79 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
__declspec(allocator) __wchar_t *__cdecl _wcsdup(const __wchar_t * _String); 
#line 89 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
extern "C++" {template < size_t _Size > inline errno_t __cdecl wcscat_s ( wchar_t ( & _Destination ) [ _Size ], wchar_t const * _Source ) throw ( ) { return wcscat_s ( _Destination, _Size, _Source ); }}
#line 98 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
#pragma warning(push)
#pragma warning(disable: 28719)
#pragma warning(disable: 28726)
__wchar_t *__cdecl wcscat(__wchar_t * _Destination, const __wchar_t * _Source); 
#line 106 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
#pragma warning(pop)
#line 109 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
int __cdecl wcscmp(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 114
extern "C++" {template < size_t _Size > inline errno_t __cdecl wcscpy_s ( wchar_t ( & _Destination ) [ _Size ], wchar_t const * _Source ) throw ( ) { return wcscpy_s ( _Destination, _Size, _Source ); }}
#line 121 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
#pragma warning(push)
#pragma warning(disable: 28719)
#pragma warning(disable: 28726)
__wchar_t *__cdecl wcscpy(__wchar_t * _Destination, const __wchar_t * _Source); 
#line 129 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
#pragma warning(pop)
#line 132
size_t __cdecl wcscspn(const __wchar_t * _String, const __wchar_t * _Control); 
#line 138
size_t __cdecl wcslen(const __wchar_t * _String); 
#line 151 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
size_t __cdecl wcsnlen(const __wchar_t * _Source, size_t _MaxCount); 
#line 167 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
static __inline size_t __cdecl wcsnlen_s(const __wchar_t *
#line 168
_Source, size_t 
#line 169
_MaxCount) 
#line 171
{ 
#line 172
return (_Source == (0)) ? 0 : wcsnlen(_Source, _MaxCount); 
#line 173
} 
#line 176 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
extern "C++" {template < size_t _Size > inline errno_t __cdecl wcsncat_s ( wchar_t ( & _Destination ) [ _Size ], wchar_t const * _Source, size_t _Count ) throw ( ) { return wcsncat_s ( _Destination, _Size, _Source, _Count ); }}
#line 184 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl wcsncat(__wchar_t * _Destination, const __wchar_t * _Source, size_t _Count); 
#line 193 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
int __cdecl wcsncmp(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount); 
#line 199
extern "C++" {template < size_t _Size > inline errno_t __cdecl wcsncpy_s ( wchar_t ( & _Destination ) [ _Size ], wchar_t const * _Source, size_t _Count ) throw ( ) { return wcsncpy_s ( _Destination, _Size, _Source, _Count ); }}
#line 206 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl wcsncpy(__wchar_t * _Destination, const __wchar_t * _Source, size_t _Count); 
#line 215 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
const __wchar_t *__cdecl wcspbrk(const __wchar_t * _String, const __wchar_t * _Control); 
#line 221
size_t __cdecl wcsspn(const __wchar_t * _String, const __wchar_t * _Control); 
#line 227
__wchar_t *__cdecl wcstok(__wchar_t * _String, const __wchar_t * _Delimiter, __wchar_t ** _Context); 
#line 243 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
#pragma warning(push)
#pragma warning(disable: 4141 4996)
#pragma warning(disable: 28719 28726 28727)
#line 247
static __inline __wchar_t *__cdecl _wcstok(__wchar_t *const 
#line 248
_String, const __wchar_t *const 
#line 249
_Delimiter) 
#line 251
{ 
#line 252
return wcstok(_String, _Delimiter, 0); 
#line 253
} 
#line 261 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
extern "C++" 
#line 260
__declspec(deprecated("wcstok has been changed to conform with the ISO C standard, adding an extra context parameter. To use the legacy Microsoft wcsto" "k, define _CRT_NON_CONFORMING_WCSTOK.")) inline __wchar_t *__cdecl 
#line 261
wcstok(__wchar_t *
#line 262
_String, const __wchar_t *
#line 263
_Delimiter) throw() 
#line 265
{ 
#line 266
return wcstok(_String, _Delimiter, 0); 
#line 267
} 
#line 270 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
#pragma warning(pop)
#line 278 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcserror(int _ErrorNumber); 
#line 283
errno_t __cdecl _wcserror_s(__wchar_t * _Buffer, size_t _SizeInWords, int _ErrorNumber); 
#line 289
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcserror_s ( wchar_t ( & _Buffer ) [ _Size ], int _Error ) throw ( ) { return _wcserror_s ( _Buffer, _Size, _Error ); }}
#line 299 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl __wcserror(const __wchar_t * _String); 
#line 303
errno_t __cdecl __wcserror_s(__wchar_t * _Buffer, size_t _SizeInWords, const __wchar_t * _ErrorMessage); 
#line 309
extern "C++" {template < size_t _Size > inline errno_t __cdecl __wcserror_s ( wchar_t ( & _Buffer ) [ _Size ], wchar_t const * _ErrorMessage ) throw ( ) { return __wcserror_s ( _Buffer, _Size, _ErrorMessage ); }}
#line 315 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
int __cdecl _wcsicmp(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 320
int __cdecl _wcsicmp_l(const __wchar_t * _String1, const __wchar_t * _String2, _locale_t _Locale); 
#line 326
int __cdecl _wcsnicmp(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount); 
#line 332
int __cdecl _wcsnicmp_l(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 339
errno_t __cdecl _wcsnset_s(__wchar_t * _Destination, size_t _SizeInWords, __wchar_t _Value, size_t _MaxCount); 
#line 346
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcsnset_s ( wchar_t ( & _Destination ) [ _Size ], wchar_t _Value, size_t _MaxCount ) throw ( ) { return _wcsnset_s ( _Destination, _Size, _Value, _MaxCount ); }}
#line 353 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcsnset(__wchar_t * _String, __wchar_t _Value, size_t _MaxCount); 
#line 361 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcsrev(__wchar_t * _String); 
#line 365
errno_t __cdecl _wcsset_s(__wchar_t * _Destination, size_t _SizeInWords, __wchar_t _Value); 
#line 371
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcsset_s ( wchar_t ( & _String ) [ _Size ], wchar_t _Value ) throw ( ) { return _wcsset_s ( _String, _Size, _Value ); }}
#line 377 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcsset(__wchar_t * _String, __wchar_t _Value); 
#line 384 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
errno_t __cdecl _wcslwr_s(__wchar_t * _String, size_t _SizeInWords); 
#line 389
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcslwr_s ( wchar_t ( & _String ) [ _Size ] ) throw ( ) { return _wcslwr_s ( _String, _Size ); }}
#line 394 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcslwr(__wchar_t * _String); 
#line 400 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
errno_t __cdecl _wcslwr_s_l(__wchar_t * _String, size_t _SizeInWords, _locale_t _Locale); 
#line 406
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcslwr_s_l ( wchar_t ( & _String ) [ _Size ], _locale_t _Locale ) throw ( ) { return _wcslwr_s_l ( _String, _Size, _Locale ); }}
#line 412 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcslwr_l(__wchar_t * _String, _locale_t _Locale); 
#line 420 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
errno_t __cdecl _wcsupr_s(__wchar_t * _String, size_t _Size); 
#line 425
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcsupr_s ( wchar_t ( & _String ) [ _Size ] ) throw ( ) { return _wcsupr_s ( _String, _Size ); }}
#line 430 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcsupr(__wchar_t * _String); 
#line 436 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
errno_t __cdecl _wcsupr_s_l(__wchar_t * _String, size_t _Size, _locale_t _Locale); 
#line 442
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcsupr_s_l ( wchar_t ( & _String ) [ _Size ], _locale_t _Locale ) throw ( ) { return _wcsupr_s_l ( _String, _Size, _Locale ); }}
#line 448 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl _wcsupr_l(__wchar_t * _String, _locale_t _Locale); 
#line 457 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
size_t __cdecl wcsxfrm(__wchar_t * _Destination, const __wchar_t * _Source, size_t _MaxCount); 
#line 465
size_t __cdecl _wcsxfrm_l(__wchar_t * _Destination, const __wchar_t * _Source, size_t _MaxCount, _locale_t _Locale); 
#line 473
int __cdecl wcscoll(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 479
int __cdecl _wcscoll_l(const __wchar_t * _String1, const __wchar_t * _String2, _locale_t _Locale); 
#line 486
int __cdecl _wcsicoll(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 492
int __cdecl _wcsicoll_l(const __wchar_t * _String1, const __wchar_t * _String2, _locale_t _Locale); 
#line 499
int __cdecl _wcsncoll(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount); 
#line 506
int __cdecl _wcsncoll_l(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 514
int __cdecl _wcsnicoll(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount); 
#line 521
int __cdecl _wcsnicoll_l(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 536
extern "C++" {
#line 540
inline __wchar_t *__cdecl wcschr(__wchar_t *_String, __wchar_t _C) 
#line 541
{ 
#line 542
return const_cast< __wchar_t *>(wcschr(static_cast< const __wchar_t *>(_String), _C)); 
#line 543
} 
#line 546
inline __wchar_t *__cdecl wcspbrk(__wchar_t *_String, const __wchar_t *_Control) 
#line 547
{ 
#line 548
return const_cast< __wchar_t *>(wcspbrk(static_cast< const __wchar_t *>(_String), _Control)); 
#line 549
} 
#line 552
inline __wchar_t *__cdecl wcsrchr(__wchar_t *_String, __wchar_t _C) 
#line 553
{ 
#line 554
return const_cast< __wchar_t *>(wcsrchr(static_cast< const __wchar_t *>(_String), _C)); 
#line 555
} 
#line 559
inline __wchar_t *__cdecl wcsstr(__wchar_t *_String, const __wchar_t *_SubStr) 
#line 560
{ 
#line 561
return const_cast< __wchar_t *>(wcsstr(static_cast< const __wchar_t *>(_String), _SubStr)); 
#line 562
} 
#line 564
}
#line 580 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
__wchar_t *__cdecl wcsdup(const __wchar_t * _String); 
#line 592 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
int __cdecl wcsicmp(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 598
int __cdecl wcsnicmp(const __wchar_t * _String1, const __wchar_t * _String2, size_t _MaxCount); 
#line 606
__wchar_t *__cdecl wcsnset(__wchar_t * _String, __wchar_t _Value, size_t _MaxCount); 
#line 614
__wchar_t *__cdecl wcsrev(__wchar_t * _String); 
#line 620
__wchar_t *__cdecl wcsset(__wchar_t * _String, __wchar_t _Value); 
#line 627
__wchar_t *__cdecl wcslwr(__wchar_t * _String); 
#line 633
__wchar_t *__cdecl wcsupr(__wchar_t * _String); 
#line 638
int __cdecl wcsicoll(const __wchar_t * _String1, const __wchar_t * _String2); 
#line 647 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstring.h"
}__pragma( pack ( pop )) 
#line 19 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\string.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 28
errno_t __cdecl strcpy_s(char * _Destination, rsize_t _SizeInBytes, const char * _Source); 
#line 35
errno_t __cdecl strcat_s(char * _Destination, rsize_t _SizeInBytes, const char * _Source); 
#line 42
errno_t __cdecl strerror_s(char * _Buffer, size_t _SizeInBytes, int _ErrorNumber); 
#line 48
errno_t __cdecl strncat_s(char * _Destination, rsize_t _SizeInBytes, const char * _Source, rsize_t _MaxCount); 
#line 56
errno_t __cdecl strncpy_s(char * _Destination, rsize_t _SizeInBytes, const char * _Source, rsize_t _MaxCount); 
#line 64
char *__cdecl strtok_s(char * _String, const char * _Delimiter, char ** _Context); 
#line 72 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\string.h"
void *__cdecl _memccpy(void * _Dst, const void * _Src, int _Val, size_t _MaxCount); 
#line 79
extern "C++" {template < size_t _Size > inline errno_t __cdecl strcat_s ( char ( & _Destination ) [ _Size ], char const * _Source ) throw ( ) { return strcat_s ( _Destination, _Size, _Source ); }}
#line 88 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\string.h"
#pragma warning(push)
#pragma warning(disable: 28719)
#pragma warning(disable: 28726)
char *__cdecl strcat(char * _Destination, const char * _Source); 
#line 96 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\string.h"
#pragma warning(pop)
#line 100 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\string.h"
int __cdecl strcmp(const char * _Str1, const char * _Str2); 
#line 106
int __cdecl _strcmpi(const char * _String1, const char * _String2); 
#line 112
int __cdecl strcoll(const char * _String1, const char * _String2); 
#line 118
int __cdecl _strcoll_l(const char * _String1, const char * _String2, _locale_t _Locale); 
#line 124
extern "C++" {template < size_t _Size > inline errno_t __cdecl strcpy_s ( char ( & _Destination ) [ _Size ], char const * _Source ) throw ( ) { return strcpy_s ( _Destination, _Size, _Source ); }}
#line 131 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\string.h"
#pragma warning(push)
#pragma warning(disable: 28719)
#pragma warning(disable: 28726)
char *__cdecl strcpy(char * _Destination, const char * _Source); 
#line 139 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\string.h"
#pragma warning(pop)
#line 142
size_t __cdecl strcspn(const char * _Str, const char * _Control); 
#line 152 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\string.h"
__declspec(allocator) char *__cdecl _strdup(const char * _Source); 
#line 163 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\string.h"
char *__cdecl _strerror(const char * _ErrorMessage); 
#line 168
errno_t __cdecl _strerror_s(char * _Buffer, size_t _SizeInBytes, const char * _ErrorMessage); 
#line 174
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strerror_s ( char ( & _Buffer ) [ _Size ], char const * _ErrorMessage ) throw ( ) { return _strerror_s ( _Buffer, _Size, _ErrorMessage ); }}
#line 183 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\string.h"
char *__cdecl strerror(int _ErrorMessage); 
#line 187
extern "C++" {template < size_t _Size > inline errno_t __cdecl strerror_s ( char ( & _Buffer ) [ _Size ], int _ErrorMessage ) throw ( ) { return strerror_s ( _Buffer, _Size, _ErrorMessage ); }}
#line 194 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\string.h"
int __cdecl _stricmp(const char * _String1, const char * _String2); 
#line 200
int __cdecl _stricoll(const char * _String1, const char * _String2); 
#line 206
int __cdecl _stricoll_l(const char * _String1, const char * _String2, _locale_t _Locale); 
#line 213
int __cdecl _stricmp_l(const char * _String1, const char * _String2, _locale_t _Locale); 
#line 220
size_t __cdecl strlen(const char * _Str); 
#line 225
errno_t __cdecl _strlwr_s(char * _String, size_t _Size); 
#line 230
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strlwr_s ( char ( & _String ) [ _Size ] ) throw ( ) { return _strlwr_s ( _String, _Size ); }}
#line 235 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\string.h"
char *__cdecl _strlwr(char * _String); 
#line 241 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\string.h"
errno_t __cdecl _strlwr_s_l(char * _String, size_t _Size, _locale_t _Locale); 
#line 247
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strlwr_s_l ( char ( & _String ) [ _Size ], _locale_t _Locale ) throw ( ) { return _strlwr_s_l ( _String, _Size, _Locale ); }}
#line 253 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\string.h"
char *__cdecl _strlwr_l(char * _String, _locale_t _Locale); 
#line 260 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\string.h"
extern "C++" {template < size_t _Size > inline errno_t __cdecl strncat_s ( char ( & _Destination ) [ _Size ], char const * _Source, size_t _Count ) throw ( ) { return strncat_s ( _Destination, _Size, _Source, _Count ); }}
#line 267 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\string.h"
char *__cdecl strncat(char * _Destination, const char * _Source, size_t _Count); 
#line 276 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\string.h"
int __cdecl strncmp(const char * _Str1, const char * _Str2, size_t _MaxCount); 
#line 283
int __cdecl _strnicmp(const char * _String1, const char * _String2, size_t _MaxCount); 
#line 290
int __cdecl _strnicmp_l(const char * _String1, const char * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 298
int __cdecl _strnicoll(const char * _String1, const char * _String2, size_t _MaxCount); 
#line 305
int __cdecl _strnicoll_l(const char * _String1, const char * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 313
int __cdecl _strncoll(const char * _String1, const char * _String2, size_t _MaxCount); 
#line 320
int __cdecl _strncoll_l(const char * _String1, const char * _String2, size_t _MaxCount, _locale_t _Locale); 
#line 327
size_t __cdecl __strncnt(const char * _String, size_t _Count); 
#line 332
extern "C++" {template < size_t _Size > inline errno_t __cdecl strncpy_s ( char ( & _Destination ) [ _Size ], char const * _Source, size_t _Count ) throw ( ) { return strncpy_s ( _Destination, _Size, _Source, _Count ); }}
#line 339 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\string.h"
char *__cdecl strncpy(char * _Destination, const char * _Source, size_t _Count); 
#line 356 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\string.h"
size_t __cdecl strnlen(const char * _String, size_t _MaxCount); 
#line 372 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\string.h"
static __inline size_t __cdecl strnlen_s(const char *
#line 373
_String, size_t 
#line 374
_MaxCount) 
#line 376
{ 
#line 377
return (_String == (0)) ? 0 : strnlen(_String, _MaxCount); 
#line 378
} 
#line 382 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\string.h"
errno_t __cdecl _strnset_s(char * _String, size_t _SizeInBytes, int _Value, size_t _MaxCount); 
#line 389
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strnset_s ( char ( & _Destination ) [ _Size ], int _Value, size_t _Count ) throw ( ) { return _strnset_s ( _Destination, _Size, _Value, _Count ); }}
#line 397 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\string.h"
char *__cdecl _strnset(char * _Destination, int _Value, size_t _Count); 
#line 406 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\string.h"
const char *__cdecl strpbrk(const char * _Str, const char * _Control); 
#line 411
char *__cdecl _strrev(char * _Str); 
#line 416
errno_t __cdecl _strset_s(char * _Destination, size_t _DestinationSize, int _Value); 
#line 422
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strset_s ( char ( & _Destination ) [ _Size ], int _Value ) throw ( ) { return _strset_s ( _Destination, _Size, _Value ); }}
#line 428 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\string.h"
char *__cdecl _strset(char * _Destination, int _Value); 
#line 435 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\string.h"
size_t __cdecl strspn(const char * _Str, const char * _Control); 
#line 441
char *__cdecl strtok(char * _String, const char * _Delimiter); 
#line 447
errno_t __cdecl _strupr_s(char * _String, size_t _Size); 
#line 452
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strupr_s ( char ( & _String ) [ _Size ] ) throw ( ) { return _strupr_s ( _String, _Size ); }}
#line 457 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\string.h"
char *__cdecl _strupr(char * _String); 
#line 463 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\string.h"
errno_t __cdecl _strupr_s_l(char * _String, size_t _Size, _locale_t _Locale); 
#line 469
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strupr_s_l ( char ( & _String ) [ _Size ], _locale_t _Locale ) throw ( ) { return _strupr_s_l ( _String, _Size, _Locale ); }}
#line 475 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\string.h"
char *__cdecl _strupr_l(char * _String, _locale_t _Locale); 
#line 484 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\string.h"
size_t __cdecl strxfrm(char * _Destination, const char * _Source, size_t _MaxCount); 
#line 492
size_t __cdecl _strxfrm_l(char * _Destination, const char * _Source, size_t _MaxCount, _locale_t _Locale); 
#line 502
extern "C++" {
#line 505
inline char *__cdecl strchr(char *const _String, const int _Ch) 
#line 506
{ 
#line 507
return const_cast< char *>(strchr(static_cast< const char *>(_String), _Ch)); 
#line 508
} 
#line 511
inline char *__cdecl strpbrk(char *const _String, const char *const _Control) 
#line 512
{ 
#line 513
return const_cast< char *>(strpbrk(static_cast< const char *>(_String), _Control)); 
#line 514
} 
#line 517
inline char *__cdecl strrchr(char *const _String, const int _Ch) 
#line 518
{ 
#line 519
return const_cast< char *>(strrchr(static_cast< const char *>(_String), _Ch)); 
#line 520
} 
#line 523
inline char *__cdecl strstr(char *const _String, const char *const _SubString) 
#line 524
{ 
#line 525
return const_cast< char *>(strstr(static_cast< const char *>(_String), _SubString)); 
#line 526
} 
#line 527
}
#line 536 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\string.h"
char *__cdecl strdup(const char * _String); 
#line 543
int __cdecl strcmpi(const char * _String1, const char * _String2); 
#line 549
int __cdecl stricmp(const char * _String1, const char * _String2); 
#line 555
char *__cdecl strlwr(char * _String); 
#line 560
int __cdecl strnicmp(const char * _String1, const char * _String2, size_t _MaxCount); 
#line 567
char *__cdecl strnset(char * _String, int _Value, size_t _MaxCount); 
#line 574
char *__cdecl strrev(char * _String); 
#line 579
char *__cdecl strset(char * _String, int _Value); 
#line 584
char *__cdecl strupr(char * _String); 
#line 592 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\string.h"
}__pragma( pack ( pop )) 
#line 13 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wtime.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 22
struct tm { 
#line 24
int tm_sec; 
#line 25
int tm_min; 
#line 26
int tm_hour; 
#line 27
int tm_mday; 
#line 28
int tm_mon; 
#line 29
int tm_year; 
#line 30
int tm_wday; 
#line 31
int tm_yday; 
#line 32
int tm_isdst; 
#line 33
}; 
#line 44
__wchar_t *__cdecl _wasctime(const tm * _Tm); 
#line 50
errno_t __cdecl _wasctime_s(__wchar_t * _Buffer, size_t _SizeInWords, const tm * _Tm); 
#line 56
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wasctime_s ( wchar_t ( & _Buffer ) [ _Size ], struct tm const * _Time ) throw ( ) { return _wasctime_s ( _Buffer, _Size, _Time ); }}
#line 66 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wtime.h"
size_t __cdecl wcsftime(__wchar_t * _Buffer, size_t _SizeInWords, const __wchar_t * _Format, const tm * _Tm); 
#line 75
size_t __cdecl _wcsftime_l(__wchar_t * _Buffer, size_t _SizeInWords, const __wchar_t * _Format, const tm * _Tm, _locale_t _Locale); 
#line 85
__wchar_t *__cdecl _wctime32(const __time32_t * _Time); 
#line 90
errno_t __cdecl _wctime32_s(__wchar_t * _Buffer, size_t _SizeInWords, const __time32_t * _Time); 
#line 96
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wctime32_s ( wchar_t ( & _Buffer ) [ _Size ], __time32_t const * _Time ) throw ( ) { return _wctime32_s ( _Buffer, _Size, _Time ); }}
#line 105 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wtime.h"
__wchar_t *__cdecl _wctime64(const __time64_t * _Time); 
#line 110
errno_t __cdecl _wctime64_s(__wchar_t * _Buffer, size_t _SizeInWords, const __time64_t * _Time); 
#line 115
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wctime64_s ( wchar_t ( & _Buffer ) [ _Size ], __time64_t const * _Time ) throw ( ) { return _wctime64_s ( _Buffer, _Size, _Time ); }}
#line 122 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wtime.h"
errno_t __cdecl _wstrdate_s(__wchar_t * _Buffer, size_t _SizeInWords); 
#line 127
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wstrdate_s ( wchar_t ( & _Buffer ) [ _Size ] ) throw ( ) { return _wstrdate_s ( _Buffer, _Size ); }}
#line 132 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wtime.h"
__wchar_t *__cdecl _wstrdate(__wchar_t * _Buffer); 
#line 138 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wtime.h"
errno_t __cdecl _wstrtime_s(__wchar_t * _Buffer, size_t _SizeInWords); 
#line 143
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wstrtime_s ( wchar_t ( & _Buffer ) [ _Size ] ) throw ( ) { return _wstrtime_s ( _Buffer, _Size ); }}
#line 148 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wtime.h"
__wchar_t *__cdecl _wstrtime(__wchar_t * _Buffer); 
#line 161 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wtime.h"
#pragma warning(push)
#pragma warning(disable: 4996)
#line 186
static __inline __wchar_t *__cdecl _wctime(const time_t *const 
#line 187
_Time) 
#line 188
{ 
#line 189
return _wctime64(_Time); 
#line 190
} 
#line 193
static __inline errno_t __cdecl _wctime_s(__wchar_t *const 
#line 194
_Buffer, const size_t 
#line 195
_SizeInWords, const time_t *const 
#line 196
_Time) 
#line 198
{ 
#line 199
return _wctime64_s(_Buffer, _SizeInWords, _Time); 
#line 200
} 
#line 208 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wtime.h"
}
#line 203 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wtime.h"
#pragma warning(pop)
#line 208 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wtime.h"
__pragma( pack ( pop )) 
#line 15 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\time.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 24
typedef long clock_t; 
#line 26
struct _timespec32 { 
#line 28
__time32_t tv_sec; 
#line 29
long tv_nsec; 
#line 30
}; 
#line 32
struct _timespec64 { 
#line 34
__time64_t tv_sec; 
#line 35
long tv_nsec; 
#line 36
}; 
#line 39
struct timespec { 
#line 41
time_t tv_sec; 
#line 42
long tv_nsec; 
#line 43
}; 
#line 62 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\time.h"
int *__cdecl __daylight(); 
#line 68
long *__cdecl __dstbias(); 
#line 74
long *__cdecl __timezone(); 
#line 80
char **__cdecl __tzname(); 
#line 85
errno_t __cdecl _get_daylight(int * _Daylight); 
#line 90
errno_t __cdecl _get_dstbias(long * _DaylightSavingsBias); 
#line 95
errno_t __cdecl _get_timezone(long * _TimeZone); 
#line 100
errno_t __cdecl _get_tzname(size_t * _ReturnValue, char * _Buffer, size_t _SizeInBytes, int _Index); 
#line 117
char *__cdecl asctime(const tm * _Tm); 
#line 124
errno_t __cdecl asctime_s(char * _Buffer, size_t _SizeInBytes, const tm * _Tm); 
#line 131 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\time.h"
extern "C++" {template < size_t _Size > inline errno_t __cdecl asctime_s ( char ( & _Buffer ) [ _Size ], struct tm const * _Time ) throw ( ) { return asctime_s ( _Buffer, _Size, _Time ); }}
#line 139 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\time.h"
clock_t __cdecl clock(); 
#line 144
char *__cdecl _ctime32(const __time32_t * _Time); 
#line 149
errno_t __cdecl _ctime32_s(char * _Buffer, size_t _SizeInBytes, const __time32_t * _Time); 
#line 155
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ctime32_s ( char ( & _Buffer ) [ _Size ], __time32_t const * _Time ) throw ( ) { return _ctime32_s ( _Buffer, _Size, _Time ); }}
#line 164 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\time.h"
char *__cdecl _ctime64(const __time64_t * _Time); 
#line 169
errno_t __cdecl _ctime64_s(char * _Buffer, size_t _SizeInBytes, const __time64_t * _Time); 
#line 175
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ctime64_s ( char ( & _Buffer ) [ _Size ], __time64_t const * _Time ) throw ( ) { return _ctime64_s ( _Buffer, _Size, _Time ); }}
#line 182 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\time.h"
double __cdecl _difftime32(__time32_t _Time1, __time32_t _Time2); 
#line 188
double __cdecl _difftime64(__time64_t _Time1, __time64_t _Time2); 
#line 195
tm *__cdecl _gmtime32(const __time32_t * _Time); 
#line 200
errno_t __cdecl _gmtime32_s(tm * _Tm, const __time32_t * _Time); 
#line 207
tm *__cdecl _gmtime64(const __time64_t * _Time); 
#line 212
errno_t __cdecl _gmtime64_s(tm * _Tm, const __time64_t * _Time); 
#line 219
tm *__cdecl _localtime32(const __time32_t * _Time); 
#line 224
errno_t __cdecl _localtime32_s(tm * _Tm, const __time32_t * _Time); 
#line 231
tm *__cdecl _localtime64(const __time64_t * _Time); 
#line 236
errno_t __cdecl _localtime64_s(tm * _Tm, const __time64_t * _Time); 
#line 242
__time32_t __cdecl _mkgmtime32(tm * _Tm); 
#line 247
__time64_t __cdecl _mkgmtime64(tm * _Tm); 
#line 252
__time32_t __cdecl _mktime32(tm * _Tm); 
#line 257
__time64_t __cdecl _mktime64(tm * _Tm); 
#line 263
size_t __cdecl strftime(char * _Buffer, size_t _SizeInBytes, const char * _Format, const tm * _Tm); 
#line 272
size_t __cdecl _strftime_l(char * _Buffer, size_t _MaxSize, const char * _Format, const tm * _Tm, _locale_t _Locale); 
#line 281
errno_t __cdecl _strdate_s(char * _Buffer, size_t _SizeInBytes); 
#line 286
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strdate_s ( char ( & _Buffer ) [ _Size ] ) throw ( ) { return _strdate_s ( _Buffer, _Size ); }}
#line 291 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\time.h"
char *__cdecl _strdate(char * _Buffer); 
#line 297 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\time.h"
errno_t __cdecl _strtime_s(char * _Buffer, size_t _SizeInBytes); 
#line 302
extern "C++" {template < size_t _Size > inline errno_t __cdecl _strtime_s ( char ( & _Buffer ) [ _Size ] ) throw ( ) { return _strtime_s ( _Buffer, _Size ); }}
#line 307 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\time.h"
char *__cdecl _strtime(char * _Buffer); 
#line 312 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\time.h"
__time32_t __cdecl _time32(__time32_t * _Time); 
#line 316
__time64_t __cdecl _time64(__time64_t * _Time); 
#line 322
int __cdecl _timespec32_get(_timespec32 * _Ts, int _Base); 
#line 329
int __cdecl _timespec64_get(_timespec64 * _Ts, int _Base); 
#line 343
void __cdecl _tzset(); 
#line 346
__declspec(deprecated("This function or variable has been superceded by newer library or operating system functionality. Consider using GetLocalTime in" "stead. See online help for details.")) unsigned __cdecl 
#line 347
_getsystime(tm * _Tm); 
#line 351
__declspec(deprecated("This function or variable has been superceded by newer library or operating system functionality. Consider using SetLocalTime in" "stead. See online help for details.")) unsigned __cdecl 
#line 352
_setsystime(tm * _Tm, unsigned _Milliseconds); 
#line 476 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\time.h"
static __inline char *__cdecl ctime(const time_t *const 
#line 477
_Time) 
#line 479
{ 
#pragma warning(push)
#pragma warning(disable: 4996)
return _ctime64(_Time); 
#pragma warning(pop)
} 
#line 487
static __inline double __cdecl difftime(const time_t 
#line 488
_Time1, const time_t 
#line 489
_Time2) 
#line 491
{ 
#line 492
return _difftime64(_Time1, _Time2); 
#line 493
} 
#line 496
static __inline tm *__cdecl gmtime(const time_t *const 
#line 497
_Time) 
#line 498
{ 
#pragma warning(push)
#pragma warning(disable: 4996)
return _gmtime64(_Time); 
#pragma warning(pop)
} 
#line 506
static __inline tm *__cdecl localtime(const time_t *const 
#line 507
_Time) 
#line 509
{ 
#pragma warning(push)
#pragma warning(disable: 4996)
return _localtime64(_Time); 
#pragma warning(pop)
} 
#line 517
static __inline time_t __cdecl _mkgmtime(tm *const 
#line 518
_Tm) 
#line 520
{ 
#line 521
return _mkgmtime64(_Tm); 
#line 522
} 
#line 525
static __inline time_t __cdecl mktime(tm *const 
#line 526
_Tm) 
#line 528
{ 
#line 529
return _mktime64(_Tm); 
#line 530
} 
#line 532
static __inline time_t __cdecl time(time_t *const 
#line 533
_Time) 
#line 535
{ 
#line 536
return _time64(_Time); 
#line 537
} 
#line 540
static __inline int __cdecl timespec_get(timespec *const 
#line 541
_Ts, const int 
#line 542
_Base) 
#line 544
{ 
#line 545
return _timespec64_get((_timespec64 *)_Ts, _Base); 
#line 546
} 
#line 550
static __inline errno_t __cdecl ctime_s(char *const 
#line 551
_Buffer, const size_t 
#line 552
_SizeInBytes, const time_t *const 
#line 553
_Time) 
#line 555
{ 
#line 556
return _ctime64_s(_Buffer, _SizeInBytes, _Time); 
#line 557
} 
#line 560
static __inline errno_t __cdecl gmtime_s(tm *const 
#line 561
_Tm, const time_t *const 
#line 562
_Time) 
#line 564
{ 
#line 565
return _gmtime64_s(_Tm, _Time); 
#line 566
} 
#line 569
static __inline errno_t __cdecl localtime_s(tm *const 
#line 570
_Tm, const time_t *const 
#line 571
_Time) 
#line 573
{ 
#line 574
return _localtime64_s(_Tm, _Time); 
#line 575
} 
#line 594 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\time.h"
void __cdecl tzset(); 
#line 601 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\time.h"
}__pragma( pack ( pop )) 
#line 88 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt/common_functions.h"
extern "C" {
#line 91 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt/common_functions.h"
extern clock_t __cdecl clock(); 
#line 96 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt/common_functions.h"
extern void *__cdecl memset(void *, int, size_t); 
#line 97
extern void *__cdecl memcpy(void *, const void *, size_t); 
#line 99
}
#line 115 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern "C" {
#line 213 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern int __cdecl abs(int a); 
#line 221
extern long __cdecl labs(long a); 
#line 229
extern __int64 llabs(__int64 a); 
#line 279 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl fabs(double x); 
#line 320
extern __inline float fabsf(float x); 
#line 330 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern inline int min(int a, int b); 
#line 337
extern inline unsigned umin(unsigned a, unsigned b); 
#line 344
extern inline __int64 llmin(__int64 a, __int64 b); 
#line 351
extern inline unsigned __int64 ullmin(unsigned __int64 a, unsigned __int64 b); 
#line 374 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float __cdecl fminf(float x, float y); 
#line 394 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl fmin(double x, double y); 
#line 405 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern inline int max(int a, int b); 
#line 413
extern inline unsigned umax(unsigned a, unsigned b); 
#line 420
extern inline __int64 llmax(__int64 a, __int64 b); 
#line 427
extern inline unsigned __int64 ullmax(unsigned __int64 a, unsigned __int64 b); 
#line 450 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float __cdecl fmaxf(float x, float y); 
#line 470 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl fmax(double, double); 
#line 512 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl sin(double x); 
#line 545
extern double __cdecl cos(double x); 
#line 564 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern void sincos(double x, double * sptr, double * cptr); 
#line 580
extern void sincosf(float x, float * sptr, float * cptr); 
#line 625 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl tan(double x); 
#line 694
extern double __cdecl sqrt(double x); 
#line 766 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double rsqrt(double x); 
#line 836
extern float rsqrtf(float x); 
#line 894 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl log2(double x); 
#line 919 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl exp2(double x); 
#line 944 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float __cdecl exp2f(float x); 
#line 969 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double exp10(double x); 
#line 992
extern float exp10f(float x); 
#line 1040 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl expm1(double x); 
#line 1085 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float __cdecl expm1f(float x); 
#line 1140 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float __cdecl log2f(float x); 
#line 1192 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl log10(double x); 
#line 1263
extern double __cdecl log(double x); 
#line 1368 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl log1p(double x); 
#line 1474 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float __cdecl log1pf(float x); 
#line 1536 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl floor(double x); 
#line 1575
extern double __cdecl exp(double x); 
#line 1606
extern double __cdecl cosh(double x); 
#line 1656
extern double __cdecl sinh(double x); 
#line 1686
extern double __cdecl tanh(double x); 
#line 1723 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl acosh(double x); 
#line 1761 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float __cdecl acoshf(float x); 
#line 1777 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl asinh(double x); 
#line 1793 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float __cdecl asinhf(float x); 
#line 1847 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl atanh(double x); 
#line 1901 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float __cdecl atanhf(float x); 
#line 1958 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl ldexp(double x, int exp); 
#line 2014
extern __inline float ldexpf(float x, int exp); 
#line 2068 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl logb(double x); 
#line 2123 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float __cdecl logbf(float x); 
#line 2154 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern int __cdecl ilogb(double x); 
#line 2185 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern int __cdecl ilogbf(float x); 
#line 2261 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl scalbn(double x, int n); 
#line 2337 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float __cdecl scalbnf(float x, int n); 
#line 2413 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl scalbln(double x, long n); 
#line 2489 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float __cdecl scalblnf(float x, long n); 
#line 2565 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl frexp(double x, int * nptr); 
#line 2640
extern __inline float frexpf(float x, int * nptr); 
#line 2656 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl round(double x); 
#line 2673 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float __cdecl roundf(float x); 
#line 2691 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern long __cdecl lround(double x); 
#line 2709 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern long __cdecl lroundf(float x); 
#line 2727 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern __int64 __cdecl llround(double x); 
#line 2745 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern __int64 __cdecl llroundf(float x); 
#line 2797 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float __cdecl rintf(float x); 
#line 2814 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern long __cdecl lrint(double x); 
#line 2831 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern long __cdecl lrintf(float x); 
#line 2848 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern __int64 __cdecl llrint(double x); 
#line 2865 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern __int64 __cdecl llrintf(float x); 
#line 2918 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl nearbyint(double x); 
#line 2971 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float __cdecl nearbyintf(float x); 
#line 3031 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl ceil(double x); 
#line 3045 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl trunc(double x); 
#line 3060 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float __cdecl truncf(float x); 
#line 3086 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl fdim(double x, double y); 
#line 3112 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float __cdecl fdimf(float x, float y); 
#line 3146 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl atan2(double y, double x); 
#line 3177
extern double __cdecl atan(double x); 
#line 3200
extern double __cdecl acos(double x); 
#line 3232
extern double __cdecl asin(double x); 
#line 3275 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl hypot(double x, double y); 
#line 3374 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
static __inline float __cdecl hypotf(float x, float y); 
#line 4110 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl cbrt(double x); 
#line 4196 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float __cdecl cbrtf(float x); 
#line 4249 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double rcbrt(double x); 
#line 4299
extern float rcbrtf(float x); 
#line 4359
extern double sinpi(double x); 
#line 4419
extern float sinpif(float x); 
#line 4471
extern double cospi(double x); 
#line 4523
extern float cospif(float x); 
#line 4553
extern void sincospi(double x, double * sptr, double * cptr); 
#line 4583
extern void sincospif(float x, float * sptr, float * cptr); 
#line 4895 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl pow(double x, double y); 
#line 4951
extern double __cdecl modf(double x, double * iptr); 
#line 5010
extern double __cdecl fmod(double x, double y); 
#line 5098 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl remainder(double x, double y); 
#line 5188 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float __cdecl remainderf(float x, float y); 
#line 5242 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl remquo(double x, double y, int * quo); 
#line 5296 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float __cdecl remquof(float x, float y, int * quo); 
#line 5335 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl j0(double x); 
#line 5377 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float j0f(float x); 
#line 5446 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl j1(double x); 
#line 5515 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float j1f(float x); 
#line 5558 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl jn(int n, double x); 
#line 5601 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float jnf(int n, float x); 
#line 5653 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl y0(double x); 
#line 5705 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float y0f(float x); 
#line 5757 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl y1(double x); 
#line 5809 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float y1f(float x); 
#line 5862 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl yn(int n, double x); 
#line 5915 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float ynf(int n, float x); 
#line 6106 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl erf(double x); 
#line 6188 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float __cdecl erff(float x); 
#line 6250 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double erfinv(double y); 
#line 6307
extern float erfinvf(float y); 
#line 6348 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl erfc(double x); 
#line 6386 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float __cdecl erfcf(float x); 
#line 6514 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl lgamma(double x); 
#line 6575 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double erfcinv(double y); 
#line 6631
extern float erfcinvf(float y); 
#line 6689
extern double normcdfinv(double y); 
#line 6747
extern float normcdfinvf(float y); 
#line 6790
extern double normcdf(double y); 
#line 6833
extern float normcdff(float y); 
#line 6908
extern double erfcx(double x); 
#line 6983
extern float erfcxf(float x); 
#line 7119 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float __cdecl lgammaf(float x); 
#line 7228 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl tgamma(double x); 
#line 7337 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float __cdecl tgammaf(float x); 
#line 7350 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl copysign(double x, double y); 
#line 7363 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float __cdecl copysignf(float x, float y); 
#line 7382 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl nextafter(double x, double y); 
#line 7401 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float __cdecl nextafterf(float x, float y); 
#line 7417 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl nan(const char * tagp); 
#line 7433 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float __cdecl nanf(const char * tagp); 
#line 7438 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern int __isinff(float); 
#line 7439
extern int __isnanf(float); 
#line 7449 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern int __finite(double); 
#line 7450
extern int __finitef(float); 
#line 7451
extern int __signbit(double); 
#line 7452
extern int __isnan(double); 
#line 7453
extern int __isinf(double); 
#line 7456 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern int __signbitf(float); 
#line 7617 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern double __cdecl fma(double x, double y, double z); 
#line 7775 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float __cdecl fmaf(float x, float y, float z); 
#line 7784 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern int __signbitl(long double); 
#line 7790 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern int __finitel(long double); 
#line 7791
extern int __isinfl(long double); 
#line 7792
extern int __isnanl(long double); 
#line 7796 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern float __cdecl acosf(float); 
#line 7797
extern float __cdecl asinf(float); 
#line 7798
extern float __cdecl atanf(float); 
#line 7799
extern float __cdecl atan2f(float, float); 
#line 7800
extern float __cdecl cosf(float); 
#line 7801
extern float __cdecl sinf(float); 
#line 7802
extern float __cdecl tanf(float); 
#line 7803
extern float __cdecl coshf(float); 
#line 7804
extern float __cdecl sinhf(float); 
#line 7805
extern float __cdecl tanhf(float); 
#line 7806
extern float __cdecl expf(float); 
#line 7807
extern float __cdecl logf(float); 
#line 7808
extern float __cdecl log10f(float); 
#line 7809
extern float __cdecl modff(float, float *); 
#line 7810
extern float __cdecl powf(float, float); 
#line 7811
extern float __cdecl sqrtf(float); 
#line 7812
extern float __cdecl ceilf(float); 
#line 7813
extern float __cdecl floorf(float); 
#line 7814
extern float __cdecl fmodf(float, float); 
#line 8951 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
}
#line 14 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_math.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 16
#pragma warning(push)
#pragma warning(disable:4738)
#pragma warning(disable:4820)
#line 25
struct _exception { 
#line 27
int type; 
#line 28
char *name; 
#line 29
double arg1; 
#line 30
double arg2; 
#line 31
double retval; 
#line 32
}; 
#line 39
struct _complex { 
#line 41
double x, y; 
#line 42
}; 
#line 61 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_math.h"
typedef float float_t; 
#line 62
typedef double double_t; 
#line 80 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_math.h"
extern const double _HUGE; 
#line 171 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_math.h"
void __cdecl _fperrraise(int _Except); 
#line 173
short __cdecl _dclass(double _X); 
#line 174
short __cdecl _ldclass(long double _X); 
#line 175
short __cdecl _fdclass(float _X); 
#line 177
int __cdecl _dsign(double _X); 
#line 178
int __cdecl _ldsign(long double _X); 
#line 179
int __cdecl _fdsign(float _X); 
#line 181
int __cdecl _dpcomp(double _X, double _Y); 
#line 182
int __cdecl _ldpcomp(long double _X, long double _Y); 
#line 183
int __cdecl _fdpcomp(float _X, float _Y); 
#line 185
short __cdecl _dtest(double * _Px); 
#line 186
short __cdecl _ldtest(long double * _Px); 
#line 187
short __cdecl _fdtest(float * _Px); 
#line 189
short __cdecl _d_int(double * _Px, short _Xexp); 
#line 190
short __cdecl _ld_int(long double * _Px, short _Xexp); 
#line 191
short __cdecl _fd_int(float * _Px, short _Xexp); 
#line 193
short __cdecl _dscale(double * _Px, long _Lexp); 
#line 194
short __cdecl _ldscale(long double * _Px, long _Lexp); 
#line 195
short __cdecl _fdscale(float * _Px, long _Lexp); 
#line 197
short __cdecl _dunscale(short * _Pex, double * _Px); 
#line 198
short __cdecl _ldunscale(short * _Pex, long double * _Px); 
#line 199
short __cdecl _fdunscale(short * _Pex, float * _Px); 
#line 201
short __cdecl _dexp(double * _Px, double _Y, long _Eoff); 
#line 202
short __cdecl _ldexp(long double * _Px, long double _Y, long _Eoff); 
#line 203
short __cdecl _fdexp(float * _Px, float _Y, long _Eoff); 
#line 205
short __cdecl _dnorm(unsigned short * _Ps); 
#line 206
short __cdecl _fdnorm(unsigned short * _Ps); 
#line 208
double __cdecl _dpoly(double _X, const double * _Tab, int _N); 
#line 209
long double __cdecl _ldpoly(long double _X, const long double * _Tab, int _N); 
#line 210
float __cdecl _fdpoly(float _X, const float * _Tab, int _N); 
#line 212
double __cdecl _dlog(double _X, int _Baseflag); 
#line 213
long double __cdecl _ldlog(long double _X, int _Baseflag); 
#line 214
float __cdecl _fdlog(float _X, int _Baseflag); 
#line 216
double __cdecl _dsin(double _X, unsigned _Qoff); 
#line 217
long double __cdecl _ldsin(long double _X, unsigned _Qoff); 
#line 218
float __cdecl _fdsin(float _X, unsigned _Qoff); 
#line 225
typedef 
#line 222
union { 
#line 223
unsigned short _Sh[4]; 
#line 224
double _Val; 
#line 225
} _double_val; 
#line 232
typedef 
#line 229
union { 
#line 230
unsigned short _Sh[2]; 
#line 231
float _Val; 
#line 232
} _float_val; 
#line 239
typedef 
#line 236
union { 
#line 237
unsigned short _Sh[4]; 
#line 238
long double _Val; 
#line 239
} _ldouble_val; 
#line 247
typedef 
#line 242
union { 
#line 243
unsigned short _Word[4]; 
#line 244
float _Float; 
#line 245
double _Double; 
#line 246
long double _Long_double; 
#line 247
} _float_const; 
#line 249
extern const _float_const _Denorm_C, _Inf_C, _Nan_C, _Snan_C, _Hugeval_C; 
#line 250
extern const _float_const _FDenorm_C, _FInf_C, _FNan_C, _FSnan_C; 
#line 251
extern const _float_const _LDenorm_C, _LInf_C, _LNan_C, _LSnan_C; 
#line 253
extern const _float_const _Eps_C, _Rteps_C; 
#line 254
extern const _float_const _FEps_C, _FRteps_C; 
#line 255
extern const _float_const _LEps_C, _LRteps_C; 
#line 257
extern const double _Zero_C, _Xbig_C; 
#line 258
extern const float _FZero_C, _FXbig_C; 
#line 259
extern const long double _LZero_C, _LXbig_C; 
#line 288
extern "C++" {
#line 290
inline int fpclassify(float _X) throw() 
#line 291
{ 
#line 292
return _fdtest(&_X); 
#line 293
} 
#line 295
inline int fpclassify(double _X) throw() 
#line 296
{ 
#line 297
return _dtest(&_X); 
#line 298
} 
#line 300
inline int fpclassify(long double _X) throw() 
#line 301
{ 
#line 302
return _ldtest(&_X); 
#line 303
} 
#line 305
inline bool signbit(float _X) throw() 
#line 306
{ 
#line 307
return _fdsign(_X) != 0; 
#line 308
} 
#line 310
inline bool signbit(double _X) throw() 
#line 311
{ 
#line 312
return _dsign(_X) != 0; 
#line 313
} 
#line 315
inline bool signbit(long double _X) throw() 
#line 316
{ 
#line 317
return _ldsign(_X) != 0; 
#line 318
} 
#line 320
inline int _fpcomp(float _X, float _Y) throw() 
#line 321
{ 
#line 322
return _fdpcomp(_X, _Y); 
#line 323
} 
#line 325
inline int _fpcomp(double _X, double _Y) throw() 
#line 326
{ 
#line 327
return _dpcomp(_X, _Y); 
#line 328
} 
#line 330
inline int _fpcomp(long double _X, long double _Y) throw() 
#line 331
{ 
#line 332
return _ldpcomp(_X, _Y); 
#line 333
} 
#line 335
template< class _Trc, class _Tre> struct _Combined_type { 
#line 337
typedef float _Type; 
#line 338
}; 
#line 340
template<> struct _Combined_type< float, double>  { 
#line 342
typedef double _Type; 
#line 343
}; 
#line 345
template<> struct _Combined_type< float, long double>  { 
#line 347
typedef long double _Type; 
#line 348
}; 
#line 350
template< class _Ty, class _T2> struct _Real_widened { 
#line 352
typedef long double _Type; 
#line 353
}; 
#line 355
template<> struct _Real_widened< float, float>  { 
#line 357
typedef float _Type; 
#line 358
}; 
#line 360
template<> struct _Real_widened< float, double>  { 
#line 362
typedef double _Type; 
#line 363
}; 
#line 365
template<> struct _Real_widened< double, float>  { 
#line 367
typedef double _Type; 
#line 368
}; 
#line 370
template<> struct _Real_widened< double, double>  { 
#line 372
typedef double _Type; 
#line 373
}; 
#line 375
template< class _Ty> struct _Real_type { 
#line 377
typedef double _Type; 
#line 378
}; 
#line 380
template<> struct _Real_type< float>  { 
#line 382
typedef float _Type; 
#line 383
}; 
#line 385
template<> struct _Real_type< long double>  { 
#line 387
typedef long double _Type; 
#line 388
}; 
#line 390
template < class _T1, class _T2 >
      inline int _fpcomp ( _T1 _X, _T2 _Y ) throw ( )
    {
        typedef typename _Combined_type < float,
            typename _Real_widened <
            typename _Real_type < _T1 > :: _Type,
            typename _Real_type < _T2 > :: _Type > :: _Type > :: _Type _Tw;
        return _fpcomp ( ( _Tw ) _X, ( _Tw ) _Y );
    }
#line 400
template < class _Ty >
      inline bool isfinite ( _Ty _X ) throw ( )
    {
        return fpclassify ( _X ) <= 0;
    }
#line 406
template < class _Ty >
      inline bool isinf ( _Ty _X ) throw ( )
    {
        return fpclassify ( _X ) == 1;
    }
#line 412
template < class _Ty >
      inline bool isnan ( _Ty _X ) throw ( )
    {
        return fpclassify ( _X ) == 2;
    }
#line 418
template < class _Ty >
      inline bool isnormal ( _Ty _X ) throw ( )
    {
        return fpclassify ( _X ) == ( - 1 );
    }
#line 424
template < class _Ty1, class _Ty2 >
      inline bool isgreater ( _Ty1 _X, _Ty2 _Y ) throw ( )
    {
        return ( _fpcomp ( _X, _Y ) & 4 ) != 0;
    }
#line 430
template < class _Ty1, class _Ty2 >
      inline bool isgreaterequal ( _Ty1 _X, _Ty2 _Y ) throw ( )
    {
        return ( _fpcomp ( _X, _Y ) & ( 2 | 4 ) ) != 0;
    }
#line 436
template < class _Ty1, class _Ty2 >
      inline bool isless ( _Ty1 _X, _Ty2 _Y ) throw ( )
    {
        return ( _fpcomp ( _X, _Y ) & 1 ) != 0;
    }
#line 442
template < class _Ty1, class _Ty2 >
      inline bool islessequal ( _Ty1 _X, _Ty2 _Y ) throw ( )
    {
        return ( _fpcomp ( _X, _Y ) & ( 1 | 2 ) ) != 0;
    }
#line 448
template < class _Ty1, class _Ty2 >
      inline bool islessgreater ( _Ty1 _X, _Ty2 _Y ) throw ( )
    {
        return ( _fpcomp ( _X, _Y ) & ( 1 | 4 ) ) != 0;
    }
#line 454
template < class _Ty1, class _Ty2 >
      inline bool isunordered ( _Ty1 _X, _Ty2 _Y ) throw ( )
    {
        return _fpcomp ( _X, _Y ) == 0;
    }
#line 459
}
#line 466 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_math.h"
int __cdecl abs(int _X); 
#line 467
long __cdecl labs(long _X); 
#line 468
__int64 __cdecl llabs(__int64 _X); 
#line 470
double __cdecl acos(double _X); 
#line 471
double __cdecl asin(double _X); 
#line 472
double __cdecl atan(double _X); 
#line 473
double __cdecl atan2(double _Y, double _X); 
#line 475
double __cdecl cos(double _X); 
#line 476
double __cdecl cosh(double _X); 
#line 477
double __cdecl exp(double _X); 
#line 478
double __cdecl fabs(double _X); 
#line 479
double __cdecl fmod(double _X, double _Y); 
#line 480
double __cdecl log(double _X); 
#line 481
double __cdecl log10(double _X); 
#line 482
double __cdecl pow(double _X, double _Y); 
#line 483
double __cdecl sin(double _X); 
#line 484
double __cdecl sinh(double _X); 
#line 485
double __cdecl sqrt(double _X); 
#line 486
double __cdecl tan(double _X); 
#line 487
double __cdecl tanh(double _X); 
#line 489
double __cdecl acosh(double _X); 
#line 490
double __cdecl asinh(double _X); 
#line 491
double __cdecl atanh(double _X); 
#line 492
double __cdecl atof(const char * _String); 
#line 493
double __cdecl _atof_l(const char * _String, _locale_t _Locale); 
#line 494
double __cdecl _cabs(_complex _Complex_value); 
#line 495
double __cdecl cbrt(double _X); 
#line 496
double __cdecl ceil(double _X); 
#line 497
double __cdecl _chgsign(double _X); 
#line 498
double __cdecl copysign(double _Number, double _Sign); 
#line 499
double __cdecl _copysign(double _Number, double _Sign); 
#line 500
double __cdecl erf(double _X); 
#line 501
double __cdecl erfc(double _X); 
#line 502
double __cdecl exp2(double _X); 
#line 503
double __cdecl expm1(double _X); 
#line 504
double __cdecl fdim(double _X, double _Y); 
#line 505
double __cdecl floor(double _X); 
#line 506
double __cdecl fma(double _X, double _Y, double _Z); 
#line 507
double __cdecl fmax(double _X, double _Y); 
#line 508
double __cdecl fmin(double _X, double _Y); 
#line 509
double __cdecl frexp(double _X, int * _Y); 
#line 510
double __cdecl hypot(double _X, double _Y); 
#line 511
double __cdecl _hypot(double _X, double _Y); 
#line 512
int __cdecl ilogb(double _X); 
#line 513
double __cdecl ldexp(double _X, int _Y); 
#line 514
double __cdecl lgamma(double _X); 
#line 515
__int64 __cdecl llrint(double _X); 
#line 516
__int64 __cdecl llround(double _X); 
#line 517
double __cdecl log1p(double _X); 
#line 518
double __cdecl log2(double _X); 
#line 519
double __cdecl logb(double _X); 
#line 520
long __cdecl lrint(double _X); 
#line 521
long __cdecl lround(double _X); 
#line 523
int __cdecl _matherr(_exception * _Except); 
#line 525
double __cdecl modf(double _X, double * _Y); 
#line 526
double __cdecl nan(const char * _X); 
#line 527
double __cdecl nearbyint(double _X); 
#line 528
double __cdecl nextafter(double _X, double _Y); 
#line 529
double __cdecl nexttoward(double _X, long double _Y); 
#line 530
double __cdecl remainder(double _X, double _Y); 
#line 531
double __cdecl remquo(double _X, double _Y, int * _Z); 
#line 532
double __cdecl rint(double _X); 
#line 533
double __cdecl round(double _X); 
#line 534
double __cdecl scalbln(double _X, long _Y); 
#line 535
double __cdecl scalbn(double _X, int _Y); 
#line 536
double __cdecl tgamma(double _X); 
#line 537
double __cdecl trunc(double _X); 
#line 538
double __cdecl _j0(double _X); 
#line 539
double __cdecl _j1(double _X); 
#line 540
double __cdecl _jn(int _X, double _Y); 
#line 541
double __cdecl _y0(double _X); 
#line 542
double __cdecl _y1(double _X); 
#line 543
double __cdecl _yn(int _X, double _Y); 
#line 545
float __cdecl acoshf(float _X); 
#line 546
float __cdecl asinhf(float _X); 
#line 547
float __cdecl atanhf(float _X); 
#line 548
float __cdecl cbrtf(float _X); 
#line 549
float __cdecl _chgsignf(float _X); 
#line 550
float __cdecl copysignf(float _Number, float _Sign); 
#line 551
float __cdecl _copysignf(float _Number, float _Sign); 
#line 552
float __cdecl erff(float _X); 
#line 553
float __cdecl erfcf(float _X); 
#line 554
float __cdecl expm1f(float _X); 
#line 555
float __cdecl exp2f(float _X); 
#line 556
float __cdecl fdimf(float _X, float _Y); 
#line 557
float __cdecl fmaf(float _X, float _Y, float _Z); 
#line 558
float __cdecl fmaxf(float _X, float _Y); 
#line 559
float __cdecl fminf(float _X, float _Y); 
#line 560
float __cdecl _hypotf(float _X, float _Y); 
#line 561
int __cdecl ilogbf(float _X); 
#line 562
float __cdecl lgammaf(float _X); 
#line 563
__int64 __cdecl llrintf(float _X); 
#line 564
__int64 __cdecl llroundf(float _X); 
#line 565
float __cdecl log1pf(float _X); 
#line 566
float __cdecl log2f(float _X); 
#line 567
float __cdecl logbf(float _X); 
#line 568
long __cdecl lrintf(float _X); 
#line 569
long __cdecl lroundf(float _X); 
#line 570
float __cdecl nanf(const char * _X); 
#line 571
float __cdecl nearbyintf(float _X); 
#line 572
float __cdecl nextafterf(float _X, float _Y); 
#line 573
float __cdecl nexttowardf(float _X, long double _Y); 
#line 574
float __cdecl remainderf(float _X, float _Y); 
#line 575
float __cdecl remquof(float _X, float _Y, int * _Z); 
#line 576
float __cdecl rintf(float _X); 
#line 577
float __cdecl roundf(float _X); 
#line 578
float __cdecl scalblnf(float _X, long _Y); 
#line 579
float __cdecl scalbnf(float _X, int _Y); 
#line 580
float __cdecl tgammaf(float _X); 
#line 581
float __cdecl truncf(float _X); 
#line 591 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_math.h"
float __cdecl _logbf(float _X); 
#line 592
float __cdecl _nextafterf(float _X, float _Y); 
#line 593
int __cdecl _finitef(float _X); 
#line 594
int __cdecl _isnanf(float _X); 
#line 595
int __cdecl _fpclassf(float _X); 
#line 597
int __cdecl _set_FMA3_enable(int _Flag); 
#line 598
int __cdecl _get_FMA3_enable(); 
#line 611 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_math.h"
float __cdecl acosf(float _X); 
#line 612
float __cdecl asinf(float _X); 
#line 613
float __cdecl atan2f(float _Y, float _X); 
#line 614
float __cdecl atanf(float _X); 
#line 615
float __cdecl ceilf(float _X); 
#line 616
float __cdecl cosf(float _X); 
#line 617
float __cdecl coshf(float _X); 
#line 618
float __cdecl expf(float _X); 
#line 670 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_math.h"
__inline float __cdecl fabsf(float _X) 
#line 671
{ 
#line 672
return (float)fabs(_X); 
#line 673
} 
#line 679 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_math.h"
float __cdecl floorf(float _X); 
#line 680
float __cdecl fmodf(float _X, float _Y); 
#line 696 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_math.h"
__inline float __cdecl frexpf(float _X, int *_Y) 
#line 697
{ 
#line 698
return (float)frexp(_X, _Y); 
#line 699
} 
#line 701
__inline float __cdecl hypotf(float _X, float _Y) 
#line 702
{ 
#line 703
return _hypotf(_X, _Y); 
#line 704
} 
#line 706
__inline float __cdecl ldexpf(float _X, int _Y) 
#line 707
{ 
#line 708
return (float)ldexp(_X, _Y); 
#line 709
} 
#line 713
float __cdecl log10f(float _X); 
#line 714
float __cdecl logf(float _X); 
#line 715
float __cdecl modff(float _X, float * _Y); 
#line 716
float __cdecl powf(float _X, float _Y); 
#line 717
float __cdecl sinf(float _X); 
#line 718
float __cdecl sinhf(float _X); 
#line 719
float __cdecl sqrtf(float _X); 
#line 720
float __cdecl tanf(float _X); 
#line 721
float __cdecl tanhf(float _X); 
#line 775 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_math.h"
long double __cdecl acoshl(long double _X); 
#line 777
__inline long double __cdecl acosl(long double _X) 
#line 778
{ 
#line 779
return acos((double)_X); 
#line 780
} 
#line 782
long double __cdecl asinhl(long double _X); 
#line 784
__inline long double __cdecl asinl(long double _X) 
#line 785
{ 
#line 786
return asin((double)_X); 
#line 787
} 
#line 789
__inline long double __cdecl atan2l(long double _Y, long double _X) 
#line 790
{ 
#line 791
return atan2((double)_Y, (double)_X); 
#line 792
} 
#line 794
long double __cdecl atanhl(long double _X); 
#line 796
__inline long double __cdecl atanl(long double _X) 
#line 797
{ 
#line 798
return atan((double)_X); 
#line 799
} 
#line 801
long double __cdecl cbrtl(long double _X); 
#line 803
__inline long double __cdecl ceill(long double _X) 
#line 804
{ 
#line 805
return ceil((double)_X); 
#line 806
} 
#line 808
__inline long double __cdecl _chgsignl(long double _X) 
#line 809
{ 
#line 810
return _chgsign((double)_X); 
#line 811
} 
#line 813
long double __cdecl copysignl(long double _Number, long double _Sign); 
#line 815
__inline long double __cdecl _copysignl(long double _Number, long double _Sign) 
#line 816
{ 
#line 817
return _copysign((double)_Number, (double)_Sign); 
#line 818
} 
#line 820
__inline long double __cdecl coshl(long double _X) 
#line 821
{ 
#line 822
return cosh((double)_X); 
#line 823
} 
#line 825
__inline long double __cdecl cosl(long double _X) 
#line 826
{ 
#line 827
return cos((double)_X); 
#line 828
} 
#line 830
long double __cdecl erfl(long double _X); 
#line 831
long double __cdecl erfcl(long double _X); 
#line 833
__inline long double __cdecl expl(long double _X) 
#line 834
{ 
#line 835
return exp((double)_X); 
#line 836
} 
#line 838
long double __cdecl exp2l(long double _X); 
#line 839
long double __cdecl expm1l(long double _X); 
#line 841
__inline long double __cdecl fabsl(long double _X) 
#line 842
{ 
#line 843
return fabs((double)_X); 
#line 844
} 
#line 846
long double __cdecl fdiml(long double _X, long double _Y); 
#line 848
__inline long double __cdecl floorl(long double _X) 
#line 849
{ 
#line 850
return floor((double)_X); 
#line 851
} 
#line 853
long double __cdecl fmal(long double _X, long double _Y, long double _Z); 
#line 854
long double __cdecl fmaxl(long double _X, long double _Y); 
#line 855
long double __cdecl fminl(long double _X, long double _Y); 
#line 857
__inline long double __cdecl fmodl(long double _X, long double _Y) 
#line 858
{ 
#line 859
return fmod((double)_X, (double)_Y); 
#line 860
} 
#line 862
__inline long double __cdecl frexpl(long double _X, int *_Y) 
#line 863
{ 
#line 864
return frexp((double)_X, _Y); 
#line 865
} 
#line 867
int __cdecl ilogbl(long double _X); 
#line 869
__inline long double __cdecl _hypotl(long double _X, long double _Y) 
#line 870
{ 
#line 871
return _hypot((double)_X, (double)_Y); 
#line 872
} 
#line 874
__inline long double __cdecl hypotl(long double _X, long double _Y) 
#line 875
{ 
#line 876
return _hypot((double)_X, (double)_Y); 
#line 877
} 
#line 879
__inline long double __cdecl ldexpl(long double _X, int _Y) 
#line 880
{ 
#line 881
return ldexp((double)_X, _Y); 
#line 882
} 
#line 884
long double __cdecl lgammal(long double _X); 
#line 885
__int64 __cdecl llrintl(long double _X); 
#line 886
__int64 __cdecl llroundl(long double _X); 
#line 888
__inline long double __cdecl logl(long double _X) 
#line 889
{ 
#line 890
return log((double)_X); 
#line 891
} 
#line 893
__inline long double __cdecl log10l(long double _X) 
#line 894
{ 
#line 895
return log10((double)_X); 
#line 896
} 
#line 898
long double __cdecl log1pl(long double _X); 
#line 899
long double __cdecl log2l(long double _X); 
#line 900
long double __cdecl logbl(long double _X); 
#line 901
long __cdecl lrintl(long double _X); 
#line 902
long __cdecl lroundl(long double _X); 
#line 904
__inline long double __cdecl modfl(long double _X, long double *_Y) 
#line 905
{ 
#line 906
double _F, _I; 
#line 907
_F = modf((double)_X, &_I); 
#line 908
(*_Y) = _I; 
#line 909
return _F; 
#line 910
} 
#line 912
long double __cdecl nanl(const char * _X); 
#line 913
long double __cdecl nearbyintl(long double _X); 
#line 914
long double __cdecl nextafterl(long double _X, long double _Y); 
#line 915
long double __cdecl nexttowardl(long double _X, long double _Y); 
#line 917
__inline long double __cdecl powl(long double _X, long double _Y) 
#line 918
{ 
#line 919
return pow((double)_X, (double)_Y); 
#line 920
} 
#line 922
long double __cdecl remainderl(long double _X, long double _Y); 
#line 923
long double __cdecl remquol(long double _X, long double _Y, int * _Z); 
#line 924
long double __cdecl rintl(long double _X); 
#line 925
long double __cdecl roundl(long double _X); 
#line 926
long double __cdecl scalblnl(long double _X, long _Y); 
#line 927
long double __cdecl scalbnl(long double _X, int _Y); 
#line 929
__inline long double __cdecl sinhl(long double _X) 
#line 930
{ 
#line 931
return sinh((double)_X); 
#line 932
} 
#line 934
__inline long double __cdecl sinl(long double _X) 
#line 935
{ 
#line 936
return sin((double)_X); 
#line 937
} 
#line 939
__inline long double __cdecl sqrtl(long double _X) 
#line 940
{ 
#line 941
return sqrt((double)_X); 
#line 942
} 
#line 944
__inline long double __cdecl tanhl(long double _X) 
#line 945
{ 
#line 946
return tanh((double)_X); 
#line 947
} 
#line 949
__inline long double __cdecl tanl(long double _X) 
#line 950
{ 
#line 951
return tan((double)_X); 
#line 952
} 
#line 954
long double __cdecl tgammal(long double _X); 
#line 955
long double __cdecl truncl(long double _X); 
#line 976 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_math.h"
extern double HUGE; 
#line 981 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_math.h"
double __cdecl j0(double _X); 
#line 982
double __cdecl j1(double _X); 
#line 983
double __cdecl jn(int _X, double _Y); 
#line 984
double __cdecl y0(double _X); 
#line 985
double __cdecl y1(double _X); 
#line 986
double __cdecl yn(int _X, double _Y); 
#line 994 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_math.h"
}
#line 992
#pragma warning(pop)
#line 994
__pragma( pack ( pop )) 
#line 13 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_malloc.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 54 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_malloc.h"
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 55
_calloc_base(size_t _Count, size_t _Size); 
#line 61
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 62
calloc(size_t _Count, size_t _Size); 
#line 68
int __cdecl _callnewh(size_t _Size); 
#line 73
__declspec(allocator) void *__cdecl 
#line 74
_expand(void * _Block, size_t _Size); 
#line 80
void __cdecl _free_base(void * _Block); 
#line 85
void __cdecl free(void * _Block); 
#line 90
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 91
_malloc_base(size_t _Size); 
#line 96
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 97
malloc(size_t _Size); 
#line 103
size_t __cdecl _msize_base(void * _Block); 
#line 109
size_t __cdecl _msize(void * _Block); 
#line 114
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 115
_realloc_base(void * _Block, size_t _Size); 
#line 121
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 122
realloc(void * _Block, size_t _Size); 
#line 128
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 129
_recalloc_base(void * _Block, size_t _Count, size_t _Size); 
#line 136
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 137
_recalloc(void * _Block, size_t _Count, size_t _Size); 
#line 144
void __cdecl _aligned_free(void * _Block); 
#line 149
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 150
_aligned_malloc(size_t _Size, size_t _Alignment); 
#line 156
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 157
_aligned_offset_malloc(size_t _Size, size_t _Alignment, size_t _Offset); 
#line 165
size_t __cdecl _aligned_msize(void * _Block, size_t _Alignment, size_t _Offset); 
#line 172
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 173
_aligned_offset_realloc(void * _Block, size_t _Size, size_t _Alignment, size_t _Offset); 
#line 181
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 182
_aligned_offset_recalloc(void * _Block, size_t _Count, size_t _Size, size_t _Alignment, size_t _Offset); 
#line 191
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 192
_aligned_realloc(void * _Block, size_t _Size, size_t _Alignment); 
#line 199
__declspec(allocator) __declspec(restrict) void *__cdecl 
#line 200
_aligned_recalloc(void * _Block, size_t _Count, size_t _Size, size_t _Alignment); 
#line 228 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_malloc.h"
}__pragma( pack ( pop )) 
#line 16 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_search.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 19
typedef int (__cdecl *_CoreCrtSecureSearchSortCompareFunction)(void *, const void *, const void *); 
#line 20
typedef int (__cdecl *_CoreCrtNonSecureSearchSortCompareFunction)(const void *, const void *); 
#line 26
void *__cdecl bsearch_s(const void * _Key, const void * _Base, rsize_t _NumOfElements, rsize_t _SizeOfElements, _CoreCrtSecureSearchSortCompareFunction _CompareFunction, void * _Context); 
#line 35
void __cdecl qsort_s(void * _Base, rsize_t _NumOfElements, rsize_t _SizeOfElements, _CoreCrtSecureSearchSortCompareFunction _CompareFunction, void * _Context); 
#line 48 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_search.h"
void *__cdecl bsearch(const void * _Key, const void * _Base, size_t _NumOfElements, size_t _SizeOfElements, _CoreCrtNonSecureSearchSortCompareFunction _CompareFunction); 
#line 56
void __cdecl qsort(void * _Base, size_t _NumOfElements, size_t _SizeOfElements, _CoreCrtNonSecureSearchSortCompareFunction _CompareFunction); 
#line 64
void *__cdecl _lfind_s(const void * _Key, const void * _Base, unsigned * _NumOfElements, size_t _SizeOfElements, _CoreCrtSecureSearchSortCompareFunction _CompareFunction, void * _Context); 
#line 74
void *__cdecl _lfind(const void * _Key, const void * _Base, unsigned * _NumOfElements, unsigned _SizeOfElements, _CoreCrtNonSecureSearchSortCompareFunction _CompareFunction); 
#line 83
void *__cdecl _lsearch_s(const void * _Key, void * _Base, unsigned * _NumOfElements, size_t _SizeOfElements, _CoreCrtSecureSearchSortCompareFunction _CompareFunction, void * _Context); 
#line 93
void *__cdecl _lsearch(const void * _Key, void * _Base, unsigned * _NumOfElements, unsigned _SizeOfElements, _CoreCrtNonSecureSearchSortCompareFunction _CompareFunction); 
#line 191 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_search.h"
void *__cdecl lfind(const void * _Key, const void * _Base, unsigned * _NumOfElements, unsigned _SizeOfElements, _CoreCrtNonSecureSearchSortCompareFunction _CompareFunction); 
#line 200
void *__cdecl lsearch(const void * _Key, void * _Base, unsigned * _NumOfElements, unsigned _SizeOfElements, _CoreCrtNonSecureSearchSortCompareFunction _CompareFunction); 
#line 212 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_search.h"
}__pragma( pack ( pop )) 
#line 13 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstdlib.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 50
errno_t __cdecl _itow_s(int _Value, __wchar_t * _Buffer, size_t _BufferCount, int _Radix); 
#line 57
extern "C++" {template < size_t _Size > inline errno_t __cdecl _itow_s ( int _Value, wchar_t ( & _Buffer ) [ _Size ], int _Radix ) throw ( ) { return _itow_s ( _Value, _Buffer, _Size, _Radix ); }}
#line 65 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstdlib.h"
__wchar_t *__cdecl _itow(int _Value, __wchar_t * _Buffer, int _Radix); 
#line 74 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstdlib.h"
errno_t __cdecl _ltow_s(long _Value, __wchar_t * _Buffer, size_t _BufferCount, int _Radix); 
#line 81
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ltow_s ( long _Value, wchar_t ( & _Buffer ) [ _Size ], int _Radix ) throw ( ) { return _ltow_s ( _Value, _Buffer, _Size, _Radix ); }}
#line 88 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstdlib.h"
__wchar_t *__cdecl _ltow(long _Value, __wchar_t * _Buffer, int _Radix); 
#line 96 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstdlib.h"
errno_t __cdecl _ultow_s(unsigned long _Value, __wchar_t * _Buffer, size_t _BufferCount, int _Radix); 
#line 103
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ultow_s ( unsigned long _Value, wchar_t ( & _Buffer ) [ _Size ], int _Radix ) throw ( ) { return _ultow_s ( _Value, _Buffer, _Size, _Radix ); }}
#line 110 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstdlib.h"
__wchar_t *__cdecl _ultow(unsigned long _Value, __wchar_t * _Buffer, int _Radix); 
#line 118 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstdlib.h"
double __cdecl wcstod(const __wchar_t * _String, __wchar_t ** _EndPtr); 
#line 124
double __cdecl _wcstod_l(const __wchar_t * _String, __wchar_t ** _EndPtr, _locale_t _Locale); 
#line 131
long __cdecl wcstol(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 138
long __cdecl _wcstol_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 146
__int64 __cdecl wcstoll(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 153
__int64 __cdecl _wcstoll_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 161
unsigned long __cdecl wcstoul(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 168
unsigned long __cdecl _wcstoul_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 176
unsigned __int64 __cdecl wcstoull(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 183
unsigned __int64 __cdecl _wcstoull_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 191
long double __cdecl wcstold(const __wchar_t * _String, __wchar_t ** _EndPtr); 
#line 197
long double __cdecl _wcstold_l(const __wchar_t * _String, __wchar_t ** _EndPtr, _locale_t _Locale); 
#line 204
float __cdecl wcstof(const __wchar_t * _String, __wchar_t ** _EndPtr); 
#line 210
float __cdecl _wcstof_l(const __wchar_t * _String, __wchar_t ** _EndPtr, _locale_t _Locale); 
#line 217
double __cdecl _wtof(const __wchar_t * _String); 
#line 222
double __cdecl _wtof_l(const __wchar_t * _String, _locale_t _Locale); 
#line 228
int __cdecl _wtoi(const __wchar_t * _String); 
#line 233
int __cdecl _wtoi_l(const __wchar_t * _String, _locale_t _Locale); 
#line 239
long __cdecl _wtol(const __wchar_t * _String); 
#line 244
long __cdecl _wtol_l(const __wchar_t * _String, _locale_t _Locale); 
#line 250
__int64 __cdecl _wtoll(const __wchar_t * _String); 
#line 255
__int64 __cdecl _wtoll_l(const __wchar_t * _String, _locale_t _Locale); 
#line 261
errno_t __cdecl _i64tow_s(__int64 _Value, __wchar_t * _Buffer, size_t _BufferCount, int _Radix); 
#line 269
__wchar_t *__cdecl _i64tow(__int64 _Value, __wchar_t * _Buffer, int _Radix); 
#line 276
errno_t __cdecl _ui64tow_s(unsigned __int64 _Value, __wchar_t * _Buffer, size_t _BufferCount, int _Radix); 
#line 284
__wchar_t *__cdecl _ui64tow(unsigned __int64 _Value, __wchar_t * _Buffer, int _Radix); 
#line 291
__int64 __cdecl _wtoi64(const __wchar_t * _String); 
#line 296
__int64 __cdecl _wtoi64_l(const __wchar_t * _String, _locale_t _Locale); 
#line 302
__int64 __cdecl _wcstoi64(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 309
__int64 __cdecl _wcstoi64_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 317
unsigned __int64 __cdecl _wcstoui64(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix); 
#line 324
unsigned __int64 __cdecl _wcstoui64_l(const __wchar_t * _String, __wchar_t ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 336
__declspec(allocator) __wchar_t *__cdecl _wfullpath(__wchar_t * _Buffer, const __wchar_t * _Path, size_t _BufferCount); 
#line 345
errno_t __cdecl _wmakepath_s(__wchar_t * _Buffer, size_t _BufferCount, const __wchar_t * _Drive, const __wchar_t * _Dir, const __wchar_t * _Filename, const __wchar_t * _Ext); 
#line 354
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wmakepath_s ( wchar_t ( & _Buffer ) [ _Size ], wchar_t const * _Drive, wchar_t const * _Dir, wchar_t const * _Filename, wchar_t const * _Ext ) throw ( ) { return _wmakepath_s ( _Buffer, _Size, _Drive, _Dir, _Filename, _Ext ); }}
#line 363 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstdlib.h"
#pragma warning(push)
#pragma warning(disable: 28719)
#pragma warning(disable: 28726)
void __cdecl _wmakepath(__wchar_t * _Buffer, const __wchar_t * _Drive, const __wchar_t * _Dir, const __wchar_t * _Filename, const __wchar_t * _Ext); 
#line 374 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstdlib.h"
#pragma warning(pop)
#line 376
void __cdecl _wperror(const __wchar_t * _ErrorMessage); 
#line 381
void __cdecl _wsplitpath(const __wchar_t * _FullPath, __wchar_t * _Drive, __wchar_t * _Dir, __wchar_t * _Filename, __wchar_t * _Ext); 
#line 389
errno_t __cdecl _wsplitpath_s(const __wchar_t * _FullPath, __wchar_t * _Drive, size_t _DriveCount, __wchar_t * _Dir, size_t _DirCount, __wchar_t * _Filename, size_t _FilenameCount, __wchar_t * _Ext, size_t _ExtCount); 
#line 401
extern "C++" {template < size_t _DriveSize, size_t _DirSize, size_t _NameSize, size_t _ExtSize > inline errno_t __cdecl _wsplitpath_s ( wchar_t const * _Path, wchar_t ( & _Drive ) [ _DriveSize ], wchar_t ( & _Dir ) [ _DirSize ], wchar_t ( & _Name ) [ _NameSize ], wchar_t ( & _Ext ) [ _ExtSize ] ) throw ( ) { return _wsplitpath_s ( _Path, _Drive, _DriveSize, _Dir, _DirSize, _Name, _NameSize, _Ext, _ExtSize ); }}
#line 410 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstdlib.h"
errno_t __cdecl _wdupenv_s(__wchar_t ** _Buffer, size_t * _BufferCount, const __wchar_t * _VarName); 
#line 419
__wchar_t *__cdecl _wgetenv(const __wchar_t * _VarName); 
#line 425
errno_t __cdecl _wgetenv_s(size_t * _RequiredCount, __wchar_t * _Buffer, size_t _BufferCount, const __wchar_t * _VarName); 
#line 432
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wgetenv_s ( size_t * _RequiredCount, wchar_t ( & _Buffer ) [ _Size ], wchar_t const * _VarName ) throw ( ) { return _wgetenv_s ( _RequiredCount, _Buffer, _Size, _VarName ); }}
#line 441 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstdlib.h"
int __cdecl _wputenv(const __wchar_t * _EnvString); 
#line 446
errno_t __cdecl _wputenv_s(const __wchar_t * _Name, const __wchar_t * _Value); 
#line 451
errno_t __cdecl _wsearchenv_s(const __wchar_t * _Filename, const __wchar_t * _VarName, __wchar_t * _Buffer, size_t _BufferCount); 
#line 458
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wsearchenv_s ( wchar_t const * _Filename, wchar_t const * _VarName, wchar_t ( & _ResultPath ) [ _Size ] ) throw ( ) { return _wsearchenv_s ( _Filename, _VarName, _ResultPath, _Size ); }}
#line 465 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstdlib.h"
void __cdecl _wsearchenv(const __wchar_t * _Filename, const __wchar_t * _VarName, __wchar_t * _ResultPath); 
#line 472 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstdlib.h"
int __cdecl _wsystem(const __wchar_t * _Command); 
#line 479 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\corecrt_wstdlib.h"
}__pragma( pack ( pop )) 
#line 18 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 34 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
void __cdecl _swab(char * _Buf1, char * _Buf2, int _SizeInBytes); 
#line 52
__declspec(noreturn) void __cdecl exit(int _Code); 
#line 53
__declspec(noreturn) void __cdecl _exit(int _Code); 
#line 54
__declspec(noreturn) void __cdecl _Exit(int _Code); 
#line 55
__declspec(noreturn) void __cdecl quick_exit(int _Code); 
#line 56
__declspec(noreturn) void __cdecl abort(); 
#line 63 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
unsigned __cdecl _set_abort_behavior(unsigned _Flags, unsigned _Mask); 
#line 73
typedef int (__cdecl *_onexit_t)(void); 
#line 140 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
int __cdecl atexit(void (__cdecl *)(void)); 
#line 141
_onexit_t __cdecl _onexit(_onexit_t _Func); 
#line 144 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
int __cdecl at_quick_exit(void (__cdecl *)(void)); 
#line 155
typedef void (__cdecl *_purecall_handler)(void); 
#line 158
typedef void (__cdecl *_invalid_parameter_handler)(const __wchar_t *, const __wchar_t *, const __wchar_t *, unsigned, uintptr_t); 
#line 167
_purecall_handler __cdecl _set_purecall_handler(_purecall_handler _Handler); 
#line 171
_purecall_handler __cdecl _get_purecall_handler(); 
#line 174
_invalid_parameter_handler __cdecl _set_invalid_parameter_handler(_invalid_parameter_handler _Handler); 
#line 178
_invalid_parameter_handler __cdecl _get_invalid_parameter_handler(); 
#line 180
_invalid_parameter_handler __cdecl _set_thread_local_invalid_parameter_handler(_invalid_parameter_handler _Handler); 
#line 184
_invalid_parameter_handler __cdecl _get_thread_local_invalid_parameter_handler(); 
#line 208 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
int __cdecl _set_error_mode(int _Mode); 
#line 213
int *__cdecl _errno(); 
#line 216
errno_t __cdecl _set_errno(int _Value); 
#line 217
errno_t __cdecl _get_errno(int * _Value); 
#line 219
unsigned long *__cdecl __doserrno(); 
#line 222
errno_t __cdecl _set_doserrno(unsigned long _Value); 
#line 223
errno_t __cdecl _get_doserrno(unsigned long * _Value); 
#line 226
char **__cdecl __sys_errlist(); 
#line 229
int *__cdecl __sys_nerr(); 
#line 232
void __cdecl perror(const char * _ErrMsg); 
#line 238 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
char **__cdecl __p__pgmptr(); 
#line 239
__wchar_t **__cdecl __p__wpgmptr(); 
#line 240
int *__cdecl __p__fmode(); 
#line 255 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
errno_t __cdecl _get_pgmptr(char ** _Value); 
#line 258
errno_t __cdecl _get_wpgmptr(__wchar_t ** _Value); 
#line 260
errno_t __cdecl _set_fmode(int _Mode); 
#line 262
errno_t __cdecl _get_fmode(int * _PMode); 
#line 275
typedef 
#line 271
struct _div_t { 
#line 273
int quot; 
#line 274
int rem; 
#line 275
} div_t; 
#line 281
typedef 
#line 277
struct _ldiv_t { 
#line 279
long quot; 
#line 280
long rem; 
#line 281
} ldiv_t; 
#line 287
typedef 
#line 283
struct _lldiv_t { 
#line 285
__int64 quot; 
#line 286
__int64 rem; 
#line 287
} lldiv_t; 
#line 289
int __cdecl abs(int _Number); 
#line 290
long __cdecl labs(long _Number); 
#line 291
__int64 __cdecl llabs(__int64 _Number); 
#line 292
__int64 __cdecl _abs64(__int64 _Number); 
#line 294
unsigned short __cdecl _byteswap_ushort(unsigned short _Number); 
#line 295
unsigned long __cdecl _byteswap_ulong(unsigned long _Number); 
#line 296
unsigned __int64 __cdecl _byteswap_uint64(unsigned __int64 _Number); 
#line 298
div_t __cdecl div(int _Numerator, int _Denominator); 
#line 299
ldiv_t __cdecl ldiv(long _Numerator, long _Denominator); 
#line 300
lldiv_t __cdecl lldiv(__int64 _Numerator, __int64 _Denominator); 
#line 304
#pragma warning (push)
#pragma warning (disable:6540)
#line 307
unsigned __cdecl _rotl(unsigned _Value, int _Shift); 
#line 313
unsigned long __cdecl _lrotl(unsigned long _Value, int _Shift); 
#line 318
unsigned __int64 __cdecl _rotl64(unsigned __int64 _Value, int _Shift); 
#line 323
unsigned __cdecl _rotr(unsigned _Value, int _Shift); 
#line 329
unsigned long __cdecl _lrotr(unsigned long _Value, int _Shift); 
#line 334
unsigned __int64 __cdecl _rotr64(unsigned __int64 _Value, int _Shift); 
#line 339
#pragma warning (pop)
#line 346
void __cdecl srand(unsigned _Seed); 
#line 348
int __cdecl rand(); 
#line 357 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
extern "C++" {
#line 359
inline long abs(const long _X) throw() 
#line 360
{ 
#line 361
return labs(_X); 
#line 362
} 
#line 364
inline __int64 abs(const __int64 _X) throw() 
#line 365
{ 
#line 366
return llabs(_X); 
#line 367
} 
#line 369
inline ldiv_t div(const long _A1, const long _A2) throw() 
#line 370
{ 
#line 371
return ldiv(_A1, _A2); 
#line 372
} 
#line 374
inline lldiv_t div(const __int64 _A1, const __int64 _A2) throw() 
#line 375
{ 
#line 376
return lldiv(_A1, _A2); 
#line 377
} 
#line 378
}
#line 390 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
#pragma pack ( push, 4 )
#line 394
typedef 
#line 392
struct { 
#line 393
unsigned char ld[10]; 
#line 394
} _LDOUBLE; 
#pragma pack ( pop )
#line 414 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
typedef 
#line 412
struct { 
#line 413
double x; 
#line 414
} _CRT_DOUBLE; 
#line 419
typedef 
#line 417
struct { 
#line 418
float f; 
#line 419
} _CRT_FLOAT; 
#line 428
typedef 
#line 426
struct { 
#line 427
long double x; 
#line 428
} _LONGDOUBLE; 
#line 432
#pragma pack ( push, 4 )
#line 436
typedef 
#line 434
struct { 
#line 435
unsigned char ld12[12]; 
#line 436
} _LDBL12; 
#pragma pack ( pop )
#line 446
double __cdecl atof(const char * _String); 
#line 447
int __cdecl atoi(const char * _String); 
#line 448
long __cdecl atol(const char * _String); 
#line 449
__int64 __cdecl atoll(const char * _String); 
#line 450
__int64 __cdecl _atoi64(const char * _String); 
#line 452
double __cdecl _atof_l(const char * _String, _locale_t _Locale); 
#line 453
int __cdecl _atoi_l(const char * _String, _locale_t _Locale); 
#line 454
long __cdecl _atol_l(const char * _String, _locale_t _Locale); 
#line 455
__int64 __cdecl _atoll_l(const char * _String, _locale_t _Locale); 
#line 456
__int64 __cdecl _atoi64_l(const char * _String, _locale_t _Locale); 
#line 458
int __cdecl _atoflt(_CRT_FLOAT * _Result, const char * _String); 
#line 459
int __cdecl _atodbl(_CRT_DOUBLE * _Result, char * _String); 
#line 460
int __cdecl _atoldbl(_LDOUBLE * _Result, char * _String); 
#line 463
int __cdecl _atoflt_l(_CRT_FLOAT * _Result, const char * _String, _locale_t _Locale); 
#line 470
int __cdecl _atodbl_l(_CRT_DOUBLE * _Result, char * _String, _locale_t _Locale); 
#line 478
int __cdecl _atoldbl_l(_LDOUBLE * _Result, char * _String, _locale_t _Locale); 
#line 485
float __cdecl strtof(const char * _String, char ** _EndPtr); 
#line 491
float __cdecl _strtof_l(const char * _String, char ** _EndPtr, _locale_t _Locale); 
#line 498
double __cdecl strtod(const char * _String, char ** _EndPtr); 
#line 504
double __cdecl _strtod_l(const char * _String, char ** _EndPtr, _locale_t _Locale); 
#line 511
long double __cdecl strtold(const char * _String, char ** _EndPtr); 
#line 517
long double __cdecl _strtold_l(const char * _String, char ** _EndPtr, _locale_t _Locale); 
#line 524
long __cdecl strtol(const char * _String, char ** _EndPtr, int _Radix); 
#line 531
long __cdecl _strtol_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 539
__int64 __cdecl strtoll(const char * _String, char ** _EndPtr, int _Radix); 
#line 546
__int64 __cdecl _strtoll_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 554
unsigned long __cdecl strtoul(const char * _String, char ** _EndPtr, int _Radix); 
#line 561
unsigned long __cdecl _strtoul_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 569
unsigned __int64 __cdecl strtoull(const char * _String, char ** _EndPtr, int _Radix); 
#line 576
unsigned __int64 __cdecl _strtoull_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 584
__int64 __cdecl _strtoi64(const char * _String, char ** _EndPtr, int _Radix); 
#line 591
__int64 __cdecl _strtoi64_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 599
unsigned __int64 __cdecl _strtoui64(const char * _String, char ** _EndPtr, int _Radix); 
#line 606
unsigned __int64 __cdecl _strtoui64_l(const char * _String, char ** _EndPtr, int _Radix, _locale_t _Locale); 
#line 622
errno_t __cdecl _itoa_s(int _Value, char * _Buffer, size_t _BufferCount, int _Radix); 
#line 629
extern "C++" {template < size_t _Size > inline errno_t __cdecl _itoa_s ( int _Value, char ( & _Buffer ) [ _Size ], int _Radix ) throw ( ) { return _itoa_s ( _Value, _Buffer, _Size, _Radix ); }}
#line 638 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
#pragma warning(push)
#pragma warning(disable: 28719)
#pragma warning(disable: 28726)
char *__cdecl _itoa(int _Value, char * _Buffer, int _Radix); 
#line 647 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
#pragma warning(pop)
#line 651
errno_t __cdecl _ltoa_s(long _Value, char * _Buffer, size_t _BufferCount, int _Radix); 
#line 658
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ltoa_s ( long _Value, char ( & _Buffer ) [ _Size ], int _Radix ) throw ( ) { return _ltoa_s ( _Value, _Buffer, _Size, _Radix ); }}
#line 665 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
char *__cdecl _ltoa(long _Value, char * _Buffer, int _Radix); 
#line 674 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
errno_t __cdecl _ultoa_s(unsigned long _Value, char * _Buffer, size_t _BufferCount, int _Radix); 
#line 681
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ultoa_s ( unsigned long _Value, char ( & _Buffer ) [ _Size ], int _Radix ) throw ( ) { return _ultoa_s ( _Value, _Buffer, _Size, _Radix ); }}
#line 688 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
#pragma warning(push)
#pragma warning(disable: 28726)
char *__cdecl _ultoa(unsigned long _Value, char * _Buffer, int _Radix); 
#line 696 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
#pragma warning(pop)
#line 700
errno_t __cdecl _i64toa_s(__int64 _Value, char * _Buffer, size_t _BufferCount, int _Radix); 
#line 709
char *__cdecl _i64toa(__int64 _Value, char * _Buffer, int _Radix); 
#line 717
errno_t __cdecl _ui64toa_s(unsigned __int64 _Value, char * _Buffer, size_t _BufferCount, int _Radix); 
#line 725
char *__cdecl _ui64toa(unsigned __int64 _Value, char * _Buffer, int _Radix); 
#line 745
errno_t __cdecl _ecvt_s(char * _Buffer, size_t _BufferCount, double _Value, int _DigitCount, int * _PtDec, int * _PtSign); 
#line 754
extern "C++" {template < size_t _Size > inline errno_t __cdecl _ecvt_s ( char ( & _Buffer ) [ _Size ], double _Value, int _DigitCount, int * _PtDec, int * _PtSign ) throw ( ) { return _ecvt_s ( _Buffer, _Size, _Value, _DigitCount, _PtDec, _PtSign ); }}
#line 764 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
char *__cdecl _ecvt(double _Value, int _DigitCount, int * _PtDec, int * _PtSign); 
#line 773
errno_t __cdecl _fcvt_s(char * _Buffer, size_t _BufferCount, double _Value, int _FractionalDigitCount, int * _PtDec, int * _PtSign); 
#line 782
extern "C++" {template < size_t _Size > inline errno_t __cdecl _fcvt_s ( char ( & _Buffer ) [ _Size ], double _Value, int _FractionalDigitCount, int * _PtDec, int * _PtSign ) throw ( ) { return _fcvt_s ( _Buffer, _Size, _Value, _FractionalDigitCount, _PtDec, _PtSign ); }}
#line 794 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
char *__cdecl _fcvt(double _Value, int _FractionalDigitCount, int * _PtDec, int * _PtSign); 
#line 802
errno_t __cdecl _gcvt_s(char * _Buffer, size_t _BufferCount, double _Value, int _DigitCount); 
#line 809
extern "C++" {template < size_t _Size > inline errno_t __cdecl _gcvt_s ( char ( & _Buffer ) [ _Size ], double _Value, int _DigitCount ) throw ( ) { return _gcvt_s ( _Buffer, _Size, _Value, _DigitCount ); }}
#line 818 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
char *__cdecl _gcvt(double _Value, int _DigitCount, char * _Buffer); 
#line 846 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
int __cdecl ___mb_cur_max_func(); 
#line 849
int __cdecl ___mb_cur_max_l_func(_locale_t _Locale); 
#line 855 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
int __cdecl mblen(const char * _Ch, size_t _MaxCount); 
#line 861
int __cdecl _mblen_l(const char * _Ch, size_t _MaxCount, _locale_t _Locale); 
#line 869
size_t __cdecl _mbstrlen(const char * _String); 
#line 875
size_t __cdecl _mbstrlen_l(const char * _String, _locale_t _Locale); 
#line 882
size_t __cdecl _mbstrnlen(const char * _String, size_t _MaxCount); 
#line 889
size_t __cdecl _mbstrnlen_l(const char * _String, size_t _MaxCount, _locale_t _Locale); 
#line 896
int __cdecl mbtowc(__wchar_t * _DstCh, const char * _SrcCh, size_t _SrcSizeInBytes); 
#line 903
int __cdecl _mbtowc_l(__wchar_t * _DstCh, const char * _SrcCh, size_t _SrcSizeInBytes, _locale_t _Locale); 
#line 911
errno_t __cdecl mbstowcs_s(size_t * _PtNumOfCharConverted, __wchar_t * _DstBuf, size_t _SizeInWords, const char * _SrcBuf, size_t _MaxCount); 
#line 919
extern "C++" {template < size_t _Size > inline errno_t __cdecl mbstowcs_s ( size_t * _PtNumOfCharConverted, wchar_t ( & _Dest ) [ _Size ], char const * _Source, size_t _MaxCount ) throw ( ) { return mbstowcs_s ( _PtNumOfCharConverted, _Dest, _Size, _Source, _MaxCount ); }}
#line 928 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
size_t __cdecl mbstowcs(__wchar_t * _Dest, const char * _Source, size_t _MaxCount); 
#line 936 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
errno_t __cdecl _mbstowcs_s_l(size_t * _PtNumOfCharConverted, __wchar_t * _DstBuf, size_t _SizeInWords, const char * _SrcBuf, size_t _MaxCount, _locale_t _Locale); 
#line 945
extern "C++" {template < size_t _Size > inline errno_t __cdecl _mbstowcs_s_l ( size_t * _PtNumOfCharConverted, wchar_t ( & _Dest ) [ _Size ], char const * _Source, size_t _MaxCount, _locale_t _Locale ) throw ( ) { return _mbstowcs_s_l ( _PtNumOfCharConverted, _Dest, _Size, _Source, _MaxCount, _Locale ); }}
#line 954 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
size_t __cdecl _mbstowcs_l(__wchar_t * _Dest, const char * _Source, size_t _MaxCount, _locale_t _Locale); 
#line 967 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
int __cdecl wctomb(char * _MbCh, __wchar_t _WCh); 
#line 973
int __cdecl _wctomb_l(char * _MbCh, __wchar_t _WCh, _locale_t _Locale); 
#line 982
errno_t __cdecl wctomb_s(int * _SizeConverted, char * _MbCh, rsize_t _SizeInBytes, __wchar_t _WCh); 
#line 991 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
errno_t __cdecl _wctomb_s_l(int * _SizeConverted, char * _MbCh, size_t _SizeInBytes, __wchar_t _WCh, _locale_t _Locale); 
#line 999
errno_t __cdecl wcstombs_s(size_t * _PtNumOfCharConverted, char * _Dst, size_t _DstSizeInBytes, const __wchar_t * _Src, size_t _MaxCountInBytes); 
#line 1007
extern "C++" {template < size_t _Size > inline errno_t __cdecl wcstombs_s ( size_t * _PtNumOfCharConverted, char ( & _Dest ) [ _Size ], wchar_t const * _Source, size_t _MaxCount ) throw ( ) { return wcstombs_s ( _PtNumOfCharConverted, _Dest, _Size, _Source, _MaxCount ); }}
#line 1016 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
size_t __cdecl wcstombs(char * _Dest, const __wchar_t * _Source, size_t _MaxCount); 
#line 1024 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
errno_t __cdecl _wcstombs_s_l(size_t * _PtNumOfCharConverted, char * _Dst, size_t _DstSizeInBytes, const __wchar_t * _Src, size_t _MaxCountInBytes, _locale_t _Locale); 
#line 1033
extern "C++" {template < size_t _Size > inline errno_t __cdecl _wcstombs_s_l ( size_t * _PtNumOfCharConverted, char ( & _Dest ) [ _Size ], wchar_t const * _Source, size_t _MaxCount, _locale_t _Locale ) throw ( ) { return _wcstombs_s_l ( _PtNumOfCharConverted, _Dest, _Size, _Source, _MaxCount, _Locale ); }}
#line 1042 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
size_t __cdecl _wcstombs_l(char * _Dest, const __wchar_t * _Source, size_t _MaxCount, _locale_t _Locale); 
#line 1072 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
__declspec(allocator) char *__cdecl _fullpath(char * _Buffer, const char * _Path, size_t _BufferCount); 
#line 1081
errno_t __cdecl _makepath_s(char * _Buffer, size_t _BufferCount, const char * _Drive, const char * _Dir, const char * _Filename, const char * _Ext); 
#line 1090
extern "C++" {template < size_t _Size > inline errno_t __cdecl _makepath_s ( char ( & _Buffer ) [ _Size ], char const * _Drive, char const * _Dir, char const * _Filename, char const * _Ext ) throw ( ) { return _makepath_s ( _Buffer, _Size, _Drive, _Dir, _Filename, _Ext ); }}
#line 1099 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
#pragma warning(push)
#pragma warning(disable: 28719)
#pragma warning(disable: 28726)
void __cdecl _makepath(char * _Buffer, const char * _Drive, const char * _Dir, const char * _Filename, const char * _Ext); 
#line 1110 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
#pragma warning(pop)
#line 1113
void __cdecl _splitpath(const char * _FullPath, char * _Drive, char * _Dir, char * _Filename, char * _Ext); 
#line 1122
errno_t __cdecl _splitpath_s(const char * _FullPath, char * _Drive, size_t _DriveCount, char * _Dir, size_t _DirCount, char * _Filename, size_t _FilenameCount, char * _Ext, size_t _ExtCount); 
#line 1134
extern "C++" {template < size_t _DriveSize, size_t _DirSize, size_t _NameSize, size_t _ExtSize > inline errno_t __cdecl _splitpath_s ( char const * _Dest, char ( & _Drive ) [ _DriveSize ], char ( & _Dir ) [ _DirSize ], char ( & _Name ) [ _NameSize ], char ( & _Ext ) [ _ExtSize ] ) throw ( ) { return _splitpath_s ( _Dest, _Drive, _DriveSize, _Dir, _DirSize, _Name, _NameSize, _Ext, _ExtSize ); }}
#line 1140
errno_t __cdecl getenv_s(size_t * _RequiredCount, char * _Buffer, rsize_t _BufferCount, const char * _VarName); 
#line 1151 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
int *__cdecl __p___argc(); 
#line 1152
char ***__cdecl __p___argv(); 
#line 1153
__wchar_t ***__cdecl __p___wargv(); 
#line 1165 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
char ***__cdecl __p__environ(); 
#line 1166
__wchar_t ***__cdecl __p__wenviron(); 
#line 1191 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
char *__cdecl getenv(const char * _VarName); 
#line 1195
extern "C++" {template < size_t _Size > inline errno_t __cdecl getenv_s ( size_t * _RequiredCount, char ( & _Buffer ) [ _Size ], char const * _VarName ) throw ( ) { return getenv_s ( _RequiredCount, _Buffer, _Size, _VarName ); }}
#line 1208 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
errno_t __cdecl _dupenv_s(char ** _Buffer, size_t * _BufferCount, const char * _VarName); 
#line 1218 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
int __cdecl system(const char * _Command); 
#line 1224
#pragma warning (push)
#pragma warning (disable:6540)
#line 1228
int __cdecl _putenv(const char * _EnvString); 
#line 1233
errno_t __cdecl _putenv_s(const char * _Name, const char * _Value); 
#line 1238
#pragma warning (pop)
#line 1240
errno_t __cdecl _searchenv_s(const char * _Filename, const char * _VarName, char * _Buffer, size_t _BufferCount); 
#line 1247
extern "C++" {template < size_t _Size > inline errno_t __cdecl _searchenv_s ( char const * _Filename, char const * _VarName, char ( & _Buffer ) [ _Size ] ) throw ( ) { return _searchenv_s ( _Filename, _VarName, _Buffer, _Size ); }}
#line 1255 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
void __cdecl _searchenv(const char * _Filename, const char * _VarName, char * _Buffer); 
#line 1263 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
__declspec(deprecated("This function or variable has been superceded by newer library or operating system functionality. Consider using SetErrorMode in" "stead. See online help for details.")) void __cdecl 
#line 1264
_seterrormode(int _Mode); 
#line 1268
__declspec(deprecated("This function or variable has been superceded by newer library or operating system functionality. Consider using Beep instead. S" "ee online help for details.")) void __cdecl 
#line 1269
_beep(unsigned _Frequency, unsigned _Duration); 
#line 1274
__declspec(deprecated("This function or variable has been superceded by newer library or operating system functionality. Consider using Sleep instead. " "See online help for details.")) void __cdecl 
#line 1275
_sleep(unsigned long _Duration); 
#line 1296 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
#pragma warning(push)
#pragma warning(disable: 4141)
#line 1300
char *__cdecl ecvt(double _Value, int _DigitCount, int * _PtDec, int * _PtSign); 
#line 1308
char *__cdecl fcvt(double _Value, int _FractionalDigitCount, int * _PtDec, int * _PtSign); 
#line 1316
char *__cdecl gcvt(double _Value, int _DigitCount, char * _DstBuf); 
#line 1323
char *__cdecl itoa(int _Value, char * _Buffer, int _Radix); 
#line 1330
char *__cdecl ltoa(long _Value, char * _Buffer, int _Radix); 
#line 1338
void __cdecl swab(char * _Buf1, char * _Buf2, int _SizeInBytes); 
#line 1345
char *__cdecl ultoa(unsigned long _Value, char * _Buffer, int _Radix); 
#line 1354
int __cdecl putenv(const char * _EnvString); 
#line 1358
#pragma warning(pop)
#line 1360
_onexit_t __cdecl onexit(_onexit_t _Func); 
#line 1366 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\stdlib.h"
}__pragma( pack ( pop )) 
#line 13 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\vcruntime_new.h"
extern "C++" {
#line 15
#pragma pack ( push, 8 )
#line 17
#pragma warning(push)
#pragma warning(disable: 4985)
#line 32
namespace std { 
#line 34
struct nothrow_t { 
#line 36
explicit nothrow_t() = default;
#line 38 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\vcruntime_new.h"
}; 
#line 43
extern const nothrow_t nothrow; 
#line 45 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\vcruntime_new.h"
}
#line 49 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\vcruntime_new.h"
__declspec(allocator) void *__cdecl operator new(size_t _Size); 
#line 54
__declspec(allocator) void *__cdecl operator new(size_t _Size, const std::nothrow_t &) noexcept; 
#line 60
__declspec(allocator) void *__cdecl operator new[](size_t _Size); 
#line 65
__declspec(allocator) void *__cdecl operator new[](size_t _Size, const std::nothrow_t &) noexcept; 
#line 70
void __cdecl operator delete(void * _Block) noexcept; 
#line 74
void __cdecl operator delete(void * _Block, const std::nothrow_t &) noexcept; 
#line 79
void __cdecl operator delete[](void * _Block) noexcept; 
#line 83
void __cdecl operator delete[](void * _Block, const std::nothrow_t &) noexcept; 
#line 88
void __cdecl operator delete(void * _Block, size_t _Size) noexcept; 
#line 93
void __cdecl operator delete[](void * _Block, size_t _Size) noexcept; 
#line 161
#pragma warning(push)
#pragma warning(disable: 4577)
#pragma warning(disable: 4514)
#line 167
inline void *__cdecl operator new(size_t _Size, void *_Where) noexcept 
#line 168
{ 
#line 169
(void)_Size; 
#line 170
return _Where; 
#line 171
} 
#line 173
inline void __cdecl operator delete(void *, void *) noexcept 
#line 174
{ 
#line 176
} 
#line 182 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\vcruntime_new.h"
inline void *__cdecl operator new[](size_t _Size, void *
#line 183
_Where) noexcept 
#line 184
{ 
#line 185
(void)_Size; 
#line 186
return _Where; 
#line 187
} 
#line 189
inline void __cdecl operator delete[](void *, void *) noexcept 
#line 190
{ 
#line 191
} 
#line 200 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\vcruntime_new.h"
}
#line 193
#pragma warning(pop)
#line 197
#pragma warning(pop)
#pragma pack ( pop )
#line 13 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\vcruntime_new_debug.h"
extern "C++" {
#line 15
#pragma pack ( push, 8 )
#line 23
__declspec(allocator) void *__cdecl operator new(size_t _Size, int _BlockUse, const char * _FileName, int _LineNumber); 
#line 31
__declspec(allocator) void *__cdecl operator new[](size_t _Size, int _BlockUse, const char * _FileName, int _LineNumber); 
#line 38
void __cdecl operator delete(void * _Block, int _BlockUse, const char * _FileName, int _LineNumber) noexcept; 
#line 45
void __cdecl operator delete[](void * _Block, int _BlockUse, const char * _FileName, int _LineNumber) noexcept; 
#line 58 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\vcruntime_new_debug.h"
}
#line 56
#pragma pack ( pop )
#line 15 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\crtdbg.h"
__pragma( pack ( push, 8 )) extern "C" {
#line 19
typedef void *_HFILE; 
#line 45
typedef int (__cdecl *_CRT_REPORT_HOOK)(int, char *, int *); 
#line 46
typedef int (__cdecl *_CRT_REPORT_HOOKW)(int, __wchar_t *, int *); 
#line 52
typedef int (__cdecl *_CRT_ALLOC_HOOK)(int, void *, size_t, int, long, const unsigned char *, int); 
#line 108
typedef void (__cdecl *_CRT_DUMP_CLIENT)(void *, size_t); 
#line 114
struct _CrtMemBlockHeader; 
#line 123
typedef 
#line 116
struct _CrtMemState { 
#line 118
_CrtMemBlockHeader *pBlockHeader; 
#line 119
size_t lCounts[5]; 
#line 120
size_t lSizes[5]; 
#line 121
size_t lHighWaterCount; 
#line 122
size_t lTotalCount; 
#line 123
} _CrtMemState; 
#line 809 "C:\\Program Files (x86)\\Windows Kits\\10\\include\\10.0.18362.0\\ucrt\\crtdbg.h"
}__pragma( pack ( pop )) 
#line 10 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\yvals.h"
#pragma pack ( push, 8 )
#pragma warning(push, 3)
#pragma warning(disable : 4455 4494 4583 4619 4643 4702 4793 4988 )
#line 130 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\yvals.h"
#pragma detect_mismatch("_MSC_VER", "1900")
#line 134 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\yvals.h"
#pragma detect_mismatch("_ITERATOR_DEBUG_LEVEL", "0")
#line 139 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\yvals.h"
#pragma detect_mismatch("RuntimeLibrary", "MT_StaticRelease")
#line 57 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\use_ansi.h"
#pragma comment(lib, "libcpmt")
#line 336 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\yvals.h"
namespace std { 
#line 337
enum _Uninitialized { 
#line 338
_Noinit
#line 339
}; 
#line 342
class _Lockit { 
#line 358 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\yvals.h"
public: __thiscall _Lockit(); 
#line 359
explicit __thiscall _Lockit(int); 
#line 360
__thiscall ~_Lockit() noexcept; 
#line 363 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\yvals.h"
static void __cdecl _Lockit_ctor(int); 
#line 364
static void __cdecl _Lockit_dtor(int); 
#line 367
private: static void __cdecl _Lockit_ctor(_Lockit *); 
#line 368
static void __cdecl _Lockit_ctor(_Lockit *, int); 
#line 369
static void __cdecl _Lockit_dtor(_Lockit *); 
#line 372
public: _Lockit(const _Lockit &) = delete;
#line 373
_Lockit &operator=(const _Lockit &) = delete;
#line 376
private: int _Locktype; 
#line 377
}; 
#line 452 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\yvals.h"
class _Init_locks { 
#line 464 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\yvals.h"
public: __thiscall _Init_locks(); 
#line 465
__thiscall ~_Init_locks() noexcept; 
#line 469 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\yvals.h"
private: static void __cdecl _Init_locks_ctor(_Init_locks *); 
#line 470
static void __cdecl _Init_locks_dtor(_Init_locks *); 
#line 471
}; 
#line 520 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\yvals.h"
}
#line 529 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\yvals.h"
#pragma warning(pop)
#pragma pack ( pop )
#line 11 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\cstdlib"
#pragma pack ( push, 8 )
#pragma warning(push, 3)
#pragma warning(disable : 4455 4494 4583 4619 4643 4702 4793 4988 )
#line 19
inline double abs(double _Xx) noexcept { 
#line 20
return ::fabs(_Xx); 
#line 21
} 
#line 23
inline float abs(float _Xx) noexcept { 
#line 24
return ::fabsf(_Xx); 
#line 25
} 
#line 27
inline long double abs(long double _Xx) noexcept { 
#line 28
return ::fabsl(_Xx); 
#line 29
} 
#line 31
namespace std { 
#line 32
using ::size_t;
#line 33
using ::div_t;
#line 34
using ::ldiv_t;
#line 35
using ::abort;
#line 36
using ::abs;
#line 37
using ::atexit;
#line 38
using ::atof;
#line 39
using ::atoi;
#line 40
using ::atol;
#line 41
using ::bsearch;
#line 42
using ::calloc;
#line 43
using ::div;
#line 44
using ::exit;
#line 45
using ::free;
#line 46
using ::labs;
#line 47
using ::ldiv;
#line 48
using ::malloc;
#line 49
using ::mblen;
#line 50
using ::mbstowcs;
#line 51
using ::mbtowc;
#line 52
using ::qsort;
#line 53
using ::rand;
#line 54
using ::realloc;
#line 55
using ::srand;
#line 56
using ::strtod;
#line 57
using ::strtol;
#line 58
using ::strtoul;
#line 59
using ::wcstombs;
#line 60
using ::wctomb;
#line 62
using ::lldiv_t;
#line 65
using ::getenv;
#line 66
using ::system;
#line 69 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\cstdlib"
using ::atoll;
#line 70
using ::llabs;
#line 71
using ::lldiv;
#line 72
using ::strtof;
#line 73
using ::strtold;
#line 74
using ::strtoll;
#line 75
using ::strtoull;
#line 77
using ::_Exit;
#line 78
using ::at_quick_exit;
#line 79
using ::quick_exit;
#line 80
}
#line 84
#pragma warning(pop)
#pragma pack ( pop )
#line 11 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\cmath"
#pragma pack ( push, 8 )
#pragma warning(push, 3)
#pragma warning(disable : 4455 4494 4583 4619 4643 4702 4793 4988 )
#line 18
inline double pow(double _Xx, int _Yx) noexcept { 
#line 19
if (_Yx == 2) { 
#line 20
return _Xx * _Xx; 
#line 21
}  
#line 23
return ::pow(_Xx, static_cast< double>(_Yx)); 
#line 24
} 
#line 26
inline float acos(float _Xx) noexcept { 
#line 27
return ::acosf(_Xx); 
#line 28
} 
#line 30
inline float acosh(float _Xx) noexcept { 
#line 31
return ::acoshf(_Xx); 
#line 32
} 
#line 34
inline float asin(float _Xx) noexcept { 
#line 35
return ::asinf(_Xx); 
#line 36
} 
#line 38
inline float asinh(float _Xx) noexcept { 
#line 39
return ::asinhf(_Xx); 
#line 40
} 
#line 42
inline float atan(float _Xx) noexcept { 
#line 43
return ::atanf(_Xx); 
#line 44
} 
#line 46
inline float atanh(float _Xx) noexcept { 
#line 47
return ::atanhf(_Xx); 
#line 48
} 
#line 50
inline float atan2(float _Yx, float _Xx) noexcept { 
#line 51
return ::atan2f(_Yx, _Xx); 
#line 52
} 
#line 54
inline float cbrt(float _Xx) noexcept { 
#line 55
return ::cbrtf(_Xx); 
#line 56
} 
#line 58
inline float ceil(float _Xx) noexcept { 
#line 59
return ::ceilf(_Xx); 
#line 60
} 
#line 62
inline float copysign(float _Number, float _Sign) noexcept { 
#line 63
return ::copysignf(_Number, _Sign); 
#line 64
} 
#line 66
inline float cos(float _Xx) noexcept { 
#line 67
return ::cosf(_Xx); 
#line 68
} 
#line 70
inline float cosh(float _Xx) noexcept { 
#line 71
return ::coshf(_Xx); 
#line 72
} 
#line 74
inline float erf(float _Xx) noexcept { 
#line 75
return ::erff(_Xx); 
#line 76
} 
#line 78
inline float erfc(float _Xx) noexcept { 
#line 79
return ::erfcf(_Xx); 
#line 80
} 
#line 82
inline float exp(float _Xx) noexcept { 
#line 83
return ::expf(_Xx); 
#line 84
} 
#line 86
inline float exp2(float _Xx) noexcept { 
#line 87
return ::exp2f(_Xx); 
#line 88
} 
#line 90
inline float expm1(float _Xx) noexcept { 
#line 91
return ::expm1f(_Xx); 
#line 92
} 
#line 94
inline float fabs(float _Xx) noexcept { 
#line 95
return ::fabsf(_Xx); 
#line 96
} 
#line 98
inline float fdim(float _Xx, float _Yx) noexcept { 
#line 99
return ::fdimf(_Xx, _Yx); 
#line 100
} 
#line 102
inline float floor(float _Xx) noexcept { 
#line 103
return ::floorf(_Xx); 
#line 104
} 
#line 106
inline float fma(float _Xx, float _Yx, float _Zx) noexcept { 
#line 107
return ::fmaf(_Xx, _Yx, _Zx); 
#line 108
} 
#line 110
inline float fmax(float _Xx, float _Yx) noexcept { 
#line 111
return ::fmaxf(_Xx, _Yx); 
#line 112
} 
#line 114
inline float fmin(float _Xx, float _Yx) noexcept { 
#line 115
return ::fminf(_Xx, _Yx); 
#line 116
} 
#line 118
inline float fmod(float _Xx, float _Yx) noexcept { 
#line 119
return ::fmodf(_Xx, _Yx); 
#line 120
} 
#line 122
inline float frexp(float _Xx, int *_Yx) noexcept { 
#line 123
return ::frexpf(_Xx, _Yx); 
#line 124
} 
#line 126
inline float hypot(float _Xx, float _Yx) noexcept { 
#line 127
return ::hypotf(_Xx, _Yx); 
#line 128
} 
#line 130
inline int ilogb(float _Xx) noexcept { 
#line 131
return ::ilogbf(_Xx); 
#line 132
} 
#line 134
inline float ldexp(float _Xx, int _Yx) noexcept { 
#line 135
return ::ldexpf(_Xx, _Yx); 
#line 136
} 
#line 138
inline float lgamma(float _Xx) noexcept { 
#line 139
return ::lgammaf(_Xx); 
#line 140
} 
#line 142
inline __int64 llrint(float _Xx) noexcept { 
#line 143
return ::llrintf(_Xx); 
#line 144
} 
#line 146
inline __int64 llround(float _Xx) noexcept { 
#line 147
return ::llroundf(_Xx); 
#line 148
} 
#line 150
inline float log(float _Xx) noexcept { 
#line 151
return ::logf(_Xx); 
#line 152
} 
#line 154
inline float log10(float _Xx) noexcept { 
#line 155
return ::log10f(_Xx); 
#line 156
} 
#line 158
inline float log1p(float _Xx) noexcept { 
#line 159
return ::log1pf(_Xx); 
#line 160
} 
#line 162
inline float log2(float _Xx) noexcept { 
#line 163
return ::log2f(_Xx); 
#line 164
} 
#line 166
inline float logb(float _Xx) noexcept { 
#line 167
return ::logbf(_Xx); 
#line 168
} 
#line 170
inline long lrint(float _Xx) noexcept { 
#line 171
return ::lrintf(_Xx); 
#line 172
} 
#line 174
inline long lround(float _Xx) noexcept { 
#line 175
return ::lroundf(_Xx); 
#line 176
} 
#line 178
inline float modf(float _Xx, float *_Yx) noexcept { 
#line 179
return ::modff(_Xx, _Yx); 
#line 180
} 
#line 182
inline float nearbyint(float _Xx) noexcept { 
#line 183
return ::nearbyintf(_Xx); 
#line 184
} 
#line 186
inline float nextafter(float _Xx, float _Yx) noexcept { 
#line 187
return ::nextafterf(_Xx, _Yx); 
#line 188
} 
#line 190
inline float nexttoward(float _Xx, long double _Yx) noexcept { 
#line 191
return ::nexttowardf(_Xx, _Yx); 
#line 192
} 
#line 194
inline float pow(float _Xx, float _Yx) noexcept { 
#line 195
return ::powf(_Xx, _Yx); 
#line 196
} 
#line 198
inline float pow(float _Xx, int _Yx) noexcept { 
#line 199
if (_Yx == 2) { 
#line 200
return _Xx * _Xx; 
#line 201
}  
#line 203
return ::powf(_Xx, static_cast< float>(_Yx)); 
#line 204
} 
#line 206
inline float remainder(float _Xx, float _Yx) noexcept { 
#line 207
return ::remainderf(_Xx, _Yx); 
#line 208
} 
#line 210
inline float remquo(float _Xx, float _Yx, int *_Zx) noexcept { 
#line 211
return ::remquof(_Xx, _Yx, _Zx); 
#line 212
} 
#line 214
inline float rint(float _Xx) noexcept { 
#line 215
return ::rintf(_Xx); 
#line 216
} 
#line 218
inline float round(float _Xx) noexcept { 
#line 219
return ::roundf(_Xx); 
#line 220
} 
#line 222
inline float scalbln(float _Xx, long _Yx) noexcept { 
#line 223
return ::scalblnf(_Xx, _Yx); 
#line 224
} 
#line 226
inline float scalbn(float _Xx, int _Yx) noexcept { 
#line 227
return ::scalbnf(_Xx, _Yx); 
#line 228
} 
#line 230
inline float sin(float _Xx) noexcept { 
#line 231
return ::sinf(_Xx); 
#line 232
} 
#line 234
inline float sinh(float _Xx) noexcept { 
#line 235
return ::sinhf(_Xx); 
#line 236
} 
#line 238
inline float sqrt(float _Xx) noexcept { 
#line 239
return ::sqrtf(_Xx); 
#line 240
} 
#line 242
inline float tan(float _Xx) noexcept { 
#line 243
return ::tanf(_Xx); 
#line 244
} 
#line 246
inline float tanh(float _Xx) noexcept { 
#line 247
return ::tanhf(_Xx); 
#line 248
} 
#line 250
inline float tgamma(float _Xx) noexcept { 
#line 251
return ::tgammaf(_Xx); 
#line 252
} 
#line 254
inline float trunc(float _Xx) noexcept { 
#line 255
return ::truncf(_Xx); 
#line 256
} 
#line 258
inline long double acos(long double _Xx) noexcept { 
#line 259
return ::acosl(_Xx); 
#line 260
} 
#line 262
inline long double acosh(long double _Xx) noexcept { 
#line 263
return ::acoshl(_Xx); 
#line 264
} 
#line 266
inline long double asin(long double _Xx) noexcept { 
#line 267
return ::asinl(_Xx); 
#line 268
} 
#line 270
inline long double asinh(long double _Xx) noexcept { 
#line 271
return ::asinhl(_Xx); 
#line 272
} 
#line 274
inline long double atan(long double _Xx) noexcept { 
#line 275
return ::atanl(_Xx); 
#line 276
} 
#line 278
inline long double atanh(long double _Xx) noexcept { 
#line 279
return ::atanhl(_Xx); 
#line 280
} 
#line 282
inline long double atan2(long double _Yx, long double _Xx) noexcept { 
#line 283
return ::atan2l(_Yx, _Xx); 
#line 284
} 
#line 286
inline long double cbrt(long double _Xx) noexcept { 
#line 287
return ::cbrtl(_Xx); 
#line 288
} 
#line 290
inline long double ceil(long double _Xx) noexcept { 
#line 291
return ::ceill(_Xx); 
#line 292
} 
#line 294
inline long double copysign(long double _Number, long double _Sign) noexcept { 
#line 295
return ::copysignl(_Number, _Sign); 
#line 296
} 
#line 298
inline long double cos(long double _Xx) noexcept { 
#line 299
return ::cosl(_Xx); 
#line 300
} 
#line 302
inline long double cosh(long double _Xx) noexcept { 
#line 303
return ::coshl(_Xx); 
#line 304
} 
#line 306
inline long double erf(long double _Xx) noexcept { 
#line 307
return ::erfl(_Xx); 
#line 308
} 
#line 310
inline long double erfc(long double _Xx) noexcept { 
#line 311
return ::erfcl(_Xx); 
#line 312
} 
#line 314
inline long double exp(long double _Xx) noexcept { 
#line 315
return ::expl(_Xx); 
#line 316
} 
#line 318
inline long double exp2(long double _Xx) noexcept { 
#line 319
return ::exp2l(_Xx); 
#line 320
} 
#line 322
inline long double expm1(long double _Xx) noexcept { 
#line 323
return ::expm1l(_Xx); 
#line 324
} 
#line 326
inline long double fabs(long double _Xx) noexcept { 
#line 327
return ::fabsl(_Xx); 
#line 328
} 
#line 330
inline long double fdim(long double _Xx, long double _Yx) noexcept { 
#line 331
return ::fdiml(_Xx, _Yx); 
#line 332
} 
#line 334
inline long double floor(long double _Xx) noexcept { 
#line 335
return ::floorl(_Xx); 
#line 336
} 
#line 338
inline long double fma(long double _Xx, long double _Yx, long double _Zx) noexcept { 
#line 339
return ::fmal(_Xx, _Yx, _Zx); 
#line 340
} 
#line 342
inline long double fmax(long double _Xx, long double _Yx) noexcept { 
#line 343
return ::fmaxl(_Xx, _Yx); 
#line 344
} 
#line 346
inline long double fmin(long double _Xx, long double _Yx) noexcept { 
#line 347
return ::fminl(_Xx, _Yx); 
#line 348
} 
#line 350
inline long double fmod(long double _Xx, long double _Yx) noexcept { 
#line 351
return ::fmodl(_Xx, _Yx); 
#line 352
} 
#line 354
inline long double frexp(long double _Xx, int *_Yx) noexcept { 
#line 355
return ::frexpl(_Xx, _Yx); 
#line 356
} 
#line 358
inline long double hypot(long double _Xx, long double _Yx) noexcept { 
#line 359
return ::hypotl(_Xx, _Yx); 
#line 360
} 
#line 362
inline int ilogb(long double _Xx) noexcept { 
#line 363
return ::ilogbl(_Xx); 
#line 364
} 
#line 366
inline long double ldexp(long double _Xx, int _Yx) noexcept { 
#line 367
return ::ldexpl(_Xx, _Yx); 
#line 368
} 
#line 370
inline long double lgamma(long double _Xx) noexcept { 
#line 371
return ::lgammal(_Xx); 
#line 372
} 
#line 374
inline __int64 llrint(long double _Xx) noexcept { 
#line 375
return ::llrintl(_Xx); 
#line 376
} 
#line 378
inline __int64 llround(long double _Xx) noexcept { 
#line 379
return ::llroundl(_Xx); 
#line 380
} 
#line 382
inline long double log(long double _Xx) noexcept { 
#line 383
return ::logl(_Xx); 
#line 384
} 
#line 386
inline long double log10(long double _Xx) noexcept { 
#line 387
return ::log10l(_Xx); 
#line 388
} 
#line 390
inline long double log1p(long double _Xx) noexcept { 
#line 391
return ::log1pl(_Xx); 
#line 392
} 
#line 394
inline long double log2(long double _Xx) noexcept { 
#line 395
return ::log2l(_Xx); 
#line 396
} 
#line 398
inline long double logb(long double _Xx) noexcept { 
#line 399
return ::logbl(_Xx); 
#line 400
} 
#line 402
inline long lrint(long double _Xx) noexcept { 
#line 403
return ::lrintl(_Xx); 
#line 404
} 
#line 406
inline long lround(long double _Xx) noexcept { 
#line 407
return ::lroundl(_Xx); 
#line 408
} 
#line 410
inline long double modf(long double _Xx, long double *_Yx) noexcept { 
#line 411
return ::modfl(_Xx, _Yx); 
#line 412
} 
#line 414
inline long double nearbyint(long double _Xx) noexcept { 
#line 415
return ::nearbyintl(_Xx); 
#line 416
} 
#line 418
inline long double nextafter(long double _Xx, long double _Yx) noexcept { 
#line 419
return ::nextafterl(_Xx, _Yx); 
#line 420
} 
#line 422
inline long double nexttoward(long double _Xx, long double _Yx) noexcept { 
#line 423
return ::nexttowardl(_Xx, _Yx); 
#line 424
} 
#line 426
inline long double pow(long double _Xx, long double _Yx) noexcept { 
#line 427
return ::powl(_Xx, _Yx); 
#line 428
} 
#line 430
inline long double pow(long double _Xx, int _Yx) noexcept { 
#line 431
if (_Yx == 2) { 
#line 432
return _Xx * _Xx; 
#line 433
}  
#line 435
return ::powl(_Xx, static_cast< long double>(_Yx)); 
#line 436
} 
#line 438
inline long double remainder(long double _Xx, long double _Yx) noexcept { 
#line 439
return ::remainderl(_Xx, _Yx); 
#line 440
} 
#line 442
inline long double remquo(long double _Xx, long double _Yx, int *_Zx) noexcept { 
#line 443
return ::remquol(_Xx, _Yx, _Zx); 
#line 444
} 
#line 446
inline long double rint(long double _Xx) noexcept { 
#line 447
return ::rintl(_Xx); 
#line 448
} 
#line 450
inline long double round(long double _Xx) noexcept { 
#line 451
return ::roundl(_Xx); 
#line 452
} 
#line 454
inline long double scalbln(long double _Xx, long _Yx) noexcept { 
#line 455
return ::scalblnl(_Xx, _Yx); 
#line 456
} 
#line 458
inline long double scalbn(long double _Xx, int _Yx) noexcept { 
#line 459
return ::scalbnl(_Xx, _Yx); 
#line 460
} 
#line 462
inline long double sin(long double _Xx) noexcept { 
#line 463
return ::sinl(_Xx); 
#line 464
} 
#line 466
inline long double sinh(long double _Xx) noexcept { 
#line 467
return ::sinhl(_Xx); 
#line 468
} 
#line 470
inline long double sqrt(long double _Xx) noexcept { 
#line 471
return ::sqrtl(_Xx); 
#line 472
} 
#line 474
inline long double tan(long double _Xx) noexcept { 
#line 475
return ::tanl(_Xx); 
#line 476
} 
#line 478
inline long double tanh(long double _Xx) noexcept { 
#line 479
return ::tanhl(_Xx); 
#line 480
} 
#line 482
inline long double tgamma(long double _Xx) noexcept { 
#line 483
return ::tgammal(_Xx); 
#line 484
} 
#line 486
inline long double trunc(long double _Xx) noexcept { 
#line 487
return ::truncl(_Xx); 
#line 488
} 
#line 8 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\xtr1common"
#pragma pack ( push, 8 )
#pragma warning(push, 3)
#pragma warning(disable : 4455 4494 4583 4619 4643 4702 4793 4988 )
#line 15
namespace std { 
#line 17
template < class _Ty, _Ty _Val >
struct integral_constant {
    static constexpr _Ty value = _Val;

    using value_type = _Ty;
    using type = integral_constant;

    constexpr operator value_type ( ) const noexcept {
        return value;
    }

     constexpr value_type operator ( ) ( ) const noexcept {
        return value;
    }
};
#line 34
template< bool _Val> using bool_constant = integral_constant< bool, _Val> ; 
#line 37
using true_type = bool_constant< true> ; 
#line 38
using false_type = bool_constant< false> ; 
#line 41
template< bool _Test, class _Ty = void> 
#line 42
struct enable_if { }; 
#line 44
template< class _Ty> 
#line 45
struct enable_if< true, _Ty>  { 
#line 46
using type = _Ty; 
#line 47
}; 
#line 49
template< bool _Test, class _Ty = void> using enable_if_t = typename enable_if< _Test, _Ty> ::type; 
#line 53
template< bool _Test, class _Ty1, class _Ty2> 
#line 54
struct conditional { 
#line 55
using type = _Ty1; 
#line 56
}; 
#line 58
template< class _Ty1, class _Ty2> 
#line 59
struct conditional< false, _Ty1, _Ty2>  { 
#line 60
using type = _Ty2; 
#line 61
}; 
#line 63
template< bool _Test, class _Ty1, class _Ty2> using conditional_t = typename conditional< _Test, _Ty1, _Ty2> ::type; 
#line 76
template< class , class > constexpr bool 
#line 77
is_same_v = false; 
#line 78
template< class _Ty> constexpr bool 
#line 79
is_same_v< _Ty, _Ty>  = true; 
#line 81
template< class _Ty1, class _Ty2> 
#line 82
struct is_same : public bool_constant< is_same_v< _Ty1, _Ty2> >  { }; 
#line 86 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\xtr1common"
template< class _Ty> 
#line 87
struct remove_const { 
#line 88
using type = _Ty; 
#line 89
}; 
#line 91
template< class _Ty> 
#line 92
struct remove_const< const _Ty>  { 
#line 93
using type = _Ty; 
#line 94
}; 
#line 96
template< class _Ty> using remove_const_t = typename remove_const< _Ty> ::type; 
#line 100
template< class _Ty> 
#line 101
struct remove_volatile { 
#line 102
using type = _Ty; 
#line 103
}; 
#line 105
template< class _Ty> 
#line 106
struct remove_volatile< volatile _Ty>  { 
#line 107
using type = _Ty; 
#line 108
}; 
#line 110
template< class _Ty> using remove_volatile_t = typename remove_volatile< _Ty> ::type; 
#line 114
template< class _Ty> 
#line 115
struct remove_cv { 
#line 116
using type = _Ty; 
#line 118
template< template< class >  class _Fn> using _Apply = _Fn< _Ty> ; 
#line 120
}; 
#line 122
template< class _Ty> 
#line 123
struct remove_cv< const _Ty>  { 
#line 124
using type = _Ty; 
#line 126
template< template< class >  class _Fn> using _Apply = const _Fn< _Ty> ; 
#line 128
}; 
#line 130
template< class _Ty> 
#line 131
struct remove_cv< volatile _Ty>  { 
#line 132
using type = _Ty; 
#line 134
template< template< class >  class _Fn> using _Apply = volatile _Fn< _Ty> ; 
#line 136
}; 
#line 138
template< class _Ty> 
#line 139
struct remove_cv< const volatile _Ty>  { 
#line 140
using type = _Ty; 
#line 142
template< template< class >  class _Fn> using _Apply = const volatile _Fn< _Ty> ; 
#line 144
}; 
#line 146
template< class _Ty> using remove_cv_t = typename remove_cv< _Ty> ::type; 
#line 151
template< bool _First_value, class _First, class ..._Rest> 
#line 152
struct _Disjunction { 
#line 153
using type = _First; 
#line 154
}; 
#line 156
template< class _False, class _Next, class ..._Rest> 
#line 157
struct _Disjunction< false, _False, _Next, _Rest...>  { 
#line 158
using type = typename std::_Disjunction< _Next::value, _Next, _Rest...> ::type; 
#line 159
}; 
#line 161
template< class ..._Traits> 
#line 162
struct disjunction : public false_type { }; 
#line 164
template< class _First, class ..._Rest> 
#line 165
struct disjunction< _First, _Rest...>  : public _Disjunction< _First::value, _First, _Rest...> ::type { 
#line 167
}; 
#line 169
template< class ..._Traits> constexpr bool 
#line 170
disjunction_v = (disjunction< _Traits...> ::value); 
#line 172
template< class _Ty, class ..._Types> constexpr bool 
#line 173
_Is_any_of_v = disjunction_v< is_same< _Ty, _Types> ...> ; 
#line 176
template< class _Ty> constexpr bool 
#line 177
_Is_integral = _Is_any_of_v< _Ty, bool, char, signed char, unsigned char, __wchar_t, char16_t, char32_t, short, unsigned short, int, unsigned, long, unsigned long, __int64, unsigned __int64> ; 
#line 242 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\xtr1common"
template< class _Ty> constexpr bool 
#line 243
is_integral_v = _Is_integral< remove_cv_t< _Ty> > ; 
#line 245
template< class _Ty> 
#line 246
struct is_integral : public bool_constant< is_integral_v< _Ty> >  { }; 
#line 250
template< class _Ty> constexpr bool 
#line 251
_Is_floating_point = _Is_any_of_v< _Ty, float, double, long double> ; 
#line 269 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\xtr1common"
template< class _Ty> constexpr bool 
#line 270
is_floating_point_v = _Is_floating_point< remove_cv_t< _Ty> > ; 
#line 273
template< class _Ty> 
#line 274
struct is_floating_point : public bool_constant< is_floating_point_v< _Ty> >  { }; 
#line 277
template< class _Ty> constexpr bool 
#line 278
is_arithmetic_v = is_integral_v< _Ty>  || is_floating_point_v< _Ty> ; 
#line 281
template< class _Ty> 
#line 282
struct is_arithmetic : public bool_constant< is_arithmetic_v< _Ty> >  { }; 
#line 285
template< class _Ty> 
#line 286
struct remove_reference { 
#line 287
using type = _Ty; 
#line 288
}; 
#line 290
template< class _Ty> 
#line 291
struct remove_reference< _Ty &>  { 
#line 292
using type = _Ty; 
#line 293
}; 
#line 295
template< class _Ty> 
#line 296
struct remove_reference< _Ty &&>  { 
#line 297
using type = _Ty; 
#line 298
}; 
#line 300
template< class _Ty> using remove_reference_t = typename remove_reference< _Ty> ::type; 
#line 303
template< class _Ty> using _Remove_cvref_t = remove_cv_t< remove_reference_t< _Ty> > ; 
#line 316 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\xtr1common"
}
#line 319
#pragma warning(pop)
#pragma pack ( pop )
#line 13 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\xtgmath.h"
#pragma pack ( push, 8 )
#pragma warning(push, 3)
#pragma warning(disable : 4455 4494 4583 4619 4643 4702 4793 4988 )
#line 20
namespace std { 
#line 21
template< class _Ty1, class _Ty2> using _Common_float_type_t = conditional_t< is_same_v< _Ty1, long double>  || is_same_v< _Ty2, long double> , long double, conditional_t< is_same_v< _Ty1, float>  && is_same_v< _Ty2, float> , float, double> > ; 
#line 25
}
#line 57
template < class _Ty1, class _Ty2, class = :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 > > >
 inline :: std :: _Common_float_type_t < _Ty1, _Ty2 > pow (
    const _Ty1 _Left, const _Ty2 _Right ) {
    using _Common = :: std :: _Common_float_type_t < _Ty1, _Ty2 >;
    return :: pow ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Right ) );
}
#line 65
extern "C" double __cdecl acos(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double acos ( _Ty _Left ) { return :: acos ( static_cast < double > ( _Left ) ); }
#line 66
extern "C" double __cdecl asin(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double asin ( _Ty _Left ) { return :: asin ( static_cast < double > ( _Left ) ); }
#line 67
extern "C" double __cdecl atan(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double atan ( _Ty _Left ) { return :: atan ( static_cast < double > ( _Left ) ); }
#line 68
extern "C" double __cdecl atan2(double, double); template < class _Ty1, class _Ty2, class = :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 > > > inline :: std :: _Common_float_type_t < _Ty1, _Ty2 > atan2 ( _Ty1 _Left, _Ty2 _Right ) { using _Common = :: std :: _Common_float_type_t < _Ty1, _Ty2 >; return :: atan2 ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Right ) ); }
#line 69
extern "C" double __cdecl ceil(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double ceil ( _Ty _Left ) { return :: ceil ( static_cast < double > ( _Left ) ); }
#line 70
extern "C" double __cdecl cos(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double cos ( _Ty _Left ) { return :: cos ( static_cast < double > ( _Left ) ); }
#line 71
extern "C" double __cdecl cosh(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double cosh ( _Ty _Left ) { return :: cosh ( static_cast < double > ( _Left ) ); }
#line 72
extern "C" double __cdecl exp(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double exp ( _Ty _Left ) { return :: exp ( static_cast < double > ( _Left ) ); }
#line 74
extern "C" double __cdecl fabs(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double fabs ( _Ty _Left ) { return :: fabs ( static_cast < double > ( _Left ) ); }
#line 76
extern "C" double __cdecl floor(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double floor ( _Ty _Left ) { return :: floor ( static_cast < double > ( _Left ) ); }
#line 77
extern "C" double __cdecl fmod(double, double); template < class _Ty1, class _Ty2, class = :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 > > > inline :: std :: _Common_float_type_t < _Ty1, _Ty2 > fmod ( _Ty1 _Left, _Ty2 _Right ) { using _Common = :: std :: _Common_float_type_t < _Ty1, _Ty2 >; return :: fmod ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Right ) ); }
#line 78
extern "C" double __cdecl frexp(double, int *); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double frexp ( _Ty _Left, int * _Arg2 ) { return :: frexp ( static_cast < double > ( _Left ), _Arg2 ); }
#line 79
extern "C" double __cdecl ldexp(double, int); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double ldexp ( _Ty _Left, int _Arg2 ) { return :: ldexp ( static_cast < double > ( _Left ), _Arg2 ); }
#line 80
extern "C" double __cdecl log(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double log ( _Ty _Left ) { return :: log ( static_cast < double > ( _Left ) ); }
#line 81
extern "C" double __cdecl log10(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double log10 ( _Ty _Left ) { return :: log10 ( static_cast < double > ( _Left ) ); }
#line 84
extern "C" double __cdecl sin(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double sin ( _Ty _Left ) { return :: sin ( static_cast < double > ( _Left ) ); }
#line 85
extern "C" double __cdecl sinh(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double sinh ( _Ty _Left ) { return :: sinh ( static_cast < double > ( _Left ) ); }
#line 86
extern "C" double __cdecl sqrt(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double sqrt ( _Ty _Left ) { return :: sqrt ( static_cast < double > ( _Left ) ); }
#line 87
extern "C" double __cdecl tan(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double tan ( _Ty _Left ) { return :: tan ( static_cast < double > ( _Left ) ); }
#line 88
extern "C" double __cdecl tanh(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double tanh ( _Ty _Left ) { return :: tanh ( static_cast < double > ( _Left ) ); }
#line 94
inline float _Fma(float _Left, float _Middle, float _Right) { 
#line 95
return ::fmaf(_Left, _Middle, _Right); 
#line 96
} 
#line 98
inline double _Fma(double _Left, double _Middle, double _Right) { 
#line 99
return ::fma(_Left, _Middle, _Right); 
#line 100
} 
#line 102
inline long double _Fma(long double _Left, long double _Middle, long double 
#line 103
_Right) { 
#line 104
return ::fmal(_Left, _Middle, _Right); 
#line 105
} 
#line 108 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\xtgmath.h"
template < class _Ty1, class _Ty2, class _Ty3,
    class = :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 > && :: std :: is_arithmetic_v < _Ty3 > > >
 inline :: std :: _Common_float_type_t < _Ty1, :: std :: _Common_float_type_t < _Ty2, _Ty3 > > fma (
    _Ty1 _Left, _Ty2 _Middle, _Ty3 _Right ) {
    using _Common = :: std :: _Common_float_type_t < _Ty1, :: std :: _Common_float_type_t < _Ty2, _Ty3 >>;









    return _Fma ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Middle ), static_cast < _Common > ( _Right ) );

}
#line 128 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\xtgmath.h"
inline float _Remquo(float _Left, float _Right, int *_Pquo) { 
#line 129
return ::remquof(_Left, _Right, _Pquo); 
#line 130
} 
#line 132
inline double _Remquo(double _Left, double _Right, int *_Pquo) { 
#line 133
return ::remquo(_Left, _Right, _Pquo); 
#line 134
} 
#line 136
inline long double _Remquo(long double _Left, long double _Right, int *_Pquo) { 
#line 137
return ::remquol(_Left, _Right, _Pquo); 
#line 138
} 
#line 141 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\xtgmath.h"
template < class _Ty1, class _Ty2, class = :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 > > >
inline :: std :: _Common_float_type_t < _Ty1, _Ty2 > remquo (
    _Ty1 _Left, _Ty2 _Right, int * _Pquo ) {
    using _Common = :: std :: _Common_float_type_t < _Ty1, _Ty2 >;









    return _Remquo ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Right ), _Pquo );

}
#line 158 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\xtgmath.h"
extern "C" double __cdecl acosh(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double acosh ( _Ty _Left ) { return :: acosh ( static_cast < double > ( _Left ) ); }
#line 159
extern "C" double __cdecl asinh(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double asinh ( _Ty _Left ) { return :: asinh ( static_cast < double > ( _Left ) ); }
#line 160
extern "C" double __cdecl atanh(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double atanh ( _Ty _Left ) { return :: atanh ( static_cast < double > ( _Left ) ); }
#line 161
extern "C" double __cdecl cbrt(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double cbrt ( _Ty _Left ) { return :: cbrt ( static_cast < double > ( _Left ) ); }
#line 162
extern "C" double __cdecl copysign(double, double); template < class _Ty1, class _Ty2, class = :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 > > > inline :: std :: _Common_float_type_t < _Ty1, _Ty2 > copysign ( _Ty1 _Left, _Ty2 _Right ) { using _Common = :: std :: _Common_float_type_t < _Ty1, _Ty2 >; return :: copysign ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Right ) ); }
#line 163
extern "C" double __cdecl erf(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double erf ( _Ty _Left ) { return :: erf ( static_cast < double > ( _Left ) ); }
#line 164
extern "C" double __cdecl erfc(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double erfc ( _Ty _Left ) { return :: erfc ( static_cast < double > ( _Left ) ); }
#line 165
extern "C" double __cdecl expm1(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double expm1 ( _Ty _Left ) { return :: expm1 ( static_cast < double > ( _Left ) ); }
#line 166
extern "C" double __cdecl exp2(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double exp2 ( _Ty _Left ) { return :: exp2 ( static_cast < double > ( _Left ) ); }
#line 167
extern "C" double __cdecl fdim(double, double); template < class _Ty1, class _Ty2, class = :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 > > > inline :: std :: _Common_float_type_t < _Ty1, _Ty2 > fdim ( _Ty1 _Left, _Ty2 _Right ) { using _Common = :: std :: _Common_float_type_t < _Ty1, _Ty2 >; return :: fdim ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Right ) ); }
#line 169
extern "C" double __cdecl fmax(double, double); template < class _Ty1, class _Ty2, class = :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 > > > inline :: std :: _Common_float_type_t < _Ty1, _Ty2 > fmax ( _Ty1 _Left, _Ty2 _Right ) { using _Common = :: std :: _Common_float_type_t < _Ty1, _Ty2 >; return :: fmax ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Right ) ); }
#line 170
extern "C" double __cdecl fmin(double, double); template < class _Ty1, class _Ty2, class = :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 > > > inline :: std :: _Common_float_type_t < _Ty1, _Ty2 > fmin ( _Ty1 _Left, _Ty2 _Right ) { using _Common = :: std :: _Common_float_type_t < _Ty1, _Ty2 >; return :: fmin ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Right ) ); }
#line 171
extern "C" double __cdecl hypot(double, double); template < class _Ty1, class _Ty2, class = :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 > > > inline :: std :: _Common_float_type_t < _Ty1, _Ty2 > hypot ( _Ty1 _Left, _Ty2 _Right ) { using _Common = :: std :: _Common_float_type_t < _Ty1, _Ty2 >; return :: hypot ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Right ) ); }
#line 172
extern "C" int __cdecl ilogb(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline int ilogb ( _Ty _Left ) { return :: ilogb ( static_cast < double > ( _Left ) ); }
#line 173
extern "C" double __cdecl lgamma(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double lgamma ( _Ty _Left ) { return :: lgamma ( static_cast < double > ( _Left ) ); }
#line 174
extern "C" __int64 __cdecl llrint(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline long long llrint ( _Ty _Left ) { return :: llrint ( static_cast < double > ( _Left ) ); }
#line 175
extern "C" __int64 __cdecl llround(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline long long llround ( _Ty _Left ) { return :: llround ( static_cast < double > ( _Left ) ); }
#line 176
extern "C" double __cdecl log1p(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double log1p ( _Ty _Left ) { return :: log1p ( static_cast < double > ( _Left ) ); }
#line 177
extern "C" double __cdecl log2(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double log2 ( _Ty _Left ) { return :: log2 ( static_cast < double > ( _Left ) ); }
#line 178
extern "C" double __cdecl logb(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double logb ( _Ty _Left ) { return :: logb ( static_cast < double > ( _Left ) ); }
#line 179
extern "C" long __cdecl lrint(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline long lrint ( _Ty _Left ) { return :: lrint ( static_cast < double > ( _Left ) ); }
#line 180
extern "C" long __cdecl lround(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline long lround ( _Ty _Left ) { return :: lround ( static_cast < double > ( _Left ) ); }
#line 181
extern "C" double __cdecl nearbyint(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double nearbyint ( _Ty _Left ) { return :: nearbyint ( static_cast < double > ( _Left ) ); }
#line 182
extern "C" double __cdecl nextafter(double, double); template < class _Ty1, class _Ty2, class = :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 > > > inline :: std :: _Common_float_type_t < _Ty1, _Ty2 > nextafter ( _Ty1 _Left, _Ty2 _Right ) { using _Common = :: std :: _Common_float_type_t < _Ty1, _Ty2 >; return :: nextafter ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Right ) ); }
#line 183
extern "C" double __cdecl nexttoward(double, long double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double nexttoward ( _Ty _Left, long double _Arg2 ) { return :: nexttoward ( static_cast < double > ( _Left ), _Arg2 ); }
#line 184
extern "C" double __cdecl remainder(double, double); template < class _Ty1, class _Ty2, class = :: std :: enable_if_t < :: std :: is_arithmetic_v < _Ty1 > && :: std :: is_arithmetic_v < _Ty2 > > > inline :: std :: _Common_float_type_t < _Ty1, _Ty2 > remainder ( _Ty1 _Left, _Ty2 _Right ) { using _Common = :: std :: _Common_float_type_t < _Ty1, _Ty2 >; return :: remainder ( static_cast < _Common > ( _Left ), static_cast < _Common > ( _Right ) ); }
#line 186
extern "C" double __cdecl rint(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double rint ( _Ty _Left ) { return :: rint ( static_cast < double > ( _Left ) ); }
#line 187
extern "C" double __cdecl round(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double round ( _Ty _Left ) { return :: round ( static_cast < double > ( _Left ) ); }
#line 188
extern "C" double __cdecl scalbln(double, long); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double scalbln ( _Ty _Left, long _Arg2 ) { return :: scalbln ( static_cast < double > ( _Left ), _Arg2 ); }
#line 189
extern "C" double __cdecl scalbn(double, int); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double scalbn ( _Ty _Left, int _Arg2 ) { return :: scalbn ( static_cast < double > ( _Left ), _Arg2 ); }
#line 190
extern "C" double __cdecl tgamma(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double tgamma ( _Ty _Left ) { return :: tgamma ( static_cast < double > ( _Left ) ); }
#line 191
extern "C" double __cdecl trunc(double); template < class _Ty, class = :: std :: enable_if_t < :: std :: is_integral_v < _Ty > > > inline double trunc ( _Ty _Left ) { return :: trunc ( static_cast < double > ( _Left ) ); }
#line 203
#pragma warning(pop)
#pragma pack ( pop )
#line 492 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\cmath"
namespace std { 
#line 493
using ::abs;
#line 494
using ::acos;
#line 495
using ::asin;
#line 496
using ::atan;
#line 497
using ::atan2;
#line 498
using ::ceil;
#line 499
using ::cos;
#line 500
using ::cosh;
#line 501
using ::exp;
#line 502
using ::fabs;
#line 503
using ::floor;
#line 504
using ::fmod;
#line 505
using ::frexp;
#line 506
using ::ldexp;
#line 507
using ::log;
#line 508
using ::log10;
#line 509
using ::modf;
#line 510
using ::pow;
#line 511
using ::sin;
#line 512
using ::sinh;
#line 513
using ::sqrt;
#line 514
using ::tan;
#line 515
using ::tanh;
#line 517
using ::acosf;
#line 518
using ::asinf;
#line 519
using ::atanf;
#line 520
using ::atan2f;
#line 521
using ::ceilf;
#line 522
using ::cosf;
#line 523
using ::coshf;
#line 524
using ::expf;
#line 525
using ::fabsf;
#line 526
using ::floorf;
#line 527
using ::fmodf;
#line 528
using ::frexpf;
#line 529
using ::ldexpf;
#line 530
using ::logf;
#line 531
using ::log10f;
#line 532
using ::modff;
#line 533
using ::powf;
#line 534
using ::sinf;
#line 535
using ::sinhf;
#line 536
using ::sqrtf;
#line 537
using ::tanf;
#line 538
using ::tanhf;
#line 540
using ::acosl;
#line 541
using ::asinl;
#line 542
using ::atanl;
#line 543
using ::atan2l;
#line 544
using ::ceill;
#line 545
using ::cosl;
#line 546
using ::coshl;
#line 547
using ::expl;
#line 548
using ::fabsl;
#line 549
using ::floorl;
#line 550
using ::fmodl;
#line 551
using ::frexpl;
#line 552
using ::ldexpl;
#line 553
using ::logl;
#line 554
using ::log10l;
#line 555
using ::modfl;
#line 556
using ::powl;
#line 557
using ::sinl;
#line 558
using ::sinhl;
#line 559
using ::sqrtl;
#line 560
using ::tanl;
#line 561
using ::tanhl;
#line 563
using ::float_t;
#line 564
using ::double_t;
#line 566
using ::acosh;
#line 567
using ::asinh;
#line 568
using ::atanh;
#line 569
using ::cbrt;
#line 570
using ::erf;
#line 571
using ::erfc;
#line 572
using ::expm1;
#line 573
using ::exp2;
#line 574
using ::hypot;
#line 575
using ::ilogb;
#line 576
using ::lgamma;
#line 577
using ::log1p;
#line 578
using ::log2;
#line 579
using ::logb;
#line 580
using ::llrint;
#line 581
using ::lrint;
#line 582
using ::nearbyint;
#line 583
using ::rint;
#line 584
using ::llround;
#line 585
using ::lround;
#line 586
using ::fdim;
#line 587
using ::fma;
#line 588
using ::fmax;
#line 589
using ::fmin;
#line 590
using ::round;
#line 591
using ::trunc;
#line 592
using ::remainder;
#line 593
using ::remquo;
#line 594
using ::copysign;
#line 595
using ::nan;
#line 596
using ::nextafter;
#line 597
using ::scalbn;
#line 598
using ::scalbln;
#line 599
using ::nexttoward;
#line 600
using ::tgamma;
#line 602
using ::acoshf;
#line 603
using ::asinhf;
#line 604
using ::atanhf;
#line 605
using ::cbrtf;
#line 606
using ::erff;
#line 607
using ::erfcf;
#line 608
using ::expm1f;
#line 609
using ::exp2f;
#line 610
using ::hypotf;
#line 611
using ::ilogbf;
#line 612
using ::lgammaf;
#line 613
using ::log1pf;
#line 614
using ::log2f;
#line 615
using ::logbf;
#line 616
using ::llrintf;
#line 617
using ::lrintf;
#line 618
using ::nearbyintf;
#line 619
using ::rintf;
#line 620
using ::llroundf;
#line 621
using ::lroundf;
#line 622
using ::fdimf;
#line 623
using ::fmaf;
#line 624
using ::fmaxf;
#line 625
using ::fminf;
#line 626
using ::roundf;
#line 627
using ::truncf;
#line 628
using ::remainderf;
#line 629
using ::remquof;
#line 630
using ::copysignf;
#line 631
using ::nanf;
#line 632
using ::nextafterf;
#line 633
using ::scalbnf;
#line 634
using ::scalblnf;
#line 635
using ::nexttowardf;
#line 636
using ::tgammaf;
#line 638
using ::acoshl;
#line 639
using ::asinhl;
#line 640
using ::atanhl;
#line 641
using ::cbrtl;
#line 642
using ::erfl;
#line 643
using ::erfcl;
#line 644
using ::expm1l;
#line 645
using ::exp2l;
#line 646
using ::hypotl;
#line 647
using ::ilogbl;
#line 648
using ::lgammal;
#line 649
using ::log1pl;
#line 650
using ::log2l;
#line 651
using ::logbl;
#line 652
using ::llrintl;
#line 653
using ::lrintl;
#line 654
using ::nearbyintl;
#line 655
using ::rintl;
#line 656
using ::llroundl;
#line 657
using ::lroundl;
#line 658
using ::fdiml;
#line 659
using ::fmal;
#line 660
using ::fmaxl;
#line 661
using ::fminl;
#line 662
using ::roundl;
#line 663
using ::truncl;
#line 664
using ::remainderl;
#line 665
using ::remquol;
#line 666
using ::copysignl;
#line 667
using ::nanl;
#line 668
using ::nextafterl;
#line 669
using ::scalbnl;
#line 670
using ::scalblnl;
#line 671
using ::nexttowardl;
#line 672
using ::tgammal;
#line 674
using ::fpclassify;
#line 675
using ::signbit;
#line 676
using ::isfinite;
#line 677
using ::isinf;
#line 678
using ::isnan;
#line 679
using ::isnormal;
#line 680
using ::isgreater;
#line 681
using ::isgreaterequal;
#line 682
using ::isless;
#line 683
using ::islessequal;
#line 684
using ::islessgreater;
#line 685
using ::isunordered;
#line 686
}
#line 1085 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.22.27905\\include\\cmath"
#pragma warning(pop)
#pragma pack ( pop )
#line 9227 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern "C" double __cdecl _hypot(double x, double y); 
#line 9228
extern "C" float __cdecl _hypotf(float x, float y); 
#line 9238 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern inline bool signbit(long double) throw(); 
#line 9239
extern "C" int _ldsign(long double); 
#line 9282 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern inline bool signbit(double) throw(); 
#line 9283
extern "C" int _dsign(double); 
#line 9327 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern inline bool signbit(float) throw(); 
#line 9328
extern "C" int _fdsign(float); 
#line 9336 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
static __inline bool isinf(long double a); 
#line 9371 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
static __inline bool isinf(double a); 
#line 9409 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
static __inline bool isinf(float a); 
#line 9416 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
static __inline bool isnan(long double a); 
#line 9449 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
static __inline bool isnan(double a); 
#line 9485 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
static __inline bool isnan(float a); 
#line 9492 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
static __inline bool isfinite(long double a); 
#line 9529 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
static __inline bool isfinite(double a); 
#line 9565 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
static __inline bool isfinite(float a); 
#line 9573 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
template< class T> extern T _Pow_int(T, int) throw(); 
#line 9574
extern inline __int64 abs(__int64) throw(); 
#line 9649 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern inline long __cdecl abs(long) throw(); 
#line 9653 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern inline float __cdecl abs(float) throw(); 
#line 9654
extern inline double __cdecl abs(double) throw(); 
#line 9655
extern inline float __cdecl fabs(float) throw(); 
#line 9656
extern inline float __cdecl ceil(float) throw(); 
#line 9657
extern inline float __cdecl floor(float) throw(); 
#line 9658
extern inline float __cdecl sqrt(float) throw(); 
#line 9659
extern inline float __cdecl pow(float, float) throw(); 
#line 9693 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern inline float __cdecl pow(float, int) throw(); 
#line 9694
extern inline double __cdecl pow(double, int) throw(); 
#line 9697 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern inline float __cdecl log(float) throw(); 
#line 9698
extern inline float __cdecl log10(float) throw(); 
#line 9699
extern inline float __cdecl fmod(float, float) throw(); 
#line 9700
extern inline float __cdecl modf(float, float *) throw(); 
#line 9701
extern inline float __cdecl exp(float) throw(); 
#line 9702
extern inline float __cdecl frexp(float, int *) throw(); 
#line 9703
extern inline float __cdecl ldexp(float, int) throw(); 
#line 9704
extern inline float __cdecl asin(float) throw(); 
#line 9705
extern inline float __cdecl sin(float) throw(); 
#line 9706
extern inline float __cdecl sinh(float) throw(); 
#line 9707
extern inline float __cdecl acos(float) throw(); 
#line 9708
extern inline float __cdecl cos(float) throw(); 
#line 9709
extern inline float __cdecl cosh(float) throw(); 
#line 9710
extern inline float __cdecl atan(float) throw(); 
#line 9711
extern inline float __cdecl atan2(float, float) throw(); 
#line 9712
extern inline float __cdecl tan(float) throw(); 
#line 9713
extern inline float __cdecl tanh(float) throw(); 
#line 9935 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
extern inline float __cdecl logb(float) throw(); 
#line 9936
extern inline int __cdecl ilogb(float) throw(); 
#line 9937
extern float __cdecl scalbn(float, float) throw(); 
#line 9938
extern inline float __cdecl scalbln(float, long) throw(); 
#line 9939
extern inline float __cdecl exp2(float) throw(); 
#line 9940
extern inline float __cdecl expm1(float) throw(); 
#line 9941
extern inline float __cdecl log2(float) throw(); 
#line 9942
extern inline float __cdecl log1p(float) throw(); 
#line 9943
extern inline float __cdecl acosh(float) throw(); 
#line 9944
extern inline float __cdecl asinh(float) throw(); 
#line 9945
extern inline float __cdecl atanh(float) throw(); 
#line 9946
extern inline float __cdecl hypot(float, float) throw(); 
#line 9947
extern inline float __cdecl cbrt(float) throw(); 
#line 9948
extern inline float __cdecl erf(float) throw(); 
#line 9949
extern inline float __cdecl erfc(float) throw(); 
#line 9950
extern inline float __cdecl lgamma(float) throw(); 
#line 9951
extern inline float __cdecl tgamma(float) throw(); 
#line 9952
extern inline float __cdecl copysign(float, float) throw(); 
#line 9953
extern inline float __cdecl nextafter(float, float) throw(); 
#line 9954
extern inline float __cdecl remainder(float, float) throw(); 
#line 9955
extern inline float __cdecl remquo(float, float, int *) throw(); 
#line 9956
extern inline float __cdecl round(float) throw(); 
#line 9957
extern inline long __cdecl lround(float) throw(); 
#line 9958
extern inline __int64 __cdecl llround(float) throw(); 
#line 9959
extern inline float __cdecl trunc(float) throw(); 
#line 9960
extern inline float __cdecl rint(float) throw(); 
#line 9961
extern inline long __cdecl lrint(float) throw(); 
#line 9962
extern inline __int64 __cdecl llrint(float) throw(); 
#line 9963
extern inline float __cdecl nearbyint(float) throw(); 
#line 9964
extern inline float __cdecl fdim(float, float) throw(); 
#line 9965
extern inline float __cdecl fma(float, float, float) throw(); 
#line 9966
extern inline float __cdecl fmax(float, float) throw(); 
#line 9967
extern inline float __cdecl fmin(float, float) throw(); 
#line 9970 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
static inline float exp10(float a); 
#line 9972
static inline float rsqrt(float a); 
#line 9974
static inline float rcbrt(float a); 
#line 9976
static inline float sinpi(float a); 
#line 9978
static inline float cospi(float a); 
#line 9980
static inline void sincospi(float a, float * sptr, float * cptr); 
#line 9982
static inline void sincos(float a, float * sptr, float * cptr); 
#line 9984
static inline float j0(float a); 
#line 9986
static inline float j1(float a); 
#line 9988
static inline float jn(int n, float a); 
#line 9990
static inline float y0(float a); 
#line 9992
static inline float y1(float a); 
#line 9994
static inline float yn(int n, float a); 
#line 9996
static inline float cyl_bessel_i0(float a); 
#line 9998
static inline float cyl_bessel_i1(float a); 
#line 10000
static inline float erfinv(float a); 
#line 10002
static inline float erfcinv(float a); 
#line 10004
static inline float normcdfinv(float a); 
#line 10006
static inline float normcdf(float a); 
#line 10008
static inline float erfcx(float a); 
#line 10010
static inline double copysign(double a, float b); 
#line 10012
static inline double copysign(float a, double b); 
#line 10020
static inline unsigned min(unsigned a, unsigned b); 
#line 10028
static inline unsigned min(int a, unsigned b); 
#line 10036
static inline unsigned min(unsigned a, int b); 
#line 10044
static inline long min(long a, long b); 
#line 10052
static inline unsigned long min(unsigned long a, unsigned long b); 
#line 10060
static inline unsigned long min(long a, unsigned long b); 
#line 10068
static inline unsigned long min(unsigned long a, long b); 
#line 10076
static inline __int64 min(__int64 a, __int64 b); 
#line 10084
static inline unsigned __int64 min(unsigned __int64 a, unsigned __int64 b); 
#line 10092
static inline unsigned __int64 min(__int64 a, unsigned __int64 b); 
#line 10100
static inline unsigned __int64 min(unsigned __int64 a, __int64 b); 
#line 10111
static inline float min(float a, float b); 
#line 10122
static inline double min(double a, double b); 
#line 10132
static inline double min(float a, double b); 
#line 10142
static inline double min(double a, float b); 
#line 10153
static inline unsigned max(unsigned a, unsigned b); 
#line 10161
static inline unsigned max(int a, unsigned b); 
#line 10169
static inline unsigned max(unsigned a, int b); 
#line 10177
static inline long max(long a, long b); 
#line 10185
static inline unsigned long max(unsigned long a, unsigned long b); 
#line 10193
static inline unsigned long max(long a, unsigned long b); 
#line 10201
static inline unsigned long max(unsigned long a, long b); 
#line 10209
static inline __int64 max(__int64 a, __int64 b); 
#line 10217
static inline unsigned __int64 max(unsigned __int64 a, unsigned __int64 b); 
#line 10225
static inline unsigned __int64 max(__int64 a, unsigned __int64 b); 
#line 10233
static inline unsigned __int64 max(unsigned __int64 a, __int64 b); 
#line 10244
static inline float max(float a, float b); 
#line 10255
static inline double max(double a, double b); 
#line 10265
static inline double max(float a, double b); 
#line 10275
static inline double max(double a, float b); 
#line 10286
extern "C" {
#line 10287
inline void *__nv_aligned_device_malloc(size_t size, size_t align) {int volatile ___ = 1;(void)size;(void)align;::exit(___);}
#if 0
#line 10288
{ 
#line 10289
void *__nv_aligned_device_malloc_impl(size_t, size_t); 
#line 10290
return __nv_aligned_device_malloc_impl(size, align); 
#line 10291
} 
#endif
#line 10292 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.h"
}
#line 433 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
static __inline bool isinf(const long double a) 
#line 434
{ 
#line 438 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
return isinf< long double> (a); 
#line 440 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
} 
#line 449 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
static __inline bool isinf(const double a) 
#line 450
{ 
#line 454 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
return isinf< double> (a); 
#line 456 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
} 
#line 465 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
static __inline bool isinf(const float a) 
#line 466
{ 
#line 470 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
return isinf< float> (a); 
#line 472 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
} 
#line 481 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
static __inline bool isnan(const long double a) 
#line 482
{ 
#line 486 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
return isnan< long double> (a); 
#line 488 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
} 
#line 497 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
static __inline bool isnan(const double a) 
#line 498
{ 
#line 502 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
return isnan< double> (a); 
#line 504 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
} 
#line 513 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
static __inline bool isnan(const float a) 
#line 514
{ 
#line 518 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
return isnan< float> (a); 
#line 520 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
} 
#line 529 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
static __inline bool isfinite(const long double a) 
#line 530
{ 
#line 534 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
return isfinite< long double> (a); 
#line 536 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
} 
#line 545 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
static __inline bool isfinite(const double a) 
#line 546
{ 
#line 550 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
return isfinite< double> (a); 
#line 552 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
} 
#line 561 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
static __inline bool isfinite(const float a) 
#line 562
{ 
#line 566 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
return isfinite< float> (a); 
#line 568 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
} 
#line 758 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
static inline float exp10(const float a) 
#line 759
{ 
#line 760
return exp10f(a); 
#line 761
} 
#line 763
static inline float rsqrt(const float a) 
#line 764
{ 
#line 765
return rsqrtf(a); 
#line 766
} 
#line 768
static inline float rcbrt(const float a) 
#line 769
{ 
#line 770
return rcbrtf(a); 
#line 771
} 
#line 773
static inline float sinpi(const float a) 
#line 774
{ 
#line 775
return sinpif(a); 
#line 776
} 
#line 778
static inline float cospi(const float a) 
#line 779
{ 
#line 780
return cospif(a); 
#line 781
} 
#line 783
static inline void sincospi(const float a, float *const sptr, float *const cptr) 
#line 784
{ 
#line 785
sincospif(a, sptr, cptr); 
#line 786
} 
#line 788
static inline void sincos(const float a, float *const sptr, float *const cptr) 
#line 789
{ 
#line 790
sincosf(a, sptr, cptr); 
#line 791
} 
#line 793
static inline float j0(const float a) 
#line 794
{ 
#line 795
return j0f(a); 
#line 796
} 
#line 798
static inline float j1(const float a) 
#line 799
{ 
#line 800
return j1f(a); 
#line 801
} 
#line 803
static inline float jn(const int n, const float a) 
#line 804
{ 
#line 805
return jnf(n, a); 
#line 806
} 
#line 808
static inline float y0(const float a) 
#line 809
{ 
#line 810
return y0f(a); 
#line 811
} 
#line 813
static inline float y1(const float a) 
#line 814
{ 
#line 815
return y1f(a); 
#line 816
} 
#line 818
static inline float yn(const int n, const float a) 
#line 819
{ 
#line 820
return ynf(n, a); 
#line 821
} 
#line 823
static inline float cyl_bessel_i0(const float a) {int volatile ___ = 1;(void)a;::exit(___);}
#if 0
#line 824
{ 
#line 825
return cyl_bessel_i0f(a); 
#line 826
} 
#endif
#line 828 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
static inline float cyl_bessel_i1(const float a) {int volatile ___ = 1;(void)a;::exit(___);}
#if 0
#line 829
{ 
#line 830
return cyl_bessel_i1f(a); 
#line 831
} 
#endif
#line 833 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
static inline float erfinv(const float a) 
#line 834
{ 
#line 835
return erfinvf(a); 
#line 836
} 
#line 838
static inline float erfcinv(const float a) 
#line 839
{ 
#line 840
return erfcinvf(a); 
#line 841
} 
#line 843
static inline float normcdfinv(const float a) 
#line 844
{ 
#line 845
return normcdfinvf(a); 
#line 846
} 
#line 848
static inline float normcdf(const float a) 
#line 849
{ 
#line 850
return normcdff(a); 
#line 851
} 
#line 853
static inline float erfcx(const float a) 
#line 854
{ 
#line 855
return erfcxf(a); 
#line 856
} 
#line 858
static inline double copysign(const double a, const float b) 
#line 859
{ 
#line 860
return copysign(a, static_cast< double>(b)); 
#line 861
} 
#line 863
static inline double copysign(const float a, const double b) 
#line 864
{ 
#line 865
return copysign(static_cast< double>(a), b); 
#line 866
} 
#line 868
static inline unsigned min(const unsigned a, const unsigned b) 
#line 869
{ 
#line 870
return umin(a, b); 
#line 871
} 
#line 873
static inline unsigned min(const int a, const unsigned b) 
#line 874
{ 
#line 875
return umin(static_cast< unsigned>(a), b); 
#line 876
} 
#line 878
static inline unsigned min(const unsigned a, const int b) 
#line 879
{ 
#line 880
return umin(a, static_cast< unsigned>(b)); 
#line 881
} 
#line 883
static inline long min(const long a, const long b) 
#line 884
{ 
#line 885
long retval; 
#line 888
#pragma warning (disable: 4127)
#line 891 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
if (sizeof(long) == sizeof(int)) { 
#line 893
#pragma warning (default: 4127)
#line 895 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
retval = (static_cast< long>(min(static_cast< int>(a), static_cast< int>(b)))); 
#line 896
} else { 
#line 897
retval = (static_cast< long>(llmin(static_cast< __int64>(a), static_cast< __int64>(b)))); 
#line 898
}  
#line 899
return retval; 
#line 900
} 
#line 902
static inline unsigned long min(const unsigned long a, const unsigned long b) 
#line 903
{ 
#line 904
unsigned long retval; 
#line 906
#pragma warning (disable: 4127)
#line 908 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
if (sizeof(unsigned long) == sizeof(unsigned)) { 
#line 910
#pragma warning (default: 4127)
#line 912 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
retval = (static_cast< unsigned long>(umin(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
#line 913
} else { 
#line 914
retval = (static_cast< unsigned long>(ullmin(static_cast< unsigned __int64>(a), static_cast< unsigned __int64>(b)))); 
#line 915
}  
#line 916
return retval; 
#line 917
} 
#line 919
static inline unsigned long min(const long a, const unsigned long b) 
#line 920
{ 
#line 921
unsigned long retval; 
#line 923
#pragma warning (disable: 4127)
#line 925 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
if (sizeof(unsigned long) == sizeof(unsigned)) { 
#line 927
#pragma warning (default: 4127)
#line 929 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
retval = (static_cast< unsigned long>(umin(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
#line 930
} else { 
#line 931
retval = (static_cast< unsigned long>(ullmin(static_cast< unsigned __int64>(a), static_cast< unsigned __int64>(b)))); 
#line 932
}  
#line 933
return retval; 
#line 934
} 
#line 936
static inline unsigned long min(const unsigned long a, const long b) 
#line 937
{ 
#line 938
unsigned long retval; 
#line 940
#pragma warning (disable: 4127)
#line 942 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
if (sizeof(unsigned long) == sizeof(unsigned)) { 
#line 944
#pragma warning (default: 4127)
#line 946 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
retval = (static_cast< unsigned long>(umin(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
#line 947
} else { 
#line 948
retval = (static_cast< unsigned long>(ullmin(static_cast< unsigned __int64>(a), static_cast< unsigned __int64>(b)))); 
#line 949
}  
#line 950
return retval; 
#line 951
} 
#line 953
static inline __int64 min(const __int64 a, const __int64 b) 
#line 954
{ 
#line 955
return llmin(a, b); 
#line 956
} 
#line 958
static inline unsigned __int64 min(const unsigned __int64 a, const unsigned __int64 b) 
#line 959
{ 
#line 960
return ullmin(a, b); 
#line 961
} 
#line 963
static inline unsigned __int64 min(const __int64 a, const unsigned __int64 b) 
#line 964
{ 
#line 965
return ullmin(static_cast< unsigned __int64>(a), b); 
#line 966
} 
#line 968
static inline unsigned __int64 min(const unsigned __int64 a, const __int64 b) 
#line 969
{ 
#line 970
return ullmin(a, static_cast< unsigned __int64>(b)); 
#line 971
} 
#line 973
static inline float min(const float a, const float b) 
#line 974
{ 
#line 975
return fminf(a, b); 
#line 976
} 
#line 978
static inline double min(const double a, const double b) 
#line 979
{ 
#line 980
return fmin(a, b); 
#line 981
} 
#line 983
static inline double min(const float a, const double b) 
#line 984
{ 
#line 985
return fmin(static_cast< double>(a), b); 
#line 986
} 
#line 988
static inline double min(const double a, const float b) 
#line 989
{ 
#line 990
return fmin(a, static_cast< double>(b)); 
#line 991
} 
#line 993
static inline unsigned max(const unsigned a, const unsigned b) 
#line 994
{ 
#line 995
return umax(a, b); 
#line 996
} 
#line 998
static inline unsigned max(const int a, const unsigned b) 
#line 999
{ 
#line 1000
return umax(static_cast< unsigned>(a), b); 
#line 1001
} 
#line 1003
static inline unsigned max(const unsigned a, const int b) 
#line 1004
{ 
#line 1005
return umax(a, static_cast< unsigned>(b)); 
#line 1006
} 
#line 1008
static inline long max(const long a, const long b) 
#line 1009
{ 
#line 1010
long retval; 
#line 1013
#pragma warning (disable: 4127)
#line 1015 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
if (sizeof(long) == sizeof(int)) { 
#line 1017
#pragma warning (default: 4127)
#line 1019 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
retval = (static_cast< long>(max(static_cast< int>(a), static_cast< int>(b)))); 
#line 1020
} else { 
#line 1021
retval = (static_cast< long>(llmax(static_cast< __int64>(a), static_cast< __int64>(b)))); 
#line 1022
}  
#line 1023
return retval; 
#line 1024
} 
#line 1026
static inline unsigned long max(const unsigned long a, const unsigned long b) 
#line 1027
{ 
#line 1028
unsigned long retval; 
#line 1030
#pragma warning (disable: 4127)
#line 1032 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
if (sizeof(unsigned long) == sizeof(unsigned)) { 
#line 1034
#pragma warning (default: 4127)
#line 1036 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
retval = (static_cast< unsigned long>(umax(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
#line 1037
} else { 
#line 1038
retval = (static_cast< unsigned long>(ullmax(static_cast< unsigned __int64>(a), static_cast< unsigned __int64>(b)))); 
#line 1039
}  
#line 1040
return retval; 
#line 1041
} 
#line 1043
static inline unsigned long max(const long a, const unsigned long b) 
#line 1044
{ 
#line 1045
unsigned long retval; 
#line 1047
#pragma warning (disable: 4127)
#line 1049 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
if (sizeof(unsigned long) == sizeof(unsigned)) { 
#line 1051
#pragma warning (default: 4127)
#line 1053 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
retval = (static_cast< unsigned long>(umax(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
#line 1054
} else { 
#line 1055
retval = (static_cast< unsigned long>(ullmax(static_cast< unsigned __int64>(a), static_cast< unsigned __int64>(b)))); 
#line 1056
}  
#line 1057
return retval; 
#line 1058
} 
#line 1060
static inline unsigned long max(const unsigned long a, const long b) 
#line 1061
{ 
#line 1062
unsigned long retval; 
#line 1064
#pragma warning (disable: 4127)
#line 1066 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
if (sizeof(unsigned long) == sizeof(unsigned)) { 
#line 1068
#pragma warning (default: 4127)
#line 1070 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
retval = (static_cast< unsigned long>(umax(static_cast< unsigned>(a), static_cast< unsigned>(b)))); 
#line 1071
} else { 
#line 1072
retval = (static_cast< unsigned long>(ullmax(static_cast< unsigned __int64>(a), static_cast< unsigned __int64>(b)))); 
#line 1073
}  
#line 1074
return retval; 
#line 1075
} 
#line 1077
static inline __int64 max(const __int64 a, const __int64 b) 
#line 1078
{ 
#line 1079
return llmax(a, b); 
#line 1080
} 
#line 1082
static inline unsigned __int64 max(const unsigned __int64 a, const unsigned __int64 b) 
#line 1083
{ 
#line 1084
return ullmax(a, b); 
#line 1085
} 
#line 1087
static inline unsigned __int64 max(const __int64 a, const unsigned __int64 b) 
#line 1088
{ 
#line 1089
return ullmax(static_cast< unsigned __int64>(a), b); 
#line 1090
} 
#line 1092
static inline unsigned __int64 max(const unsigned __int64 a, const __int64 b) 
#line 1093
{ 
#line 1094
return ullmax(a, static_cast< unsigned __int64>(b)); 
#line 1095
} 
#line 1097
static inline float max(const float a, const float b) 
#line 1098
{ 
#line 1099
return fmaxf(a, b); 
#line 1100
} 
#line 1102
static inline double max(const double a, const double b) 
#line 1103
{ 
#line 1104
return fmax(a, b); 
#line 1105
} 
#line 1107
static inline double max(const float a, const double b) 
#line 1108
{ 
#line 1109
return fmax(static_cast< double>(a), b); 
#line 1110
} 
#line 1112
static inline double max(const double a, const float b) 
#line 1113
{ 
#line 1114
return fmax(a, static_cast< double>(b)); 
#line 1115
} 
#line 1121
#pragma warning(disable : 4211)
#line 1126 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\math_functions.hpp"
static inline int min(const int a, const int b) 
#line 1127
{ 
#line 1128
return (a < b) ? a : b; 
#line 1129
} 
#line 1131
static inline unsigned umin(const unsigned a, const unsigned b) 
#line 1132
{ 
#line 1133
return (a < b) ? a : b; 
#line 1134
} 
#line 1136
static inline __int64 llmin(const __int64 a, const __int64 b) 
#line 1137
{ 
#line 1138
return (a < b) ? a : b; 
#line 1139
} 
#line 1141
static inline unsigned __int64 ullmin(const unsigned __int64 a, const unsigned __int64 
#line 1142
b) 
#line 1143
{ 
#line 1144
return (a < b) ? a : b; 
#line 1145
} 
#line 1147
static inline int max(const int a, const int b) 
#line 1148
{ 
#line 1149
return (a > b) ? a : b; 
#line 1150
} 
#line 1152
static inline unsigned umax(const unsigned a, const unsigned b) 
#line 1153
{ 
#line 1154
return (a > b) ? a : b; 
#line 1155
} 
#line 1157
static inline __int64 llmax(const __int64 a, const __int64 b) 
#line 1158
{ 
#line 1159
return (a > b) ? a : b; 
#line 1160
} 
#line 1162
static inline unsigned __int64 ullmax(const unsigned __int64 a, const unsigned __int64 
#line 1163
b) 
#line 1164
{ 
#line 1165
return (a > b) ? a : b; 
#line 1166
} 
#line 1169
#pragma warning(default: 4211)
#line 74 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_surface_types.h"
template< class T, int dim = 1> 
#line 75
struct surface : public surfaceReference { 
#line 78
surface() 
#line 79
{ 
#line 80
(channelDesc) = cudaCreateChannelDesc< T> (); 
#line 81
} 
#line 83
surface(::cudaChannelFormatDesc desc) 
#line 84
{ 
#line 85
(channelDesc) = desc; 
#line 86
} 
#line 88 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_surface_types.h"
}; 
#line 90
template< int dim> 
#line 91
struct surface< void, dim>  : public surfaceReference { 
#line 94
surface() 
#line 95
{ 
#line 96
(channelDesc) = cudaCreateChannelDesc< void> (); 
#line 97
} 
#line 99 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_surface_types.h"
}; 
#line 74 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_texture_types.h"
template< class T, int texType = 1, cudaTextureReadMode mode = cudaReadModeElementType> 
#line 75
struct texture : public textureReference { 
#line 78
texture(int norm = 0, ::cudaTextureFilterMode 
#line 79
fMode = cudaFilterModePoint, ::cudaTextureAddressMode 
#line 80
aMode = cudaAddressModeClamp) 
#line 81
{ 
#line 82
(normalized) = norm; 
#line 83
(filterMode) = fMode; 
#line 84
((addressMode)[0]) = aMode; 
#line 85
((addressMode)[1]) = aMode; 
#line 86
((addressMode)[2]) = aMode; 
#line 87
(channelDesc) = cudaCreateChannelDesc< T> (); 
#line 88
(sRGB) = 0; 
#line 89
} 
#line 91
texture(int norm, ::cudaTextureFilterMode 
#line 92
fMode, ::cudaTextureAddressMode 
#line 93
aMode, ::cudaChannelFormatDesc 
#line 94
desc) 
#line 95
{ 
#line 96
(normalized) = norm; 
#line 97
(filterMode) = fMode; 
#line 98
((addressMode)[0]) = aMode; 
#line 99
((addressMode)[1]) = aMode; 
#line 100
((addressMode)[2]) = aMode; 
#line 101
(channelDesc) = desc; 
#line 102
(sRGB) = 0; 
#line 103
} 
#line 105 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\cuda_texture_types.h"
}; 
#line 89 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt/device_functions.h"
extern "C" {
#line 3217 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt/device_functions.h"
}
#line 3225
static __inline int mulhi(int a, int b); 
#line 3227
static __inline unsigned mulhi(unsigned a, unsigned b); 
#line 3229
static __inline unsigned mulhi(int a, unsigned b); 
#line 3231
static __inline unsigned mulhi(unsigned a, int b); 
#line 3233
static __inline __int64 mul64hi(__int64 a, __int64 b); 
#line 3235
static __inline unsigned __int64 mul64hi(unsigned __int64 a, unsigned __int64 b); 
#line 3237
static __inline unsigned __int64 mul64hi(__int64 a, unsigned __int64 b); 
#line 3239
static __inline unsigned __int64 mul64hi(unsigned __int64 a, __int64 b); 
#line 3241
static __inline int float_as_int(float a); 
#line 3243
static __inline float int_as_float(int a); 
#line 3245
static __inline unsigned float_as_uint(float a); 
#line 3247
static __inline float uint_as_float(unsigned a); 
#line 3249
static __inline float saturate(float a); 
#line 3251
static __inline int mul24(int a, int b); 
#line 3253
static __inline unsigned umul24(unsigned a, unsigned b); 
#line 3255
static __inline int float2int(float a, cudaRoundMode mode = cudaRoundZero); 
#line 3257
static __inline unsigned float2uint(float a, cudaRoundMode mode = cudaRoundZero); 
#line 3259
static __inline float int2float(int a, cudaRoundMode mode = cudaRoundNearest); 
#line 3261
static __inline float uint2float(unsigned a, cudaRoundMode mode = cudaRoundNearest); 
#line 90 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\device_functions.hpp"
static __inline int mulhi(const int a, const int b) {int volatile ___ = 1;(void)a;(void)b;::exit(___);}
#if 0
#line 91
{ 
#line 92
return __mulhi(a, b); 
#line 93
} 
#endif
#line 95 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\device_functions.hpp"
static __inline unsigned mulhi(const unsigned a, const unsigned b) {int volatile ___ = 1;(void)a;(void)b;::exit(___);}
#if 0
#line 96
{ 
#line 97
return __umulhi(a, b); 
#line 98
} 
#endif
#line 100 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\device_functions.hpp"
static __inline unsigned mulhi(const int a, const unsigned b) {int volatile ___ = 1;(void)a;(void)b;::exit(___);}
#if 0
#line 101
{ 
#line 102
return __umulhi(static_cast< unsigned>(a), b); 
#line 103
} 
#endif
#line 105 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\device_functions.hpp"
static __inline unsigned mulhi(const unsigned a, const int b) {int volatile ___ = 1;(void)a;(void)b;::exit(___);}
#if 0
#line 106
{ 
#line 107
return __umulhi(a, static_cast< unsigned>(b)); 
#line 108
} 
#endif
#line 110 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\device_functions.hpp"
static __inline __int64 mul64hi(const __int64 a, const __int64 b) {int volatile ___ = 1;(void)a;(void)b;::exit(___);}
#if 0
#line 111
{ 
#line 112
return __mul64hi(a, b); 
#line 113
} 
#endif
#line 115 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\device_functions.hpp"
static __inline unsigned __int64 mul64hi(const unsigned __int64 a, const unsigned __int64 b) {int volatile ___ = 1;(void)a;(void)b;::exit(___);}
#if 0
#line 116
{ 
#line 117
return __umul64hi(a, b); 
#line 118
} 
#endif
#line 120 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\device_functions.hpp"
static __inline unsigned __int64 mul64hi(const __int64 a, const unsigned __int64 b) {int volatile ___ = 1;(void)a;(void)b;::exit(___);}
#if 0
#line 121
{ 
#line 122
return __umul64hi(static_cast< unsigned __int64>(a), b); 
#line 123
} 
#endif
#line 125 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\device_functions.hpp"
static __inline unsigned __int64 mul64hi(const unsigned __int64 a, const __int64 b) {int volatile ___ = 1;(void)a;(void)b;::exit(___);}
#if 0
#line 126
{ 
#line 127
return __umul64hi(a, static_cast< unsigned __int64>(b)); 
#line 128
} 
#endif
#line 130 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\device_functions.hpp"
static __inline int float_as_int(const float a) {int volatile ___ = 1;(void)a;::exit(___);}
#if 0
#line 131
{ 
#line 132
return __float_as_int(a); 
#line 133
} 
#endif
#line 135 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\device_functions.hpp"
static __inline float int_as_float(const int a) {int volatile ___ = 1;(void)a;::exit(___);}
#if 0
#line 136
{ 
#line 137
return __int_as_float(a); 
#line 138
} 
#endif
#line 140 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\device_functions.hpp"
static __inline unsigned float_as_uint(const float a) {int volatile ___ = 1;(void)a;::exit(___);}
#if 0
#line 141
{ 
#line 142
return __float_as_uint(a); 
#line 143
} 
#endif
#line 145 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\device_functions.hpp"
static __inline float uint_as_float(const unsigned a) {int volatile ___ = 1;(void)a;::exit(___);}
#if 0
#line 146
{ 
#line 147
return __uint_as_float(a); 
#line 148
} 
#endif
#line 149 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\device_functions.hpp"
static __inline float saturate(const float a) {int volatile ___ = 1;(void)a;::exit(___);}
#if 0
#line 150
{ 
#line 151
return __saturatef(a); 
#line 152
} 
#endif
#line 154 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\device_functions.hpp"
static __inline int mul24(const int a, const int b) {int volatile ___ = 1;(void)a;(void)b;::exit(___);}
#if 0
#line 155
{ 
#line 156
return __mul24(a, b); 
#line 157
} 
#endif
#line 159 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\device_functions.hpp"
static __inline unsigned umul24(const unsigned a, const unsigned b) {int volatile ___ = 1;(void)a;(void)b;::exit(___);}
#if 0
#line 160
{ 
#line 161
return __umul24(a, b); 
#line 162
} 
#endif
#line 164 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\device_functions.hpp"
static __inline int float2int(const float a, const cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 165
{ 
#line 166
return (mode == (cudaRoundNearest)) ? __float2int_rn(a) : ((mode == (cudaRoundPosInf)) ? __float2int_ru(a) : ((mode == (cudaRoundMinInf)) ? __float2int_rd(a) : __float2int_rz(a))); 
#line 170
} 
#endif
#line 172 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\device_functions.hpp"
static __inline unsigned float2uint(const float a, const cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 173
{ 
#line 174
return (mode == (cudaRoundNearest)) ? __float2uint_rn(a) : ((mode == (cudaRoundPosInf)) ? __float2uint_ru(a) : ((mode == (cudaRoundMinInf)) ? __float2uint_rd(a) : __float2uint_rz(a))); 
#line 178
} 
#endif
#line 180 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\device_functions.hpp"
static __inline float int2float(const int a, const cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 181
{ 
#line 182
return (mode == (cudaRoundZero)) ? __int2float_rz(a) : ((mode == (cudaRoundPosInf)) ? __int2float_ru(a) : ((mode == (cudaRoundMinInf)) ? __int2float_rd(a) : __int2float_rn(a))); 
#line 186
} 
#endif
#line 188 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\device_functions.hpp"
static __inline float uint2float(const unsigned a, const cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 189
{ 
#line 190
return (mode == (cudaRoundZero)) ? __uint2float_rz(a) : ((mode == (cudaRoundPosInf)) ? __uint2float_ru(a) : ((mode == (cudaRoundMinInf)) ? __uint2float_rd(a) : __uint2float_rn(a))); 
#line 194
} 
#endif
#line 106 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\device_atomic_functions.h"
static __inline int atomicAdd(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 106
{ } 
#endif
#line 108 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\device_atomic_functions.h"
static __inline unsigned atomicAdd(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 108
{ } 
#endif
#line 110 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\device_atomic_functions.h"
static __inline int atomicSub(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 110
{ } 
#endif
#line 112 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\device_atomic_functions.h"
static __inline unsigned atomicSub(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 112
{ } 
#endif
#line 114 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\device_atomic_functions.h"
static __inline int atomicExch(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 114
{ } 
#endif
#line 116 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\device_atomic_functions.h"
static __inline unsigned atomicExch(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 116
{ } 
#endif
#line 118 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\device_atomic_functions.h"
static __inline float atomicExch(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 118
{ } 
#endif
#line 120 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\device_atomic_functions.h"
static __inline int atomicMin(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 120
{ } 
#endif
#line 122 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\device_atomic_functions.h"
static __inline unsigned atomicMin(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 122
{ } 
#endif
#line 124 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\device_atomic_functions.h"
static __inline int atomicMax(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 124
{ } 
#endif
#line 126 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\device_atomic_functions.h"
static __inline unsigned atomicMax(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 126
{ } 
#endif
#line 128 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\device_atomic_functions.h"
static __inline unsigned atomicInc(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 128
{ } 
#endif
#line 130 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\device_atomic_functions.h"
static __inline unsigned atomicDec(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 130
{ } 
#endif
#line 132 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\device_atomic_functions.h"
static __inline int atomicAnd(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 132
{ } 
#endif
#line 134 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\device_atomic_functions.h"
static __inline unsigned atomicAnd(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 134
{ } 
#endif
#line 136 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\device_atomic_functions.h"
static __inline int atomicOr(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 136
{ } 
#endif
#line 138 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\device_atomic_functions.h"
static __inline unsigned atomicOr(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 138
{ } 
#endif
#line 140 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\device_atomic_functions.h"
static __inline int atomicXor(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 140
{ } 
#endif
#line 142 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\device_atomic_functions.h"
static __inline unsigned atomicXor(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 142
{ } 
#endif
#line 144 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\device_atomic_functions.h"
static __inline int atomicCAS(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 144
{ } 
#endif
#line 146 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\device_atomic_functions.h"
static __inline unsigned atomicCAS(unsigned *address, unsigned compare, unsigned val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 146
{ } 
#endif
#line 171 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\device_atomic_functions.h"
extern "C" {
#line 180
}
#line 189
static __inline unsigned __int64 atomicAdd(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 189
{ } 
#endif
#line 191 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\device_atomic_functions.h"
static __inline unsigned __int64 atomicExch(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 191
{ } 
#endif
#line 193 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\device_atomic_functions.h"
static __inline unsigned __int64 atomicCAS(unsigned __int64 *address, unsigned __int64 compare, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 193
{ } 
#endif
#line 195 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\device_atomic_functions.h"
__declspec(deprecated("__any() is deprecated in favor of __any_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppr" "ess this warning).")) static __inline bool any(bool cond) {int volatile ___ = 1;(void)cond;::exit(___);}
#if 0
#line 195
{ } 
#endif
#line 197 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\device_atomic_functions.h"
__declspec(deprecated("__all() is deprecated in favor of __all_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppr" "ess this warning).")) static __inline bool all(bool cond) {int volatile ___ = 1;(void)cond;::exit(___);}
#if 0
#line 197
{ } 
#endif
#line 87 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\device_double_functions.h"
extern "C" {
#line 1139
}
#line 1147
static __inline double fma(double a, double b, double c, cudaRoundMode mode); 
#line 1149
static __inline double dmul(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
#line 1151
static __inline double dadd(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
#line 1153
static __inline double dsub(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
#line 1155
static __inline int double2int(double a, cudaRoundMode mode = cudaRoundZero); 
#line 1157
static __inline unsigned double2uint(double a, cudaRoundMode mode = cudaRoundZero); 
#line 1159
static __inline __int64 double2ll(double a, cudaRoundMode mode = cudaRoundZero); 
#line 1161
static __inline unsigned __int64 double2ull(double a, cudaRoundMode mode = cudaRoundZero); 
#line 1163
static __inline double ll2double(__int64 a, cudaRoundMode mode = cudaRoundNearest); 
#line 1165
static __inline double ull2double(unsigned __int64 a, cudaRoundMode mode = cudaRoundNearest); 
#line 1167
static __inline double int2double(int a, cudaRoundMode mode = cudaRoundNearest); 
#line 1169
static __inline double uint2double(unsigned a, cudaRoundMode mode = cudaRoundNearest); 
#line 1171
static __inline double float2double(float a, cudaRoundMode mode = cudaRoundNearest); 
#line 93 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\device_double_functions.hpp"
static __inline double fma(double a, double b, double c, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)b;(void)c;(void)mode;::exit(___);}
#if 0
#line 94
{ 
#line 95
return (mode == (cudaRoundZero)) ? __fma_rz(a, b, c) : ((mode == (cudaRoundPosInf)) ? __fma_ru(a, b, c) : ((mode == (cudaRoundMinInf)) ? __fma_rd(a, b, c) : __fma_rn(a, b, c))); 
#line 99
} 
#endif
#line 101 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\device_double_functions.hpp"
static __inline double dmul(double a, double b, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)b;(void)mode;::exit(___);}
#if 0
#line 102
{ 
#line 103
return (mode == (cudaRoundZero)) ? __dmul_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dmul_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dmul_rd(a, b) : __dmul_rn(a, b))); 
#line 107
} 
#endif
#line 109 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\device_double_functions.hpp"
static __inline double dadd(double a, double b, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)b;(void)mode;::exit(___);}
#if 0
#line 110
{ 
#line 111
return (mode == (cudaRoundZero)) ? __dadd_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dadd_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dadd_rd(a, b) : __dadd_rn(a, b))); 
#line 115
} 
#endif
#line 117 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\device_double_functions.hpp"
static __inline double dsub(double a, double b, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)b;(void)mode;::exit(___);}
#if 0
#line 118
{ 
#line 119
return (mode == (cudaRoundZero)) ? __dsub_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dsub_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dsub_rd(a, b) : __dsub_rn(a, b))); 
#line 123
} 
#endif
#line 125 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\device_double_functions.hpp"
static __inline int double2int(double a, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 126
{ 
#line 127
return (mode == (cudaRoundNearest)) ? __double2int_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2int_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2int_rd(a) : __double2int_rz(a))); 
#line 131
} 
#endif
#line 133 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\device_double_functions.hpp"
static __inline unsigned double2uint(double a, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 134
{ 
#line 135
return (mode == (cudaRoundNearest)) ? __double2uint_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2uint_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2uint_rd(a) : __double2uint_rz(a))); 
#line 139
} 
#endif
#line 141 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\device_double_functions.hpp"
static __inline __int64 double2ll(double a, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 142
{ 
#line 143
return (mode == (cudaRoundNearest)) ? __double2ll_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2ll_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2ll_rd(a) : __double2ll_rz(a))); 
#line 147
} 
#endif
#line 149 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\device_double_functions.hpp"
static __inline unsigned __int64 double2ull(double a, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 150
{ 
#line 151
return (mode == (cudaRoundNearest)) ? __double2ull_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2ull_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2ull_rd(a) : __double2ull_rz(a))); 
#line 155
} 
#endif
#line 157 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\device_double_functions.hpp"
static __inline double ll2double(__int64 a, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 158
{ 
#line 159
return (mode == (cudaRoundZero)) ? __ll2double_rz(a) : ((mode == (cudaRoundPosInf)) ? __ll2double_ru(a) : ((mode == (cudaRoundMinInf)) ? __ll2double_rd(a) : __ll2double_rn(a))); 
#line 163
} 
#endif
#line 165 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\device_double_functions.hpp"
static __inline double ull2double(unsigned __int64 a, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 166
{ 
#line 167
return (mode == (cudaRoundZero)) ? __ull2double_rz(a) : ((mode == (cudaRoundPosInf)) ? __ull2double_ru(a) : ((mode == (cudaRoundMinInf)) ? __ull2double_rd(a) : __ull2double_rn(a))); 
#line 171
} 
#endif
#line 173 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\device_double_functions.hpp"
static __inline double int2double(int a, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 174
{ 
#line 175
return (double)a; 
#line 176
} 
#endif
#line 178 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\device_double_functions.hpp"
static __inline double uint2double(unsigned a, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 179
{ 
#line 180
return (double)a; 
#line 181
} 
#endif
#line 183 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\device_double_functions.hpp"
static __inline double float2double(float a, cudaRoundMode mode) {int volatile ___ = 1;(void)a;(void)mode;::exit(___);}
#if 0
#line 184
{ 
#line 185
return (double)a; 
#line 186
} 
#endif
#line 89 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_20_atomic_functions.h"
static __inline float atomicAdd(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 89
{ } 
#endif
#line 100 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_atomic_functions.h"
static __inline __int64 atomicMin(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 100
{ } 
#endif
#line 102 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_atomic_functions.h"
static __inline __int64 atomicMax(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 102
{ } 
#endif
#line 104 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_atomic_functions.h"
static __inline __int64 atomicAnd(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 104
{ } 
#endif
#line 106 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_atomic_functions.h"
static __inline __int64 atomicOr(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 106
{ } 
#endif
#line 108 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_atomic_functions.h"
static __inline __int64 atomicXor(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 108
{ } 
#endif
#line 110 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_atomic_functions.h"
static __inline unsigned __int64 atomicMin(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 110
{ } 
#endif
#line 112 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_atomic_functions.h"
static __inline unsigned __int64 atomicMax(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 112
{ } 
#endif
#line 114 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_atomic_functions.h"
static __inline unsigned __int64 atomicAnd(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 114
{ } 
#endif
#line 116 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_atomic_functions.h"
static __inline unsigned __int64 atomicOr(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 116
{ } 
#endif
#line 118 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_atomic_functions.h"
static __inline unsigned __int64 atomicXor(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 118
{ } 
#endif
#line 303 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline double atomicAdd(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 303
{ } 
#endif
#line 306 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline int atomicAdd_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 306
{ } 
#endif
#line 309 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline int atomicAdd_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 309
{ } 
#endif
#line 312 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicAdd_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 312
{ } 
#endif
#line 315 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicAdd_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 315
{ } 
#endif
#line 318 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicAdd_block(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 318
{ } 
#endif
#line 321 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicAdd_system(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 321
{ } 
#endif
#line 324 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline float atomicAdd_block(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 324
{ } 
#endif
#line 327 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline float atomicAdd_system(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 327
{ } 
#endif
#line 330 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline double atomicAdd_block(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 330
{ } 
#endif
#line 333 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline double atomicAdd_system(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 333
{ } 
#endif
#line 336 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline int atomicSub_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 336
{ } 
#endif
#line 339 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline int atomicSub_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 339
{ } 
#endif
#line 342 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicSub_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 342
{ } 
#endif
#line 345 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicSub_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 345
{ } 
#endif
#line 348 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline int atomicExch_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 348
{ } 
#endif
#line 351 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline int atomicExch_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 351
{ } 
#endif
#line 354 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicExch_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 354
{ } 
#endif
#line 357 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicExch_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 357
{ } 
#endif
#line 360 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicExch_block(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 360
{ } 
#endif
#line 363 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicExch_system(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 363
{ } 
#endif
#line 366 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline float atomicExch_block(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 366
{ } 
#endif
#line 369 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline float atomicExch_system(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 369
{ } 
#endif
#line 372 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline int atomicMin_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 372
{ } 
#endif
#line 375 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline int atomicMin_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 375
{ } 
#endif
#line 378 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicMin_block(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 378
{ } 
#endif
#line 381 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicMin_system(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 381
{ } 
#endif
#line 384 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicMin_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 384
{ } 
#endif
#line 387 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicMin_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 387
{ } 
#endif
#line 390 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicMin_block(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 390
{ } 
#endif
#line 393 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicMin_system(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 393
{ } 
#endif
#line 396 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline int atomicMax_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 396
{ } 
#endif
#line 399 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline int atomicMax_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 399
{ } 
#endif
#line 402 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicMax_block(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 402
{ } 
#endif
#line 405 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicMax_system(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 405
{ } 
#endif
#line 408 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicMax_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 408
{ } 
#endif
#line 411 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicMax_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 411
{ } 
#endif
#line 414 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicMax_block(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 414
{ } 
#endif
#line 417 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicMax_system(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 417
{ } 
#endif
#line 420 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicInc_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 420
{ } 
#endif
#line 423 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicInc_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 423
{ } 
#endif
#line 426 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicDec_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 426
{ } 
#endif
#line 429 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicDec_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 429
{ } 
#endif
#line 432 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline int atomicCAS_block(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 432
{ } 
#endif
#line 435 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline int atomicCAS_system(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 435
{ } 
#endif
#line 438 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicCAS_block(unsigned *address, unsigned compare, unsigned 
#line 439
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 439
{ } 
#endif
#line 442 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicCAS_system(unsigned *address, unsigned compare, unsigned 
#line 443
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 443
{ } 
#endif
#line 446 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicCAS_block(unsigned __int64 *address, unsigned __int64 
#line 447
compare, unsigned __int64 
#line 448
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 448
{ } 
#endif
#line 451 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicCAS_system(unsigned __int64 *address, unsigned __int64 
#line 452
compare, unsigned __int64 
#line 453
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 453
{ } 
#endif
#line 456 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline int atomicAnd_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 456
{ } 
#endif
#line 459 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline int atomicAnd_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 459
{ } 
#endif
#line 462 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicAnd_block(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 462
{ } 
#endif
#line 465 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicAnd_system(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 465
{ } 
#endif
#line 468 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicAnd_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 468
{ } 
#endif
#line 471 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicAnd_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 471
{ } 
#endif
#line 474 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicAnd_block(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 474
{ } 
#endif
#line 477 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicAnd_system(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 477
{ } 
#endif
#line 480 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline int atomicOr_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 480
{ } 
#endif
#line 483 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline int atomicOr_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 483
{ } 
#endif
#line 486 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicOr_block(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 486
{ } 
#endif
#line 489 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicOr_system(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 489
{ } 
#endif
#line 492 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicOr_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 492
{ } 
#endif
#line 495 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicOr_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 495
{ } 
#endif
#line 498 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicOr_block(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 498
{ } 
#endif
#line 501 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicOr_system(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 501
{ } 
#endif
#line 504 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline int atomicXor_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 504
{ } 
#endif
#line 507 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline int atomicXor_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 507
{ } 
#endif
#line 510 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicXor_block(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 510
{ } 
#endif
#line 513 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline __int64 atomicXor_system(__int64 *address, __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 513
{ } 
#endif
#line 516 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicXor_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 516
{ } 
#endif
#line 519 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned atomicXor_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 519
{ } 
#endif
#line 522 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicXor_block(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 522
{ } 
#endif
#line 525 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_60_atomic_functions.h"
static __inline unsigned __int64 atomicXor_system(unsigned __int64 *address, unsigned __int64 val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
#line 525
{ } 
#endif
#line 90 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_20_intrinsics.h"
extern "C" {
#line 1503
}
#line 1510
__declspec(deprecated("__ballot() is deprecated in favor of __ballot_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to" " suppress this warning).")) static __inline unsigned ballot(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
#line 1510
{ } 
#endif
#line 1512 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_20_intrinsics.h"
static __inline int syncthreads_count(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
#line 1512
{ } 
#endif
#line 1514 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_20_intrinsics.h"
static __inline bool syncthreads_and(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
#line 1514
{ } 
#endif
#line 1516 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_20_intrinsics.h"
static __inline bool syncthreads_or(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
#line 1516
{ } 
#endif
#line 1521 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_20_intrinsics.h"
static __inline unsigned __isGlobal(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 1521
{ } 
#endif
#line 1522 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_20_intrinsics.h"
static __inline unsigned __isShared(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 1522
{ } 
#endif
#line 1523 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_20_intrinsics.h"
static __inline unsigned __isConstant(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 1523
{ } 
#endif
#line 1524 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_20_intrinsics.h"
static __inline unsigned __isLocal(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 1524
{ } 
#endif
#line 1526 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_20_intrinsics.h"
static __inline size_t __cvta_generic_to_global(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 1526
{ } 
#endif
#line 1527 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_20_intrinsics.h"
static __inline size_t __cvta_generic_to_shared(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 1527
{ } 
#endif
#line 1528 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_20_intrinsics.h"
static __inline size_t __cvta_generic_to_constant(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 1528
{ } 
#endif
#line 1529 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_20_intrinsics.h"
static __inline size_t __cvta_generic_to_local(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 1529
{ } 
#endif
#line 1531 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_20_intrinsics.h"
static __inline void *__cvta_global_to_generic(size_t rawbits) {int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
#line 1531
{ } 
#endif
#line 1532 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_20_intrinsics.h"
static __inline void *__cvta_shared_to_generic(size_t rawbits) {int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
#line 1532
{ } 
#endif
#line 1533 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_20_intrinsics.h"
static __inline void *__cvta_constant_to_generic(size_t rawbits) {int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
#line 1533
{ } 
#endif
#line 1534 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_20_intrinsics.h"
static __inline void *__cvta_local_to_generic(size_t rawbits) {int volatile ___ = 1;(void)rawbits;::exit(___);}
#if 0
#line 1534
{ } 
#endif
#line 102 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline unsigned __fns(unsigned mask, unsigned base, int offset) {int volatile ___ = 1;(void)mask;(void)base;(void)offset;::exit(___);}
#if 0
#line 102
{ } 
#endif
#line 103 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline void __barrier_sync(unsigned id) {int volatile ___ = 1;(void)id;::exit(___);}
#if 0
#line 103
{ } 
#endif
#line 104 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline void __barrier_sync_count(unsigned id, unsigned cnt) {int volatile ___ = 1;(void)id;(void)cnt;::exit(___);}
#if 0
#line 104
{ } 
#endif
#line 105 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline void __syncwarp(unsigned mask = 4294967295U) {int volatile ___ = 1;(void)mask;::exit(___);}
#if 0
#line 105
{ } 
#endif
#line 106 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline int __all_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
#line 106
{ } 
#endif
#line 107 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline int __any_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
#line 107
{ } 
#endif
#line 108 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline int __uni_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
#line 108
{ } 
#endif
#line 109 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline unsigned __ballot_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
#line 109
{ } 
#endif
#line 110 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline unsigned __activemask() {int volatile ___ = 1;::exit(___);}
#if 0
#line 110
{ } 
#endif
#line 119 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning).")) static __inline int __shfl(int var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 119
{ } 
#endif
#line 120 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning).")) static __inline unsigned __shfl(unsigned var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 120
{ } 
#endif
#line 121 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning).")) static __inline int __shfl_up(int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 121
{ } 
#endif
#line 122 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning).")) static __inline unsigned __shfl_up(unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 122
{ } 
#endif
#line 123 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning).")) static __inline int __shfl_down(int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 123
{ } 
#endif
#line 124 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning).")) static __inline unsigned __shfl_down(unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 124
{ } 
#endif
#line 125 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning).")) static __inline int __shfl_xor(int var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 125
{ } 
#endif
#line 126 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning).")) static __inline unsigned __shfl_xor(unsigned var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 126
{ } 
#endif
#line 127 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning).")) static __inline float __shfl(float var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 127
{ } 
#endif
#line 128 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning).")) static __inline float __shfl_up(float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 128
{ } 
#endif
#line 129 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning).")) static __inline float __shfl_down(float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 129
{ } 
#endif
#line 130 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning).")) static __inline float __shfl_xor(float var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 130
{ } 
#endif
#line 133 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline int __shfl_sync(unsigned mask, int var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 133
{ } 
#endif
#line 134 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline unsigned __shfl_sync(unsigned mask, unsigned var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 134
{ } 
#endif
#line 135 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline int __shfl_up_sync(unsigned mask, int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 135
{ } 
#endif
#line 136 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline unsigned __shfl_up_sync(unsigned mask, unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 136
{ } 
#endif
#line 137 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline int __shfl_down_sync(unsigned mask, int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 137
{ } 
#endif
#line 138 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline unsigned __shfl_down_sync(unsigned mask, unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 138
{ } 
#endif
#line 139 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline int __shfl_xor_sync(unsigned mask, int var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 139
{ } 
#endif
#line 140 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline unsigned __shfl_xor_sync(unsigned mask, unsigned var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 140
{ } 
#endif
#line 141 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline float __shfl_sync(unsigned mask, float var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 141
{ } 
#endif
#line 142 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline float __shfl_up_sync(unsigned mask, float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 142
{ } 
#endif
#line 143 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline float __shfl_down_sync(unsigned mask, float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 143
{ } 
#endif
#line 144 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline float __shfl_xor_sync(unsigned mask, float var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 144
{ } 
#endif
#line 148 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning).")) static __inline unsigned __int64 __shfl(unsigned __int64 var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 148
{ } 
#endif
#line 149 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning).")) static __inline __int64 __shfl(__int64 var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 149
{ } 
#endif
#line 150 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning).")) static __inline __int64 __shfl_up(__int64 var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 150
{ } 
#endif
#line 151 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning).")) static __inline unsigned __int64 __shfl_up(unsigned __int64 var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 151
{ } 
#endif
#line 152 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning).")) static __inline __int64 __shfl_down(__int64 var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 152
{ } 
#endif
#line 153 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning).")) static __inline unsigned __int64 __shfl_down(unsigned __int64 var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 153
{ } 
#endif
#line 154 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning).")) static __inline __int64 __shfl_xor(__int64 var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 154
{ } 
#endif
#line 155 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning).")) static __inline unsigned __int64 __shfl_xor(unsigned __int64 var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 155
{ } 
#endif
#line 156 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning).")) static __inline double __shfl(double var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 156
{ } 
#endif
#line 157 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning).")) static __inline double __shfl_up(double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 157
{ } 
#endif
#line 158 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning).")) static __inline double __shfl_down(double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 158
{ } 
#endif
#line 159 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning).")) static __inline double __shfl_xor(double var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 159
{ } 
#endif
#line 162 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline __int64 __shfl_sync(unsigned mask, __int64 var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 162
{ } 
#endif
#line 163 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline unsigned __int64 __shfl_sync(unsigned mask, unsigned __int64 var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 163
{ } 
#endif
#line 164 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline __int64 __shfl_up_sync(unsigned mask, __int64 var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 164
{ } 
#endif
#line 165 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline unsigned __int64 __shfl_up_sync(unsigned mask, unsigned __int64 var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 165
{ } 
#endif
#line 166 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline __int64 __shfl_down_sync(unsigned mask, __int64 var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 166
{ } 
#endif
#line 167 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline unsigned __int64 __shfl_down_sync(unsigned mask, unsigned __int64 var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 167
{ } 
#endif
#line 168 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline __int64 __shfl_xor_sync(unsigned mask, __int64 var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 168
{ } 
#endif
#line 169 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline unsigned __int64 __shfl_xor_sync(unsigned mask, unsigned __int64 var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 169
{ } 
#endif
#line 170 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline double __shfl_sync(unsigned mask, double var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 170
{ } 
#endif
#line 171 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline double __shfl_up_sync(unsigned mask, double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 171
{ } 
#endif
#line 172 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline double __shfl_down_sync(unsigned mask, double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 172
{ } 
#endif
#line 173 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline double __shfl_xor_sync(unsigned mask, double var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 173
{ } 
#endif
#line 177 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning).")) static __inline long __shfl(long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 177
{ } 
#endif
#line 178 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning).")) static __inline unsigned long __shfl(unsigned long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 178
{ } 
#endif
#line 179 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning).")) static __inline long __shfl_up(long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 179
{ } 
#endif
#line 180 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning).")) static __inline unsigned long __shfl_up(unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 180
{ } 
#endif
#line 181 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning).")) static __inline long __shfl_down(long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 181
{ } 
#endif
#line 182 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning).")) static __inline unsigned long __shfl_down(unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 182
{ } 
#endif
#line 183 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning).")) static __inline long __shfl_xor(long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 183
{ } 
#endif
#line 184 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
__declspec(deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning).")) static __inline unsigned long __shfl_xor(unsigned long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 184
{ } 
#endif
#line 187 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline long __shfl_sync(unsigned mask, long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 187
{ } 
#endif
#line 188 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline unsigned long __shfl_sync(unsigned mask, unsigned long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
#line 188
{ } 
#endif
#line 189 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline long __shfl_up_sync(unsigned mask, long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 189
{ } 
#endif
#line 190 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline unsigned long __shfl_up_sync(unsigned mask, unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 190
{ } 
#endif
#line 191 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline long __shfl_down_sync(unsigned mask, long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 191
{ } 
#endif
#line 192 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline unsigned long __shfl_down_sync(unsigned mask, unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
#line 192
{ } 
#endif
#line 193 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline long __shfl_xor_sync(unsigned mask, long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 193
{ } 
#endif
#line 194 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_30_intrinsics.h"
static __inline unsigned long __shfl_xor_sync(unsigned mask, unsigned long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
#line 194
{ } 
#endif
#line 87 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline long __ldg(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 87
{ } 
#endif
#line 88 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned long __ldg(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 88
{ } 
#endif
#line 90 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline char __ldg(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 90
{ } 
#endif
#line 91 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline signed char __ldg(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 91
{ } 
#endif
#line 92 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline short __ldg(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 92
{ } 
#endif
#line 93 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline int __ldg(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 93
{ } 
#endif
#line 94 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline __int64 __ldg(const __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 94
{ } 
#endif
#line 95 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline char2 __ldg(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 95
{ } 
#endif
#line 96 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline char4 __ldg(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 96
{ } 
#endif
#line 97 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline short2 __ldg(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 97
{ } 
#endif
#line 98 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline short4 __ldg(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 98
{ } 
#endif
#line 99 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline int2 __ldg(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 99
{ } 
#endif
#line 100 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline int4 __ldg(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 100
{ } 
#endif
#line 101 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline longlong2 __ldg(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 101
{ } 
#endif
#line 103 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned char __ldg(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 103
{ } 
#endif
#line 104 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned short __ldg(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 104
{ } 
#endif
#line 105 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned __ldg(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 105
{ } 
#endif
#line 106 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned __int64 __ldg(const unsigned __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 106
{ } 
#endif
#line 107 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline uchar2 __ldg(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 107
{ } 
#endif
#line 108 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline uchar4 __ldg(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 108
{ } 
#endif
#line 109 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline ushort2 __ldg(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 109
{ } 
#endif
#line 110 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline ushort4 __ldg(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 110
{ } 
#endif
#line 111 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline uint2 __ldg(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 111
{ } 
#endif
#line 112 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline uint4 __ldg(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 112
{ } 
#endif
#line 113 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline ulonglong2 __ldg(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 113
{ } 
#endif
#line 115 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline float __ldg(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 115
{ } 
#endif
#line 116 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline double __ldg(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 116
{ } 
#endif
#line 117 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline float2 __ldg(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 117
{ } 
#endif
#line 118 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline float4 __ldg(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 118
{ } 
#endif
#line 119 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline double2 __ldg(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 119
{ } 
#endif
#line 123 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline long __ldcg(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 123
{ } 
#endif
#line 124 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned long __ldcg(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 124
{ } 
#endif
#line 126 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline char __ldcg(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 126
{ } 
#endif
#line 127 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline signed char __ldcg(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 127
{ } 
#endif
#line 128 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline short __ldcg(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 128
{ } 
#endif
#line 129 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline int __ldcg(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 129
{ } 
#endif
#line 130 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline __int64 __ldcg(const __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 130
{ } 
#endif
#line 131 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline char2 __ldcg(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 131
{ } 
#endif
#line 132 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline char4 __ldcg(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 132
{ } 
#endif
#line 133 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline short2 __ldcg(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 133
{ } 
#endif
#line 134 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline short4 __ldcg(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 134
{ } 
#endif
#line 135 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline int2 __ldcg(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 135
{ } 
#endif
#line 136 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline int4 __ldcg(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 136
{ } 
#endif
#line 137 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline longlong2 __ldcg(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 137
{ } 
#endif
#line 139 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned char __ldcg(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 139
{ } 
#endif
#line 140 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned short __ldcg(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 140
{ } 
#endif
#line 141 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned __ldcg(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 141
{ } 
#endif
#line 142 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned __int64 __ldcg(const unsigned __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 142
{ } 
#endif
#line 143 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline uchar2 __ldcg(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 143
{ } 
#endif
#line 144 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline uchar4 __ldcg(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 144
{ } 
#endif
#line 145 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline ushort2 __ldcg(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 145
{ } 
#endif
#line 146 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline ushort4 __ldcg(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 146
{ } 
#endif
#line 147 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline uint2 __ldcg(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 147
{ } 
#endif
#line 148 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline uint4 __ldcg(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 148
{ } 
#endif
#line 149 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline ulonglong2 __ldcg(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 149
{ } 
#endif
#line 151 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline float __ldcg(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 151
{ } 
#endif
#line 152 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline double __ldcg(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 152
{ } 
#endif
#line 153 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline float2 __ldcg(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 153
{ } 
#endif
#line 154 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline float4 __ldcg(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 154
{ } 
#endif
#line 155 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline double2 __ldcg(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 155
{ } 
#endif
#line 159 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline long __ldca(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 159
{ } 
#endif
#line 160 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned long __ldca(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 160
{ } 
#endif
#line 162 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline char __ldca(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 162
{ } 
#endif
#line 163 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline signed char __ldca(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 163
{ } 
#endif
#line 164 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline short __ldca(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 164
{ } 
#endif
#line 165 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline int __ldca(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 165
{ } 
#endif
#line 166 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline __int64 __ldca(const __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 166
{ } 
#endif
#line 167 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline char2 __ldca(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 167
{ } 
#endif
#line 168 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline char4 __ldca(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 168
{ } 
#endif
#line 169 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline short2 __ldca(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 169
{ } 
#endif
#line 170 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline short4 __ldca(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 170
{ } 
#endif
#line 171 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline int2 __ldca(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 171
{ } 
#endif
#line 172 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline int4 __ldca(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 172
{ } 
#endif
#line 173 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline longlong2 __ldca(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 173
{ } 
#endif
#line 175 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned char __ldca(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 175
{ } 
#endif
#line 176 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned short __ldca(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 176
{ } 
#endif
#line 177 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned __ldca(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 177
{ } 
#endif
#line 178 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned __int64 __ldca(const unsigned __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 178
{ } 
#endif
#line 179 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline uchar2 __ldca(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 179
{ } 
#endif
#line 180 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline uchar4 __ldca(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 180
{ } 
#endif
#line 181 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline ushort2 __ldca(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 181
{ } 
#endif
#line 182 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline ushort4 __ldca(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 182
{ } 
#endif
#line 183 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline uint2 __ldca(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 183
{ } 
#endif
#line 184 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline uint4 __ldca(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 184
{ } 
#endif
#line 185 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline ulonglong2 __ldca(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 185
{ } 
#endif
#line 187 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline float __ldca(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 187
{ } 
#endif
#line 188 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline double __ldca(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 188
{ } 
#endif
#line 189 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline float2 __ldca(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 189
{ } 
#endif
#line 190 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline float4 __ldca(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 190
{ } 
#endif
#line 191 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline double2 __ldca(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 191
{ } 
#endif
#line 195 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline long __ldcs(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 195
{ } 
#endif
#line 196 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned long __ldcs(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 196
{ } 
#endif
#line 198 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline char __ldcs(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 198
{ } 
#endif
#line 199 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline signed char __ldcs(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 199
{ } 
#endif
#line 200 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline short __ldcs(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 200
{ } 
#endif
#line 201 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline int __ldcs(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 201
{ } 
#endif
#line 202 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline __int64 __ldcs(const __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 202
{ } 
#endif
#line 203 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline char2 __ldcs(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 203
{ } 
#endif
#line 204 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline char4 __ldcs(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 204
{ } 
#endif
#line 205 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline short2 __ldcs(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 205
{ } 
#endif
#line 206 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline short4 __ldcs(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 206
{ } 
#endif
#line 207 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline int2 __ldcs(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 207
{ } 
#endif
#line 208 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline int4 __ldcs(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 208
{ } 
#endif
#line 209 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline longlong2 __ldcs(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 209
{ } 
#endif
#line 211 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned char __ldcs(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 211
{ } 
#endif
#line 212 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned short __ldcs(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 212
{ } 
#endif
#line 213 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned __ldcs(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 213
{ } 
#endif
#line 214 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned __int64 __ldcs(const unsigned __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 214
{ } 
#endif
#line 215 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline uchar2 __ldcs(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 215
{ } 
#endif
#line 216 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline uchar4 __ldcs(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 216
{ } 
#endif
#line 217 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline ushort2 __ldcs(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 217
{ } 
#endif
#line 218 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline ushort4 __ldcs(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 218
{ } 
#endif
#line 219 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline uint2 __ldcs(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 219
{ } 
#endif
#line 220 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline uint4 __ldcs(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 220
{ } 
#endif
#line 221 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline ulonglong2 __ldcs(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 221
{ } 
#endif
#line 223 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline float __ldcs(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 223
{ } 
#endif
#line 224 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline double __ldcs(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 224
{ } 
#endif
#line 225 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline float2 __ldcs(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 225
{ } 
#endif
#line 226 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline float4 __ldcs(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 226
{ } 
#endif
#line 227 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline double2 __ldcs(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 227
{ } 
#endif
#line 231 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline long __ldlu(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 231
{ } 
#endif
#line 232 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned long __ldlu(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 232
{ } 
#endif
#line 234 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline char __ldlu(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 234
{ } 
#endif
#line 235 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline signed char __ldlu(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 235
{ } 
#endif
#line 236 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline short __ldlu(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 236
{ } 
#endif
#line 237 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline int __ldlu(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 237
{ } 
#endif
#line 238 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline __int64 __ldlu(const __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 238
{ } 
#endif
#line 239 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline char2 __ldlu(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 239
{ } 
#endif
#line 240 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline char4 __ldlu(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 240
{ } 
#endif
#line 241 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline short2 __ldlu(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 241
{ } 
#endif
#line 242 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline short4 __ldlu(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 242
{ } 
#endif
#line 243 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline int2 __ldlu(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 243
{ } 
#endif
#line 244 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline int4 __ldlu(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 244
{ } 
#endif
#line 245 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline longlong2 __ldlu(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 245
{ } 
#endif
#line 247 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned char __ldlu(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 247
{ } 
#endif
#line 248 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned short __ldlu(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 248
{ } 
#endif
#line 249 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned __ldlu(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 249
{ } 
#endif
#line 250 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned __int64 __ldlu(const unsigned __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 250
{ } 
#endif
#line 251 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline uchar2 __ldlu(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 251
{ } 
#endif
#line 252 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline uchar4 __ldlu(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 252
{ } 
#endif
#line 253 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline ushort2 __ldlu(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 253
{ } 
#endif
#line 254 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline ushort4 __ldlu(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 254
{ } 
#endif
#line 255 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline uint2 __ldlu(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 255
{ } 
#endif
#line 256 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline uint4 __ldlu(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 256
{ } 
#endif
#line 257 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline ulonglong2 __ldlu(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 257
{ } 
#endif
#line 259 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline float __ldlu(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 259
{ } 
#endif
#line 260 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline double __ldlu(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 260
{ } 
#endif
#line 261 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline float2 __ldlu(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 261
{ } 
#endif
#line 262 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline float4 __ldlu(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 262
{ } 
#endif
#line 263 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline double2 __ldlu(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 263
{ } 
#endif
#line 267 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline long __ldcv(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 267
{ } 
#endif
#line 268 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned long __ldcv(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 268
{ } 
#endif
#line 270 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline char __ldcv(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 270
{ } 
#endif
#line 271 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline signed char __ldcv(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 271
{ } 
#endif
#line 272 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline short __ldcv(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 272
{ } 
#endif
#line 273 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline int __ldcv(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 273
{ } 
#endif
#line 274 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline __int64 __ldcv(const __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 274
{ } 
#endif
#line 275 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline char2 __ldcv(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 275
{ } 
#endif
#line 276 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline char4 __ldcv(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 276
{ } 
#endif
#line 277 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline short2 __ldcv(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 277
{ } 
#endif
#line 278 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline short4 __ldcv(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 278
{ } 
#endif
#line 279 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline int2 __ldcv(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 279
{ } 
#endif
#line 280 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline int4 __ldcv(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 280
{ } 
#endif
#line 281 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline longlong2 __ldcv(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 281
{ } 
#endif
#line 283 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned char __ldcv(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 283
{ } 
#endif
#line 284 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned short __ldcv(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 284
{ } 
#endif
#line 285 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned __ldcv(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 285
{ } 
#endif
#line 286 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned __int64 __ldcv(const unsigned __int64 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 286
{ } 
#endif
#line 287 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline uchar2 __ldcv(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 287
{ } 
#endif
#line 288 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline uchar4 __ldcv(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 288
{ } 
#endif
#line 289 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline ushort2 __ldcv(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 289
{ } 
#endif
#line 290 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline ushort4 __ldcv(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 290
{ } 
#endif
#line 291 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline uint2 __ldcv(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 291
{ } 
#endif
#line 292 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline uint4 __ldcv(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 292
{ } 
#endif
#line 293 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline ulonglong2 __ldcv(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 293
{ } 
#endif
#line 295 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline float __ldcv(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 295
{ } 
#endif
#line 296 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline double __ldcv(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 296
{ } 
#endif
#line 297 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline float2 __ldcv(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 297
{ } 
#endif
#line 298 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline float4 __ldcv(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 298
{ } 
#endif
#line 299 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline double2 __ldcv(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
#line 299
{ } 
#endif
#line 303 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwb(long *ptr, long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 303
{ } 
#endif
#line 304 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwb(unsigned long *ptr, unsigned long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 304
{ } 
#endif
#line 306 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwb(char *ptr, char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 306
{ } 
#endif
#line 307 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwb(signed char *ptr, signed char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 307
{ } 
#endif
#line 308 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwb(short *ptr, short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 308
{ } 
#endif
#line 309 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwb(int *ptr, int value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 309
{ } 
#endif
#line 310 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwb(__int64 *ptr, __int64 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 310
{ } 
#endif
#line 311 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwb(char2 *ptr, char2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 311
{ } 
#endif
#line 312 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwb(char4 *ptr, char4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 312
{ } 
#endif
#line 313 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwb(short2 *ptr, short2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 313
{ } 
#endif
#line 314 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwb(short4 *ptr, short4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 314
{ } 
#endif
#line 315 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwb(int2 *ptr, int2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 315
{ } 
#endif
#line 316 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwb(int4 *ptr, int4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 316
{ } 
#endif
#line 317 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwb(longlong2 *ptr, longlong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 317
{ } 
#endif
#line 319 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwb(unsigned char *ptr, unsigned char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 319
{ } 
#endif
#line 320 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwb(unsigned short *ptr, unsigned short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 320
{ } 
#endif
#line 321 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwb(unsigned *ptr, unsigned value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 321
{ } 
#endif
#line 322 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwb(unsigned __int64 *ptr, unsigned __int64 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 322
{ } 
#endif
#line 323 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwb(uchar2 *ptr, uchar2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 323
{ } 
#endif
#line 324 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwb(uchar4 *ptr, uchar4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 324
{ } 
#endif
#line 325 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwb(ushort2 *ptr, ushort2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 325
{ } 
#endif
#line 326 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwb(ushort4 *ptr, ushort4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 326
{ } 
#endif
#line 327 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwb(uint2 *ptr, uint2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 327
{ } 
#endif
#line 328 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwb(uint4 *ptr, uint4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 328
{ } 
#endif
#line 329 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwb(ulonglong2 *ptr, ulonglong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 329
{ } 
#endif
#line 331 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwb(float *ptr, float value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 331
{ } 
#endif
#line 332 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwb(double *ptr, double value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 332
{ } 
#endif
#line 333 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwb(float2 *ptr, float2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 333
{ } 
#endif
#line 334 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwb(float4 *ptr, float4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 334
{ } 
#endif
#line 335 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwb(double2 *ptr, double2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 335
{ } 
#endif
#line 339 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcg(long *ptr, long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 339
{ } 
#endif
#line 340 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcg(unsigned long *ptr, unsigned long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 340
{ } 
#endif
#line 342 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcg(char *ptr, char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 342
{ } 
#endif
#line 343 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcg(signed char *ptr, signed char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 343
{ } 
#endif
#line 344 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcg(short *ptr, short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 344
{ } 
#endif
#line 345 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcg(int *ptr, int value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 345
{ } 
#endif
#line 346 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcg(__int64 *ptr, __int64 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 346
{ } 
#endif
#line 347 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcg(char2 *ptr, char2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 347
{ } 
#endif
#line 348 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcg(char4 *ptr, char4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 348
{ } 
#endif
#line 349 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcg(short2 *ptr, short2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 349
{ } 
#endif
#line 350 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcg(short4 *ptr, short4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 350
{ } 
#endif
#line 351 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcg(int2 *ptr, int2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 351
{ } 
#endif
#line 352 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcg(int4 *ptr, int4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 352
{ } 
#endif
#line 353 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcg(longlong2 *ptr, longlong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 353
{ } 
#endif
#line 355 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcg(unsigned char *ptr, unsigned char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 355
{ } 
#endif
#line 356 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcg(unsigned short *ptr, unsigned short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 356
{ } 
#endif
#line 357 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcg(unsigned *ptr, unsigned value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 357
{ } 
#endif
#line 358 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcg(unsigned __int64 *ptr, unsigned __int64 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 358
{ } 
#endif
#line 359 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcg(uchar2 *ptr, uchar2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 359
{ } 
#endif
#line 360 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcg(uchar4 *ptr, uchar4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 360
{ } 
#endif
#line 361 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcg(ushort2 *ptr, ushort2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 361
{ } 
#endif
#line 362 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcg(ushort4 *ptr, ushort4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 362
{ } 
#endif
#line 363 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcg(uint2 *ptr, uint2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 363
{ } 
#endif
#line 364 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcg(uint4 *ptr, uint4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 364
{ } 
#endif
#line 365 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcg(ulonglong2 *ptr, ulonglong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 365
{ } 
#endif
#line 367 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcg(float *ptr, float value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 367
{ } 
#endif
#line 368 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcg(double *ptr, double value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 368
{ } 
#endif
#line 369 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcg(float2 *ptr, float2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 369
{ } 
#endif
#line 370 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcg(float4 *ptr, float4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 370
{ } 
#endif
#line 371 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcg(double2 *ptr, double2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 371
{ } 
#endif
#line 375 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcs(long *ptr, long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 375
{ } 
#endif
#line 376 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcs(unsigned long *ptr, unsigned long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 376
{ } 
#endif
#line 378 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcs(char *ptr, char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 378
{ } 
#endif
#line 379 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcs(signed char *ptr, signed char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 379
{ } 
#endif
#line 380 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcs(short *ptr, short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 380
{ } 
#endif
#line 381 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcs(int *ptr, int value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 381
{ } 
#endif
#line 382 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcs(__int64 *ptr, __int64 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 382
{ } 
#endif
#line 383 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcs(char2 *ptr, char2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 383
{ } 
#endif
#line 384 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcs(char4 *ptr, char4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 384
{ } 
#endif
#line 385 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcs(short2 *ptr, short2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 385
{ } 
#endif
#line 386 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcs(short4 *ptr, short4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 386
{ } 
#endif
#line 387 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcs(int2 *ptr, int2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 387
{ } 
#endif
#line 388 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcs(int4 *ptr, int4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 388
{ } 
#endif
#line 389 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcs(longlong2 *ptr, longlong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 389
{ } 
#endif
#line 391 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcs(unsigned char *ptr, unsigned char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 391
{ } 
#endif
#line 392 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcs(unsigned short *ptr, unsigned short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 392
{ } 
#endif
#line 393 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcs(unsigned *ptr, unsigned value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 393
{ } 
#endif
#line 394 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcs(unsigned __int64 *ptr, unsigned __int64 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 394
{ } 
#endif
#line 395 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcs(uchar2 *ptr, uchar2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 395
{ } 
#endif
#line 396 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcs(uchar4 *ptr, uchar4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 396
{ } 
#endif
#line 397 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcs(ushort2 *ptr, ushort2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 397
{ } 
#endif
#line 398 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcs(ushort4 *ptr, ushort4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 398
{ } 
#endif
#line 399 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcs(uint2 *ptr, uint2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 399
{ } 
#endif
#line 400 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcs(uint4 *ptr, uint4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 400
{ } 
#endif
#line 401 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcs(ulonglong2 *ptr, ulonglong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 401
{ } 
#endif
#line 403 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcs(float *ptr, float value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 403
{ } 
#endif
#line 404 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcs(double *ptr, double value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 404
{ } 
#endif
#line 405 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcs(float2 *ptr, float2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 405
{ } 
#endif
#line 406 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcs(float4 *ptr, float4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 406
{ } 
#endif
#line 407 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stcs(double2 *ptr, double2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 407
{ } 
#endif
#line 411 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwt(long *ptr, long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 411
{ } 
#endif
#line 412 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwt(unsigned long *ptr, unsigned long value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 412
{ } 
#endif
#line 414 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwt(char *ptr, char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 414
{ } 
#endif
#line 415 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwt(signed char *ptr, signed char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 415
{ } 
#endif
#line 416 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwt(short *ptr, short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 416
{ } 
#endif
#line 417 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwt(int *ptr, int value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 417
{ } 
#endif
#line 418 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwt(__int64 *ptr, __int64 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 418
{ } 
#endif
#line 419 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwt(char2 *ptr, char2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 419
{ } 
#endif
#line 420 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwt(char4 *ptr, char4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 420
{ } 
#endif
#line 421 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwt(short2 *ptr, short2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 421
{ } 
#endif
#line 422 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwt(short4 *ptr, short4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 422
{ } 
#endif
#line 423 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwt(int2 *ptr, int2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 423
{ } 
#endif
#line 424 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwt(int4 *ptr, int4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 424
{ } 
#endif
#line 425 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwt(longlong2 *ptr, longlong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 425
{ } 
#endif
#line 427 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwt(unsigned char *ptr, unsigned char value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 427
{ } 
#endif
#line 428 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwt(unsigned short *ptr, unsigned short value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 428
{ } 
#endif
#line 429 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwt(unsigned *ptr, unsigned value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 429
{ } 
#endif
#line 430 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwt(unsigned __int64 *ptr, unsigned __int64 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 430
{ } 
#endif
#line 431 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwt(uchar2 *ptr, uchar2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 431
{ } 
#endif
#line 432 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwt(uchar4 *ptr, uchar4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 432
{ } 
#endif
#line 433 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwt(ushort2 *ptr, ushort2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 433
{ } 
#endif
#line 434 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwt(ushort4 *ptr, ushort4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 434
{ } 
#endif
#line 435 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwt(uint2 *ptr, uint2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 435
{ } 
#endif
#line 436 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwt(uint4 *ptr, uint4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 436
{ } 
#endif
#line 437 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwt(ulonglong2 *ptr, ulonglong2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 437
{ } 
#endif
#line 439 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwt(float *ptr, float value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 439
{ } 
#endif
#line 440 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwt(double *ptr, double value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 440
{ } 
#endif
#line 441 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwt(float2 *ptr, float2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 441
{ } 
#endif
#line 442 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwt(float4 *ptr, float4 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 442
{ } 
#endif
#line 443 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline void __stwt(double2 *ptr, double2 value) {int volatile ___ = 1;(void)ptr;(void)value;::exit(___);}
#if 0
#line 443
{ } 
#endif
#line 460 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned __funnelshift_l(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
#line 460
{ } 
#endif
#line 472 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned __funnelshift_lc(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
#line 472
{ } 
#endif
#line 485 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned __funnelshift_r(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
#line 485
{ } 
#endif
#line 497 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_32_intrinsics.h"
static __inline unsigned __funnelshift_rc(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
#line 497
{ } 
#endif
#line 89 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_61_intrinsics.h"
static __inline int __dp2a_lo(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 89
{ } 
#endif
#line 90 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_61_intrinsics.h"
static __inline unsigned __dp2a_lo(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 90
{ } 
#endif
#line 92 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_61_intrinsics.h"
static __inline int __dp2a_lo(short2 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 92
{ } 
#endif
#line 93 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_61_intrinsics.h"
static __inline unsigned __dp2a_lo(ushort2 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 93
{ } 
#endif
#line 95 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_61_intrinsics.h"
static __inline int __dp2a_hi(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 95
{ } 
#endif
#line 96 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_61_intrinsics.h"
static __inline unsigned __dp2a_hi(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 96
{ } 
#endif
#line 98 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_61_intrinsics.h"
static __inline int __dp2a_hi(short2 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 98
{ } 
#endif
#line 99 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_61_intrinsics.h"
static __inline unsigned __dp2a_hi(ushort2 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 99
{ } 
#endif
#line 106 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_61_intrinsics.h"
static __inline int __dp4a(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 106
{ } 
#endif
#line 107 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_61_intrinsics.h"
static __inline unsigned __dp4a(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 107
{ } 
#endif
#line 109 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_61_intrinsics.h"
static __inline int __dp4a(char4 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 109
{ } 
#endif
#line 110 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\sm_61_intrinsics.h"
static __inline unsigned __dp4a(uchar4 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
#line 110
{ } 
#endif
#line 93 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_any_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 93
{ } 
#endif
#line 94 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_any_sync(unsigned mask, int value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 94
{ } 
#endif
#line 95 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_any_sync(unsigned mask, unsigned long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 95
{ } 
#endif
#line 96 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_any_sync(unsigned mask, long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 96
{ } 
#endif
#line 97 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_any_sync(unsigned mask, unsigned __int64 value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 97
{ } 
#endif
#line 98 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_any_sync(unsigned mask, __int64 value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 98
{ } 
#endif
#line 99 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_any_sync(unsigned mask, float value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 99
{ } 
#endif
#line 100 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_any_sync(unsigned mask, double value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 100
{ } 
#endif
#line 102 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_all_sync(unsigned mask, unsigned value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
#line 102
{ } 
#endif
#line 103 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_all_sync(unsigned mask, int value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
#line 103
{ } 
#endif
#line 104 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_all_sync(unsigned mask, unsigned long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
#line 104
{ } 
#endif
#line 105 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_all_sync(unsigned mask, long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
#line 105
{ } 
#endif
#line 106 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_all_sync(unsigned mask, unsigned __int64 value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
#line 106
{ } 
#endif
#line 107 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_all_sync(unsigned mask, __int64 value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
#line 107
{ } 
#endif
#line 108 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_all_sync(unsigned mask, float value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
#line 108
{ } 
#endif
#line 109 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\sm_70_rt.h"
static __inline unsigned __match_all_sync(unsigned mask, double value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
#line 109
{ } 
#endif
#line 111 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\sm_70_rt.h"
static __inline void __nanosleep(unsigned ns) {int volatile ___ = 1;(void)ns;::exit(___);}
#if 0
#line 111
{ } 
#endif
#line 113 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\sm_70_rt.h"
static __inline unsigned short atomicCAS(unsigned short *address, unsigned short compare, unsigned short val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
#line 113
{ } 
#endif
#line 93 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\sm_80_rt.h"
static __inline unsigned __reduce_add_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 93
{ } 
#endif
#line 94 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\sm_80_rt.h"
static __inline unsigned __reduce_min_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 94
{ } 
#endif
#line 95 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\sm_80_rt.h"
static __inline unsigned __reduce_max_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 95
{ } 
#endif
#line 97 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\sm_80_rt.h"
static __inline int __reduce_add_sync(unsigned mask, int value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 97
{ } 
#endif
#line 98 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\sm_80_rt.h"
static __inline int __reduce_min_sync(unsigned mask, int value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 98
{ } 
#endif
#line 99 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\sm_80_rt.h"
static __inline int __reduce_max_sync(unsigned mask, int value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 99
{ } 
#endif
#line 101 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\sm_80_rt.h"
static __inline unsigned __reduce_and_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 101
{ } 
#endif
#line 102 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\sm_80_rt.h"
static __inline unsigned __reduce_or_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 102
{ } 
#endif
#line 103 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt\\sm_80_rt.h"
static __inline unsigned __reduce_xor_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
#line 103
{ } 
#endif
#line 114 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 115
surf1Dread(T *res, ::surface< void, 1>  surf, int x, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)s;(void)mode;::exit(___);}
#if 0
#line 116
{ 
#line 120
} 
#endif
#line 122 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline T 
#line 123
surf1Dread(::surface< void, 1>  surf, int x, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surf;(void)x;(void)mode;::exit(___);}
#if 0
#line 124
{ 
#line 130
} 
#endif
#line 132 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 133
surf1Dread(T *res, ::surface< void, 1>  surf, int x, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)mode;::exit(___);}
#if 0
#line 134
{ 
#line 138
} 
#endif
#line 141 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 142
surf2Dread(T *res, ::surface< void, 2>  surf, int x, int y, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)s;(void)mode;::exit(___);}
#if 0
#line 143
{ 
#line 147
} 
#endif
#line 149 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline T 
#line 150
surf2Dread(::surface< void, 2>  surf, int x, int y, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)mode;::exit(___);}
#if 0
#line 151
{ 
#line 157
} 
#endif
#line 159 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 160
surf2Dread(T *res, ::surface< void, 2>  surf, int x, int y, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)mode;::exit(___);}
#if 0
#line 161
{ 
#line 165
} 
#endif
#line 168 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 169
surf3Dread(T *res, ::surface< void, 3>  surf, int x, int y, int z, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)z;(void)s;(void)mode;::exit(___);}
#if 0
#line 170
{ 
#line 174
} 
#endif
#line 176 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline T 
#line 177
surf3Dread(::surface< void, 3>  surf, int x, int y, int z, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)z;(void)mode;::exit(___);}
#if 0
#line 178
{ 
#line 184
} 
#endif
#line 186 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 187
surf3Dread(T *res, ::surface< void, 3>  surf, int x, int y, int z, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)z;(void)mode;::exit(___);}
#if 0
#line 188
{ 
#line 192
} 
#endif
#line 196 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 197
surf1DLayeredread(T *res, ::surface< void, 241>  surf, int x, int layer, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)layer;(void)s;(void)mode;::exit(___);}
#if 0
#line 198
{ 
#line 202
} 
#endif
#line 204 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline T 
#line 205
surf1DLayeredread(::surface< void, 241>  surf, int x, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surf;(void)x;(void)layer;(void)mode;::exit(___);}
#if 0
#line 206
{ 
#line 212
} 
#endif
#line 215 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 216
surf1DLayeredread(T *res, ::surface< void, 241>  surf, int x, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)layer;(void)mode;::exit(___);}
#if 0
#line 217
{ 
#line 221
} 
#endif
#line 224 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 225
surf2DLayeredread(T *res, ::surface< void, 242>  surf, int x, int y, int layer, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layer;(void)s;(void)mode;::exit(___);}
#if 0
#line 226
{ 
#line 230
} 
#endif
#line 232 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline T 
#line 233
surf2DLayeredread(::surface< void, 242>  surf, int x, int y, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)layer;(void)mode;::exit(___);}
#if 0
#line 234
{ 
#line 240
} 
#endif
#line 243 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 244
surf2DLayeredread(T *res, ::surface< void, 242>  surf, int x, int y, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layer;(void)mode;::exit(___);}
#if 0
#line 245
{ 
#line 249
} 
#endif
#line 252 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 253
surfCubemapread(T *res, ::surface< void, 12>  surf, int x, int y, int face, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)face;(void)s;(void)mode;::exit(___);}
#if 0
#line 254
{ 
#line 258
} 
#endif
#line 260 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline T 
#line 261
surfCubemapread(::surface< void, 12>  surf, int x, int y, int face, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)face;(void)mode;::exit(___);}
#if 0
#line 262
{ 
#line 269
} 
#endif
#line 271 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 272
surfCubemapread(T *res, ::surface< void, 12>  surf, int x, int y, int face, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)face;(void)mode;::exit(___);}
#if 0
#line 273
{ 
#line 277
} 
#endif
#line 280 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 281
surfCubemapLayeredread(T *res, ::surface< void, 252>  surf, int x, int y, int layerFace, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layerFace;(void)s;(void)mode;::exit(___);}
#if 0
#line 282
{ 
#line 286
} 
#endif
#line 288 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline T 
#line 289
surfCubemapLayeredread(::surface< void, 252>  surf, int x, int y, int layerFace, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)layerFace;(void)mode;::exit(___);}
#if 0
#line 290
{ 
#line 296
} 
#endif
#line 298 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 299
surfCubemapLayeredread(T *res, ::surface< void, 252>  surf, int x, int y, int layerFace, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layerFace;(void)mode;::exit(___);}
#if 0
#line 300
{ 
#line 304
} 
#endif
#line 307 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 308
surf1Dwrite(T val, ::surface< void, 1>  surf, int x, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)s;(void)mode;::exit(___);}
#if 0
#line 309
{ 
#line 313
} 
#endif
#line 315 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 316
surf1Dwrite(T val, ::surface< void, 1>  surf, int x, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)mode;::exit(___);}
#if 0
#line 317
{ 
#line 321
} 
#endif
#line 325 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 326
surf2Dwrite(T val, ::surface< void, 2>  surf, int x, int y, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)s;(void)mode;::exit(___);}
#if 0
#line 327
{ 
#line 331
} 
#endif
#line 333 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 334
surf2Dwrite(T val, ::surface< void, 2>  surf, int x, int y, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)mode;::exit(___);}
#if 0
#line 335
{ 
#line 339
} 
#endif
#line 342 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 343
surf3Dwrite(T val, ::surface< void, 3>  surf, int x, int y, int z, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)z;(void)s;(void)mode;::exit(___);}
#if 0
#line 344
{ 
#line 348
} 
#endif
#line 350 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 351
surf3Dwrite(T val, ::surface< void, 3>  surf, int x, int y, int z, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)z;(void)mode;::exit(___);}
#if 0
#line 352
{ 
#line 356
} 
#endif
#line 359 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 360
surf1DLayeredwrite(T val, ::surface< void, 241>  surf, int x, int layer, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)layer;(void)s;(void)mode;::exit(___);}
#if 0
#line 361
{ 
#line 365
} 
#endif
#line 367 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 368
surf1DLayeredwrite(T val, ::surface< void, 241>  surf, int x, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)layer;(void)mode;::exit(___);}
#if 0
#line 369
{ 
#line 373
} 
#endif
#line 376 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 377
surf2DLayeredwrite(T val, ::surface< void, 242>  surf, int x, int y, int layer, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layer;(void)s;(void)mode;::exit(___);}
#if 0
#line 378
{ 
#line 382
} 
#endif
#line 384 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 385
surf2DLayeredwrite(T val, ::surface< void, 242>  surf, int x, int y, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layer;(void)mode;::exit(___);}
#if 0
#line 386
{ 
#line 390
} 
#endif
#line 393 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 394
surfCubemapwrite(T val, ::surface< void, 12>  surf, int x, int y, int face, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)face;(void)s;(void)mode;::exit(___);}
#if 0
#line 395
{ 
#line 399
} 
#endif
#line 401 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 402
surfCubemapwrite(T val, ::surface< void, 12>  surf, int x, int y, int face, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)face;(void)mode;::exit(___);}
#if 0
#line 403
{ 
#line 407
} 
#endif
#line 411 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 412
surfCubemapLayeredwrite(T val, ::surface< void, 252>  surf, int x, int y, int layerFace, int s, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layerFace;(void)s;(void)mode;::exit(___);}
#if 0
#line 413
{ 
#line 417
} 
#endif
#line 419 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_functions.h"
template< class T> static __forceinline void 
#line 420
surfCubemapLayeredwrite(T val, ::surface< void, 252>  surf, int x, int y, int layerFace, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layerFace;(void)mode;::exit(___);}
#if 0
#line 421
{ 
#line 425
} 
#endif
#line 66 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> 
#line 67
struct __nv_tex_rmet_ret { }; 
#line 69
template<> struct __nv_tex_rmet_ret< char>  { typedef char type; }; 
#line 70
template<> struct __nv_tex_rmet_ret< signed char>  { typedef signed char type; }; 
#line 71
template<> struct __nv_tex_rmet_ret< unsigned char>  { typedef unsigned char type; }; 
#line 72
template<> struct __nv_tex_rmet_ret< char1>  { typedef char1 type; }; 
#line 73
template<> struct __nv_tex_rmet_ret< uchar1>  { typedef uchar1 type; }; 
#line 74
template<> struct __nv_tex_rmet_ret< char2>  { typedef char2 type; }; 
#line 75
template<> struct __nv_tex_rmet_ret< uchar2>  { typedef uchar2 type; }; 
#line 76
template<> struct __nv_tex_rmet_ret< char4>  { typedef char4 type; }; 
#line 77
template<> struct __nv_tex_rmet_ret< uchar4>  { typedef uchar4 type; }; 
#line 79
template<> struct __nv_tex_rmet_ret< short>  { typedef short type; }; 
#line 80
template<> struct __nv_tex_rmet_ret< unsigned short>  { typedef unsigned short type; }; 
#line 81
template<> struct __nv_tex_rmet_ret< short1>  { typedef short1 type; }; 
#line 82
template<> struct __nv_tex_rmet_ret< ushort1>  { typedef ushort1 type; }; 
#line 83
template<> struct __nv_tex_rmet_ret< short2>  { typedef short2 type; }; 
#line 84
template<> struct __nv_tex_rmet_ret< ushort2>  { typedef ushort2 type; }; 
#line 85
template<> struct __nv_tex_rmet_ret< short4>  { typedef short4 type; }; 
#line 86
template<> struct __nv_tex_rmet_ret< ushort4>  { typedef ushort4 type; }; 
#line 88
template<> struct __nv_tex_rmet_ret< int>  { typedef int type; }; 
#line 89
template<> struct __nv_tex_rmet_ret< unsigned>  { typedef unsigned type; }; 
#line 90
template<> struct __nv_tex_rmet_ret< int1>  { typedef int1 type; }; 
#line 91
template<> struct __nv_tex_rmet_ret< uint1>  { typedef uint1 type; }; 
#line 92
template<> struct __nv_tex_rmet_ret< int2>  { typedef int2 type; }; 
#line 93
template<> struct __nv_tex_rmet_ret< uint2>  { typedef uint2 type; }; 
#line 94
template<> struct __nv_tex_rmet_ret< int4>  { typedef int4 type; }; 
#line 95
template<> struct __nv_tex_rmet_ret< uint4>  { typedef uint4 type; }; 
#line 98
template<> struct __nv_tex_rmet_ret< long>  { typedef long type; }; 
#line 99
template<> struct __nv_tex_rmet_ret< unsigned long>  { typedef unsigned long type; }; 
#line 100
template<> struct __nv_tex_rmet_ret< long1>  { typedef long1 type; }; 
#line 101
template<> struct __nv_tex_rmet_ret< ulong1>  { typedef ulong1 type; }; 
#line 102
template<> struct __nv_tex_rmet_ret< long2>  { typedef long2 type; }; 
#line 103
template<> struct __nv_tex_rmet_ret< ulong2>  { typedef ulong2 type; }; 
#line 104
template<> struct __nv_tex_rmet_ret< long4>  { typedef long4 type; }; 
#line 105
template<> struct __nv_tex_rmet_ret< ulong4>  { typedef ulong4 type; }; 
#line 107 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template<> struct __nv_tex_rmet_ret< float>  { typedef float type; }; 
#line 108
template<> struct __nv_tex_rmet_ret< float1>  { typedef float1 type; }; 
#line 109
template<> struct __nv_tex_rmet_ret< float2>  { typedef float2 type; }; 
#line 110
template<> struct __nv_tex_rmet_ret< float4>  { typedef float4 type; }; 
#line 113
template< class T> struct __nv_tex_rmet_cast { typedef T *type; }; 
#line 115
template<> struct __nv_tex_rmet_cast< long>  { typedef int *type; }; 
#line 116
template<> struct __nv_tex_rmet_cast< unsigned long>  { typedef unsigned *type; }; 
#line 117
template<> struct __nv_tex_rmet_cast< long1>  { typedef int1 *type; }; 
#line 118
template<> struct __nv_tex_rmet_cast< ulong1>  { typedef uint1 *type; }; 
#line 119
template<> struct __nv_tex_rmet_cast< long2>  { typedef int2 *type; }; 
#line 120
template<> struct __nv_tex_rmet_cast< ulong2>  { typedef uint2 *type; }; 
#line 121
template<> struct __nv_tex_rmet_cast< long4>  { typedef int4 *type; }; 
#line 122
template<> struct __nv_tex_rmet_cast< ulong4>  { typedef uint4 *type; }; 
#line 125 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 126
tex1Dfetch(texture< T, 1, cudaReadModeElementType>  t, int x) {int volatile ___ = 1;(void)t;(void)x;::exit(___);}
#if 0
#line 127
{ 
#line 133
} 
#endif
#line 135 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> 
#line 136
struct __nv_tex_rmnf_ret { }; 
#line 138
template<> struct __nv_tex_rmnf_ret< char>  { typedef float type; }; 
#line 139
template<> struct __nv_tex_rmnf_ret< signed char>  { typedef float type; }; 
#line 140
template<> struct __nv_tex_rmnf_ret< unsigned char>  { typedef float type; }; 
#line 141
template<> struct __nv_tex_rmnf_ret< short>  { typedef float type; }; 
#line 142
template<> struct __nv_tex_rmnf_ret< unsigned short>  { typedef float type; }; 
#line 143
template<> struct __nv_tex_rmnf_ret< char1>  { typedef float1 type; }; 
#line 144
template<> struct __nv_tex_rmnf_ret< uchar1>  { typedef float1 type; }; 
#line 145
template<> struct __nv_tex_rmnf_ret< short1>  { typedef float1 type; }; 
#line 146
template<> struct __nv_tex_rmnf_ret< ushort1>  { typedef float1 type; }; 
#line 147
template<> struct __nv_tex_rmnf_ret< char2>  { typedef float2 type; }; 
#line 148
template<> struct __nv_tex_rmnf_ret< uchar2>  { typedef float2 type; }; 
#line 149
template<> struct __nv_tex_rmnf_ret< short2>  { typedef float2 type; }; 
#line 150
template<> struct __nv_tex_rmnf_ret< ushort2>  { typedef float2 type; }; 
#line 151
template<> struct __nv_tex_rmnf_ret< char4>  { typedef float4 type; }; 
#line 152
template<> struct __nv_tex_rmnf_ret< uchar4>  { typedef float4 type; }; 
#line 153
template<> struct __nv_tex_rmnf_ret< short4>  { typedef float4 type; }; 
#line 154
template<> struct __nv_tex_rmnf_ret< ushort4>  { typedef float4 type; }; 
#line 156
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 157
tex1Dfetch(texture< T, 1, cudaReadModeNormalizedFloat>  t, int x) {int volatile ___ = 1;(void)t;(void)x;::exit(___);}
#if 0
#line 158
{ 
#line 165
} 
#endif
#line 168 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 169
tex1D(texture< T, 1, cudaReadModeElementType>  t, float x) {int volatile ___ = 1;(void)t;(void)x;::exit(___);}
#if 0
#line 170
{ 
#line 176
} 
#endif
#line 178 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 179
tex1D(texture< T, 1, cudaReadModeNormalizedFloat>  t, float x) {int volatile ___ = 1;(void)t;(void)x;::exit(___);}
#if 0
#line 180
{ 
#line 187
} 
#endif
#line 191 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 192
tex2D(texture< T, 2, cudaReadModeElementType>  t, float x, float y) {int volatile ___ = 1;(void)t;(void)x;(void)y;::exit(___);}
#if 0
#line 193
{ 
#line 200
} 
#endif
#line 202 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 203
tex2D(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y) {int volatile ___ = 1;(void)t;(void)x;(void)y;::exit(___);}
#if 0
#line 204
{ 
#line 211
} 
#endif
#line 215 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 216
tex1DLayered(texture< T, 241, cudaReadModeElementType>  t, float x, int layer) {int volatile ___ = 1;(void)t;(void)x;(void)layer;::exit(___);}
#if 0
#line 217
{ 
#line 223
} 
#endif
#line 225 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 226
tex1DLayered(texture< T, 241, cudaReadModeNormalizedFloat>  t, float x, int layer) {int volatile ___ = 1;(void)t;(void)x;(void)layer;::exit(___);}
#if 0
#line 227
{ 
#line 234
} 
#endif
#line 238 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 239
tex2DLayered(texture< T, 242, cudaReadModeElementType>  t, float x, float y, int layer) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;::exit(___);}
#if 0
#line 240
{ 
#line 246
} 
#endif
#line 248 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 249
tex2DLayered(texture< T, 242, cudaReadModeNormalizedFloat>  t, float x, float y, int layer) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;::exit(___);}
#if 0
#line 250
{ 
#line 257
} 
#endif
#line 260 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 261
tex3D(texture< T, 3, cudaReadModeElementType>  t, float x, float y, float z) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;::exit(___);}
#if 0
#line 262
{ 
#line 268
} 
#endif
#line 270 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 271
tex3D(texture< T, 3, cudaReadModeNormalizedFloat>  t, float x, float y, float z) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;::exit(___);}
#if 0
#line 272
{ 
#line 279
} 
#endif
#line 282 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 283
texCubemap(texture< T, 12, cudaReadModeElementType>  t, float x, float y, float z) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;::exit(___);}
#if 0
#line 284
{ 
#line 290
} 
#endif
#line 292 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 293
texCubemap(texture< T, 12, cudaReadModeNormalizedFloat>  t, float x, float y, float z) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;::exit(___);}
#if 0
#line 294
{ 
#line 301
} 
#endif
#line 304 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> 
#line 305
struct __nv_tex2dgather_ret { }; 
#line 306
template<> struct __nv_tex2dgather_ret< char>  { typedef char4 type; }; 
#line 307
template<> struct __nv_tex2dgather_ret< signed char>  { typedef char4 type; }; 
#line 308
template<> struct __nv_tex2dgather_ret< char1>  { typedef char4 type; }; 
#line 309
template<> struct __nv_tex2dgather_ret< char2>  { typedef char4 type; }; 
#line 310
template<> struct __nv_tex2dgather_ret< char3>  { typedef char4 type; }; 
#line 311
template<> struct __nv_tex2dgather_ret< char4>  { typedef char4 type; }; 
#line 312
template<> struct __nv_tex2dgather_ret< unsigned char>  { typedef uchar4 type; }; 
#line 313
template<> struct __nv_tex2dgather_ret< uchar1>  { typedef uchar4 type; }; 
#line 314
template<> struct __nv_tex2dgather_ret< uchar2>  { typedef uchar4 type; }; 
#line 315
template<> struct __nv_tex2dgather_ret< uchar3>  { typedef uchar4 type; }; 
#line 316
template<> struct __nv_tex2dgather_ret< uchar4>  { typedef uchar4 type; }; 
#line 318
template<> struct __nv_tex2dgather_ret< short>  { typedef short4 type; }; 
#line 319
template<> struct __nv_tex2dgather_ret< short1>  { typedef short4 type; }; 
#line 320
template<> struct __nv_tex2dgather_ret< short2>  { typedef short4 type; }; 
#line 321
template<> struct __nv_tex2dgather_ret< short3>  { typedef short4 type; }; 
#line 322
template<> struct __nv_tex2dgather_ret< short4>  { typedef short4 type; }; 
#line 323
template<> struct __nv_tex2dgather_ret< unsigned short>  { typedef ushort4 type; }; 
#line 324
template<> struct __nv_tex2dgather_ret< ushort1>  { typedef ushort4 type; }; 
#line 325
template<> struct __nv_tex2dgather_ret< ushort2>  { typedef ushort4 type; }; 
#line 326
template<> struct __nv_tex2dgather_ret< ushort3>  { typedef ushort4 type; }; 
#line 327
template<> struct __nv_tex2dgather_ret< ushort4>  { typedef ushort4 type; }; 
#line 329
template<> struct __nv_tex2dgather_ret< int>  { typedef int4 type; }; 
#line 330
template<> struct __nv_tex2dgather_ret< int1>  { typedef int4 type; }; 
#line 331
template<> struct __nv_tex2dgather_ret< int2>  { typedef int4 type; }; 
#line 332
template<> struct __nv_tex2dgather_ret< int3>  { typedef int4 type; }; 
#line 333
template<> struct __nv_tex2dgather_ret< int4>  { typedef int4 type; }; 
#line 334
template<> struct __nv_tex2dgather_ret< unsigned>  { typedef uint4 type; }; 
#line 335
template<> struct __nv_tex2dgather_ret< uint1>  { typedef uint4 type; }; 
#line 336
template<> struct __nv_tex2dgather_ret< uint2>  { typedef uint4 type; }; 
#line 337
template<> struct __nv_tex2dgather_ret< uint3>  { typedef uint4 type; }; 
#line 338
template<> struct __nv_tex2dgather_ret< uint4>  { typedef uint4 type; }; 
#line 340
template<> struct __nv_tex2dgather_ret< float>  { typedef float4 type; }; 
#line 341
template<> struct __nv_tex2dgather_ret< float1>  { typedef float4 type; }; 
#line 342
template<> struct __nv_tex2dgather_ret< float2>  { typedef float4 type; }; 
#line 343
template<> struct __nv_tex2dgather_ret< float3>  { typedef float4 type; }; 
#line 344
template<> struct __nv_tex2dgather_ret< float4>  { typedef float4 type; }; 
#line 346
template< class T> static __forceinline typename __nv_tex2dgather_ret< T> ::type 
#line 347
tex2Dgather(texture< T, 2, cudaReadModeElementType>  t, float x, float y, int comp = 0) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)comp;::exit(___);}
#if 0
#line 348
{ 
#line 355
} 
#endif
#line 358 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> struct __nv_tex2dgather_rmnf_ret { }; 
#line 359
template<> struct __nv_tex2dgather_rmnf_ret< char>  { typedef float4 type; }; 
#line 360
template<> struct __nv_tex2dgather_rmnf_ret< signed char>  { typedef float4 type; }; 
#line 361
template<> struct __nv_tex2dgather_rmnf_ret< unsigned char>  { typedef float4 type; }; 
#line 362
template<> struct __nv_tex2dgather_rmnf_ret< char1>  { typedef float4 type; }; 
#line 363
template<> struct __nv_tex2dgather_rmnf_ret< uchar1>  { typedef float4 type; }; 
#line 364
template<> struct __nv_tex2dgather_rmnf_ret< char2>  { typedef float4 type; }; 
#line 365
template<> struct __nv_tex2dgather_rmnf_ret< uchar2>  { typedef float4 type; }; 
#line 366
template<> struct __nv_tex2dgather_rmnf_ret< char3>  { typedef float4 type; }; 
#line 367
template<> struct __nv_tex2dgather_rmnf_ret< uchar3>  { typedef float4 type; }; 
#line 368
template<> struct __nv_tex2dgather_rmnf_ret< char4>  { typedef float4 type; }; 
#line 369
template<> struct __nv_tex2dgather_rmnf_ret< uchar4>  { typedef float4 type; }; 
#line 370
template<> struct __nv_tex2dgather_rmnf_ret< signed short>  { typedef float4 type; }; 
#line 371
template<> struct __nv_tex2dgather_rmnf_ret< unsigned short>  { typedef float4 type; }; 
#line 372
template<> struct __nv_tex2dgather_rmnf_ret< short1>  { typedef float4 type; }; 
#line 373
template<> struct __nv_tex2dgather_rmnf_ret< ushort1>  { typedef float4 type; }; 
#line 374
template<> struct __nv_tex2dgather_rmnf_ret< short2>  { typedef float4 type; }; 
#line 375
template<> struct __nv_tex2dgather_rmnf_ret< ushort2>  { typedef float4 type; }; 
#line 376
template<> struct __nv_tex2dgather_rmnf_ret< short3>  { typedef float4 type; }; 
#line 377
template<> struct __nv_tex2dgather_rmnf_ret< ushort3>  { typedef float4 type; }; 
#line 378
template<> struct __nv_tex2dgather_rmnf_ret< short4>  { typedef float4 type; }; 
#line 379
template<> struct __nv_tex2dgather_rmnf_ret< ushort4>  { typedef float4 type; }; 
#line 381
template< class T> static __forceinline typename __nv_tex2dgather_rmnf_ret< T> ::type 
#line 382
tex2Dgather(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y, int comp = 0) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)comp;::exit(___);}
#if 0
#line 383
{ 
#line 390
} 
#endif
#line 394 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 395
tex1DLod(texture< T, 1, cudaReadModeElementType>  t, float x, float level) {int volatile ___ = 1;(void)t;(void)x;(void)level;::exit(___);}
#if 0
#line 396
{ 
#line 402
} 
#endif
#line 404 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 405
tex1DLod(texture< T, 1, cudaReadModeNormalizedFloat>  t, float x, float level) {int volatile ___ = 1;(void)t;(void)x;(void)level;::exit(___);}
#if 0
#line 406
{ 
#line 413
} 
#endif
#line 416 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 417
tex2DLod(texture< T, 2, cudaReadModeElementType>  t, float x, float y, float level) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)level;::exit(___);}
#if 0
#line 418
{ 
#line 424
} 
#endif
#line 426 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 427
tex2DLod(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y, float level) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)level;::exit(___);}
#if 0
#line 428
{ 
#line 435
} 
#endif
#line 438 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 439
tex1DLayeredLod(texture< T, 241, cudaReadModeElementType>  t, float x, int layer, float level) {int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)level;::exit(___);}
#if 0
#line 440
{ 
#line 446
} 
#endif
#line 448 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 449
tex1DLayeredLod(texture< T, 241, cudaReadModeNormalizedFloat>  t, float x, int layer, float level) {int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)level;::exit(___);}
#if 0
#line 450
{ 
#line 457
} 
#endif
#line 460 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 461
tex2DLayeredLod(texture< T, 242, cudaReadModeElementType>  t, float x, float y, int layer, float level) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)level;::exit(___);}
#if 0
#line 462
{ 
#line 468
} 
#endif
#line 470 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 471
tex2DLayeredLod(texture< T, 242, cudaReadModeNormalizedFloat>  t, float x, float y, int layer, float level) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)level;::exit(___);}
#if 0
#line 472
{ 
#line 479
} 
#endif
#line 482 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 483
tex3DLod(texture< T, 3, cudaReadModeElementType>  t, float x, float y, float z, float level) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;::exit(___);}
#if 0
#line 484
{ 
#line 490
} 
#endif
#line 492 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 493
tex3DLod(texture< T, 3, cudaReadModeNormalizedFloat>  t, float x, float y, float z, float level) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;::exit(___);}
#if 0
#line 494
{ 
#line 501
} 
#endif
#line 504 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 505
texCubemapLod(texture< T, 12, cudaReadModeElementType>  t, float x, float y, float z, float level) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;::exit(___);}
#if 0
#line 506
{ 
#line 512
} 
#endif
#line 514 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 515
texCubemapLod(texture< T, 12, cudaReadModeNormalizedFloat>  t, float x, float y, float z, float level) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;::exit(___);}
#if 0
#line 516
{ 
#line 523
} 
#endif
#line 527 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 528
texCubemapLayered(texture< T, 252, cudaReadModeElementType>  t, float x, float y, float z, int layer) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;::exit(___);}
#if 0
#line 529
{ 
#line 535
} 
#endif
#line 537 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 538
texCubemapLayered(texture< T, 252, cudaReadModeNormalizedFloat>  t, float x, float y, float z, int layer) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;::exit(___);}
#if 0
#line 539
{ 
#line 546
} 
#endif
#line 550 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 551
texCubemapLayeredLod(texture< T, 252, cudaReadModeElementType>  t, float x, float y, float z, int layer, float level) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)level;::exit(___);}
#if 0
#line 552
{ 
#line 558
} 
#endif
#line 560 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 561
texCubemapLayeredLod(texture< T, 252, cudaReadModeNormalizedFloat>  t, float x, float y, float z, int layer, float level) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)level;::exit(___);}
#if 0
#line 562
{ 
#line 569
} 
#endif
#line 573 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 574
texCubemapGrad(texture< T, 12, cudaReadModeElementType>  t, float x, float y, float z, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 575
{ 
#line 581
} 
#endif
#line 583 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 584
texCubemapGrad(texture< T, 12, cudaReadModeNormalizedFloat>  t, float x, float y, float z, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 585
{ 
#line 592
} 
#endif
#line 596 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 597
texCubemapLayeredGrad(texture< T, 252, cudaReadModeElementType>  t, float x, float y, float z, int layer, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 598
{ 
#line 604
} 
#endif
#line 606 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 607
texCubemapLayeredGrad(texture< T, 252, cudaReadModeNormalizedFloat>  t, float x, float y, float z, int layer, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 608
{ 
#line 615
} 
#endif
#line 619 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 620
tex1DGrad(texture< T, 1, cudaReadModeElementType>  t, float x, float dPdx, float dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 621
{ 
#line 627
} 
#endif
#line 629 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 630
tex1DGrad(texture< T, 1, cudaReadModeNormalizedFloat>  t, float x, float dPdx, float dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 631
{ 
#line 638
} 
#endif
#line 642 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 643
tex2DGrad(texture< T, 2, cudaReadModeElementType>  t, float x, float y, ::float2 dPdx, ::float2 dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 644
{ 
#line 650
} 
#endif
#line 652 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 653
tex2DGrad(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y, ::float2 dPdx, ::float2 dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 654
{ 
#line 661
} 
#endif
#line 664 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 665
tex1DLayeredGrad(texture< T, 241, cudaReadModeElementType>  t, float x, int layer, float dPdx, float dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 666
{ 
#line 672
} 
#endif
#line 674 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 675
tex1DLayeredGrad(texture< T, 241, cudaReadModeNormalizedFloat>  t, float x, int layer, float dPdx, float dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 676
{ 
#line 683
} 
#endif
#line 686 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 687
tex2DLayeredGrad(texture< T, 242, cudaReadModeElementType>  t, float x, float y, int layer, ::float2 dPdx, ::float2 dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 688
{ 
#line 694
} 
#endif
#line 696 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 697
tex2DLayeredGrad(texture< T, 242, cudaReadModeNormalizedFloat>  t, float x, float y, int layer, ::float2 dPdx, ::float2 dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 698
{ 
#line 705
} 
#endif
#line 708 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmet_ret< T> ::type 
#line 709
tex3DGrad(texture< T, 3, cudaReadModeElementType>  t, float x, float y, float z, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 710
{ 
#line 716
} 
#endif
#line 718 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_fetch_functions.h"
template< class T> static __forceinline typename __nv_tex_rmnf_ret< T> ::type 
#line 719
tex3DGrad(texture< T, 3, cudaReadModeNormalizedFloat>  t, float x, float y, float z, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 720
{ 
#line 727
} 
#endif
#line 64 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> struct __nv_itex_trait { }; 
#line 65
template<> struct __nv_itex_trait< char>  { typedef void type; }; 
#line 66
template<> struct __nv_itex_trait< signed char>  { typedef void type; }; 
#line 67
template<> struct __nv_itex_trait< char1>  { typedef void type; }; 
#line 68
template<> struct __nv_itex_trait< char2>  { typedef void type; }; 
#line 69
template<> struct __nv_itex_trait< char4>  { typedef void type; }; 
#line 70
template<> struct __nv_itex_trait< unsigned char>  { typedef void type; }; 
#line 71
template<> struct __nv_itex_trait< uchar1>  { typedef void type; }; 
#line 72
template<> struct __nv_itex_trait< uchar2>  { typedef void type; }; 
#line 73
template<> struct __nv_itex_trait< uchar4>  { typedef void type; }; 
#line 74
template<> struct __nv_itex_trait< short>  { typedef void type; }; 
#line 75
template<> struct __nv_itex_trait< short1>  { typedef void type; }; 
#line 76
template<> struct __nv_itex_trait< short2>  { typedef void type; }; 
#line 77
template<> struct __nv_itex_trait< short4>  { typedef void type; }; 
#line 78
template<> struct __nv_itex_trait< unsigned short>  { typedef void type; }; 
#line 79
template<> struct __nv_itex_trait< ushort1>  { typedef void type; }; 
#line 80
template<> struct __nv_itex_trait< ushort2>  { typedef void type; }; 
#line 81
template<> struct __nv_itex_trait< ushort4>  { typedef void type; }; 
#line 82
template<> struct __nv_itex_trait< int>  { typedef void type; }; 
#line 83
template<> struct __nv_itex_trait< int1>  { typedef void type; }; 
#line 84
template<> struct __nv_itex_trait< int2>  { typedef void type; }; 
#line 85
template<> struct __nv_itex_trait< int4>  { typedef void type; }; 
#line 86
template<> struct __nv_itex_trait< unsigned>  { typedef void type; }; 
#line 87
template<> struct __nv_itex_trait< uint1>  { typedef void type; }; 
#line 88
template<> struct __nv_itex_trait< uint2>  { typedef void type; }; 
#line 89
template<> struct __nv_itex_trait< uint4>  { typedef void type; }; 
#line 91
template<> struct __nv_itex_trait< long>  { typedef void type; }; 
#line 92
template<> struct __nv_itex_trait< long1>  { typedef void type; }; 
#line 93
template<> struct __nv_itex_trait< long2>  { typedef void type; }; 
#line 94
template<> struct __nv_itex_trait< long4>  { typedef void type; }; 
#line 95
template<> struct __nv_itex_trait< unsigned long>  { typedef void type; }; 
#line 96
template<> struct __nv_itex_trait< ulong1>  { typedef void type; }; 
#line 97
template<> struct __nv_itex_trait< ulong2>  { typedef void type; }; 
#line 98
template<> struct __nv_itex_trait< ulong4>  { typedef void type; }; 
#line 100 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template<> struct __nv_itex_trait< float>  { typedef void type; }; 
#line 101
template<> struct __nv_itex_trait< float1>  { typedef void type; }; 
#line 102
template<> struct __nv_itex_trait< float2>  { typedef void type; }; 
#line 103
template<> struct __nv_itex_trait< float4>  { typedef void type; }; 
#line 107
template< class T> static typename __nv_itex_trait< T> ::type 
#line 108
tex1Dfetch(T *ptr, ::cudaTextureObject_t obj, int x) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;::exit(___);}
#if 0
#line 109
{ 
#line 113
} 
#endif
#line 115 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 116
tex1Dfetch(::cudaTextureObject_t texObject, int x) {int volatile ___ = 1;(void)texObject;(void)x;::exit(___);}
#if 0
#line 117
{ 
#line 123
} 
#endif
#line 125 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 126
tex1D(T *ptr, ::cudaTextureObject_t obj, float x) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;::exit(___);}
#if 0
#line 127
{ 
#line 131
} 
#endif
#line 134 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 135
tex1D(::cudaTextureObject_t texObject, float x) {int volatile ___ = 1;(void)texObject;(void)x;::exit(___);}
#if 0
#line 136
{ 
#line 142
} 
#endif
#line 145 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 146
tex2D(T *ptr, ::cudaTextureObject_t obj, float x, float y) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;::exit(___);}
#if 0
#line 147
{ 
#line 151
} 
#endif
#line 153 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 154
tex2D(::cudaTextureObject_t texObject, float x, float y) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;::exit(___);}
#if 0
#line 155
{ 
#line 161
} 
#endif
#line 164 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 165
tex2D(T *ptr, ::cudaTextureObject_t obj, float x, float y, bool *
#line 166
isResident) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)isResident;::exit(___);}
#if 0
#line 167
{ 
#line 173
} 
#endif
#line 175 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 176
tex2D(::cudaTextureObject_t texObject, float x, float y, bool *isResident) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)isResident;::exit(___);}
#if 0
#line 177
{ 
#line 183
} 
#endif
#line 188 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 189
tex3D(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;::exit(___);}
#if 0
#line 190
{ 
#line 194
} 
#endif
#line 196 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 197
tex3D(::cudaTextureObject_t texObject, float x, float y, float z) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;::exit(___);}
#if 0
#line 198
{ 
#line 204
} 
#endif
#line 207 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 208
tex3D(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, bool *
#line 209
isResident) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)isResident;::exit(___);}
#if 0
#line 210
{ 
#line 216
} 
#endif
#line 218 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 219
tex3D(::cudaTextureObject_t texObject, float x, float y, float z, bool *isResident) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)isResident;::exit(___);}
#if 0
#line 220
{ 
#line 226
} 
#endif
#line 230 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 231
tex1DLayered(T *ptr, ::cudaTextureObject_t obj, float x, int layer) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;::exit(___);}
#if 0
#line 232
{ 
#line 236
} 
#endif
#line 238 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 239
tex1DLayered(::cudaTextureObject_t texObject, float x, int layer) {int volatile ___ = 1;(void)texObject;(void)x;(void)layer;::exit(___);}
#if 0
#line 240
{ 
#line 246
} 
#endif
#line 248 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 249
tex2DLayered(T *ptr, ::cudaTextureObject_t obj, float x, float y, int layer) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;::exit(___);}
#if 0
#line 250
{ 
#line 254
} 
#endif
#line 256 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 257
tex2DLayered(::cudaTextureObject_t texObject, float x, float y, int layer) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;::exit(___);}
#if 0
#line 258
{ 
#line 264
} 
#endif
#line 267 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 268
tex2DLayered(T *ptr, ::cudaTextureObject_t obj, float x, float y, int layer, bool *isResident) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)isResident;::exit(___);}
#if 0
#line 269
{ 
#line 275
} 
#endif
#line 277 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 278
tex2DLayered(::cudaTextureObject_t texObject, float x, float y, int layer, bool *isResident) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)isResident;::exit(___);}
#if 0
#line 279
{ 
#line 285
} 
#endif
#line 289 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 290
texCubemap(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;::exit(___);}
#if 0
#line 291
{ 
#line 295
} 
#endif
#line 298 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 299
texCubemap(::cudaTextureObject_t texObject, float x, float y, float z) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;::exit(___);}
#if 0
#line 300
{ 
#line 306
} 
#endif
#line 309 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 310
texCubemapLayered(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, int layer) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;::exit(___);}
#if 0
#line 311
{ 
#line 315
} 
#endif
#line 317 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 318
texCubemapLayered(::cudaTextureObject_t texObject, float x, float y, float z, int layer) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;::exit(___);}
#if 0
#line 319
{ 
#line 325
} 
#endif
#line 327 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 328
tex2Dgather(T *ptr, ::cudaTextureObject_t obj, float x, float y, int comp = 0) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)comp;::exit(___);}
#if 0
#line 329
{ 
#line 333
} 
#endif
#line 335 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 336
tex2Dgather(::cudaTextureObject_t to, float x, float y, int comp = 0) {int volatile ___ = 1;(void)to;(void)x;(void)y;(void)comp;::exit(___);}
#if 0
#line 337
{ 
#line 343
} 
#endif
#line 346 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 347
tex2Dgather(T *ptr, ::cudaTextureObject_t obj, float x, float y, bool *isResident, int comp = 0) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)isResident;(void)comp;::exit(___);}
#if 0
#line 348
{ 
#line 354
} 
#endif
#line 356 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 357
tex2Dgather(::cudaTextureObject_t to, float x, float y, bool *isResident, int comp = 0) {int volatile ___ = 1;(void)to;(void)x;(void)y;(void)isResident;(void)comp;::exit(___);}
#if 0
#line 358
{ 
#line 364
} 
#endif
#line 368 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 369
tex1DLod(T *ptr, ::cudaTextureObject_t obj, float x, float level) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)level;::exit(___);}
#if 0
#line 370
{ 
#line 374
} 
#endif
#line 376 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 377
tex1DLod(::cudaTextureObject_t texObject, float x, float level) {int volatile ___ = 1;(void)texObject;(void)x;(void)level;::exit(___);}
#if 0
#line 378
{ 
#line 384
} 
#endif
#line 387 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 388
tex2DLod(T *ptr, ::cudaTextureObject_t obj, float x, float y, float level) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)level;::exit(___);}
#if 0
#line 389
{ 
#line 393
} 
#endif
#line 395 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 396
tex2DLod(::cudaTextureObject_t texObject, float x, float y, float level) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)level;::exit(___);}
#if 0
#line 397
{ 
#line 403
} 
#endif
#line 407 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 408
tex2DLod(T *ptr, ::cudaTextureObject_t obj, float x, float y, float level, bool *isResident) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)level;(void)isResident;::exit(___);}
#if 0
#line 409
{ 
#line 415
} 
#endif
#line 417 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 418
tex2DLod(::cudaTextureObject_t texObject, float x, float y, float level, bool *isResident) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)level;(void)isResident;::exit(___);}
#if 0
#line 419
{ 
#line 425
} 
#endif
#line 430 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 431
tex3DLod(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, float level) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)level;::exit(___);}
#if 0
#line 432
{ 
#line 436
} 
#endif
#line 438 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 439
tex3DLod(::cudaTextureObject_t texObject, float x, float y, float z, float level) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;::exit(___);}
#if 0
#line 440
{ 
#line 446
} 
#endif
#line 449 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 450
tex3DLod(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, float level, bool *isResident) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)level;(void)isResident;::exit(___);}
#if 0
#line 451
{ 
#line 457
} 
#endif
#line 459 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 460
tex3DLod(::cudaTextureObject_t texObject, float x, float y, float z, float level, bool *isResident) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;(void)isResident;::exit(___);}
#if 0
#line 461
{ 
#line 467
} 
#endif
#line 472 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 473
tex1DLayeredLod(T *ptr, ::cudaTextureObject_t obj, float x, int layer, float level) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)level;::exit(___);}
#if 0
#line 474
{ 
#line 478
} 
#endif
#line 480 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 481
tex1DLayeredLod(::cudaTextureObject_t texObject, float x, int layer, float level) {int volatile ___ = 1;(void)texObject;(void)x;(void)layer;(void)level;::exit(___);}
#if 0
#line 482
{ 
#line 488
} 
#endif
#line 491 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 492
tex2DLayeredLod(T *ptr, ::cudaTextureObject_t obj, float x, float y, int layer, float level) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)level;::exit(___);}
#if 0
#line 493
{ 
#line 497
} 
#endif
#line 499 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 500
tex2DLayeredLod(::cudaTextureObject_t texObject, float x, float y, int layer, float level) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)level;::exit(___);}
#if 0
#line 501
{ 
#line 507
} 
#endif
#line 510 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 511
tex2DLayeredLod(T *ptr, ::cudaTextureObject_t obj, float x, float y, int layer, float level, bool *isResident) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)level;(void)isResident;::exit(___);}
#if 0
#line 512
{ 
#line 518
} 
#endif
#line 520 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 521
tex2DLayeredLod(::cudaTextureObject_t texObject, float x, float y, int layer, float level, bool *isResident) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)level;(void)isResident;::exit(___);}
#if 0
#line 522
{ 
#line 528
} 
#endif
#line 531 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 532
texCubemapLod(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, float level) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)level;::exit(___);}
#if 0
#line 533
{ 
#line 537
} 
#endif
#line 539 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 540
texCubemapLod(::cudaTextureObject_t texObject, float x, float y, float z, float level) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;::exit(___);}
#if 0
#line 541
{ 
#line 547
} 
#endif
#line 550 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 551
texCubemapGrad(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 552
{ 
#line 556
} 
#endif
#line 558 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 559
texCubemapGrad(::cudaTextureObject_t texObject, float x, float y, float z, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 560
{ 
#line 566
} 
#endif
#line 568 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 569
texCubemapLayeredLod(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, int layer, float level) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;(void)level;::exit(___);}
#if 0
#line 570
{ 
#line 574
} 
#endif
#line 576 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 577
texCubemapLayeredLod(::cudaTextureObject_t texObject, float x, float y, float z, int layer, float level) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;(void)level;::exit(___);}
#if 0
#line 578
{ 
#line 584
} 
#endif
#line 586 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 587
tex1DGrad(T *ptr, ::cudaTextureObject_t obj, float x, float dPdx, float dPdy) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 588
{ 
#line 592
} 
#endif
#line 594 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 595
tex1DGrad(::cudaTextureObject_t texObject, float x, float dPdx, float dPdy) {int volatile ___ = 1;(void)texObject;(void)x;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 596
{ 
#line 602
} 
#endif
#line 605 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 606
tex2DGrad(T *ptr, ::cudaTextureObject_t obj, float x, float y, ::float2 dPdx, ::float2 dPdy) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 607
{ 
#line 612
} 
#endif
#line 614 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 615
tex2DGrad(::cudaTextureObject_t texObject, float x, float y, ::float2 dPdx, ::float2 dPdy) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 616
{ 
#line 622
} 
#endif
#line 625 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 626
tex2DGrad(T *ptr, ::cudaTextureObject_t obj, float x, float y, ::float2 dPdx, ::float2 dPdy, bool *isResident) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)dPdx;(void)dPdy;(void)isResident;::exit(___);}
#if 0
#line 627
{ 
#line 634
} 
#endif
#line 636 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 637
tex2DGrad(::cudaTextureObject_t texObject, float x, float y, ::float2 dPdx, ::float2 dPdy, bool *isResident) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)dPdx;(void)dPdy;(void)isResident;::exit(___);}
#if 0
#line 638
{ 
#line 644
} 
#endif
#line 648 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 649
tex3DGrad(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 650
{ 
#line 654
} 
#endif
#line 656 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 657
tex3DGrad(::cudaTextureObject_t texObject, float x, float y, float z, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 658
{ 
#line 664
} 
#endif
#line 667 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 668
tex3DGrad(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, ::float4 dPdx, ::float4 dPdy, bool *isResident) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;(void)isResident;::exit(___);}
#if 0
#line 669
{ 
#line 675
} 
#endif
#line 677 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 678
tex3DGrad(::cudaTextureObject_t texObject, float x, float y, float z, ::float4 dPdx, ::float4 dPdy, bool *isResident) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;(void)isResident;::exit(___);}
#if 0
#line 679
{ 
#line 685
} 
#endif
#line 690 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 691
tex1DLayeredGrad(T *ptr, ::cudaTextureObject_t obj, float x, int layer, float dPdx, float dPdy) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 692
{ 
#line 696
} 
#endif
#line 698 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 699
tex1DLayeredGrad(::cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy) {int volatile ___ = 1;(void)texObject;(void)x;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 700
{ 
#line 706
} 
#endif
#line 709 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 710
tex2DLayeredGrad(T *ptr, ::cudaTextureObject_t obj, float x, float y, int layer, ::float2 dPdx, ::float2 dPdy) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 711
{ 
#line 715
} 
#endif
#line 717 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 718
tex2DLayeredGrad(::cudaTextureObject_t texObject, float x, float y, int layer, ::float2 dPdx, ::float2 dPdy) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 719
{ 
#line 725
} 
#endif
#line 728 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 729
tex2DLayeredGrad(T *ptr, ::cudaTextureObject_t obj, float x, float y, int layer, ::float2 dPdx, ::float2 dPdy, bool *isResident) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;(void)isResident;::exit(___);}
#if 0
#line 730
{ 
#line 736
} 
#endif
#line 738 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 739
tex2DLayeredGrad(::cudaTextureObject_t texObject, float x, float y, int layer, ::float2 dPdx, ::float2 dPdy, bool *isResident) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;(void)isResident;::exit(___);}
#if 0
#line 740
{ 
#line 746
} 
#endif
#line 750 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static typename __nv_itex_trait< T> ::type 
#line 751
texCubemapLayeredGrad(T *ptr, ::cudaTextureObject_t obj, float x, float y, float z, int layer, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 752
{ 
#line 756
} 
#endif
#line 758 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\texture_indirect_functions.h"
template< class T> static T 
#line 759
texCubemapLayeredGrad(::cudaTextureObject_t texObject, float x, float y, float z, int layer, ::float4 dPdx, ::float4 dPdy) {int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;::exit(___);}
#if 0
#line 760
{ 
#line 766
} 
#endif
#line 59 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_indirect_functions.h"
template< class T> struct __nv_isurf_trait { }; 
#line 60
template<> struct __nv_isurf_trait< char>  { typedef void type; }; 
#line 61
template<> struct __nv_isurf_trait< signed char>  { typedef void type; }; 
#line 62
template<> struct __nv_isurf_trait< char1>  { typedef void type; }; 
#line 63
template<> struct __nv_isurf_trait< unsigned char>  { typedef void type; }; 
#line 64
template<> struct __nv_isurf_trait< uchar1>  { typedef void type; }; 
#line 65
template<> struct __nv_isurf_trait< short>  { typedef void type; }; 
#line 66
template<> struct __nv_isurf_trait< short1>  { typedef void type; }; 
#line 67
template<> struct __nv_isurf_trait< unsigned short>  { typedef void type; }; 
#line 68
template<> struct __nv_isurf_trait< ushort1>  { typedef void type; }; 
#line 69
template<> struct __nv_isurf_trait< int>  { typedef void type; }; 
#line 70
template<> struct __nv_isurf_trait< int1>  { typedef void type; }; 
#line 71
template<> struct __nv_isurf_trait< unsigned>  { typedef void type; }; 
#line 72
template<> struct __nv_isurf_trait< uint1>  { typedef void type; }; 
#line 73
template<> struct __nv_isurf_trait< __int64>  { typedef void type; }; 
#line 74
template<> struct __nv_isurf_trait< longlong1>  { typedef void type; }; 
#line 75
template<> struct __nv_isurf_trait< unsigned __int64>  { typedef void type; }; 
#line 76
template<> struct __nv_isurf_trait< ulonglong1>  { typedef void type; }; 
#line 77
template<> struct __nv_isurf_trait< float>  { typedef void type; }; 
#line 78
template<> struct __nv_isurf_trait< float1>  { typedef void type; }; 
#line 80
template<> struct __nv_isurf_trait< char2>  { typedef void type; }; 
#line 81
template<> struct __nv_isurf_trait< uchar2>  { typedef void type; }; 
#line 82
template<> struct __nv_isurf_trait< short2>  { typedef void type; }; 
#line 83
template<> struct __nv_isurf_trait< ushort2>  { typedef void type; }; 
#line 84
template<> struct __nv_isurf_trait< int2>  { typedef void type; }; 
#line 85
template<> struct __nv_isurf_trait< uint2>  { typedef void type; }; 
#line 86
template<> struct __nv_isurf_trait< longlong2>  { typedef void type; }; 
#line 87
template<> struct __nv_isurf_trait< ulonglong2>  { typedef void type; }; 
#line 88
template<> struct __nv_isurf_trait< float2>  { typedef void type; }; 
#line 90
template<> struct __nv_isurf_trait< char4>  { typedef void type; }; 
#line 91
template<> struct __nv_isurf_trait< uchar4>  { typedef void type; }; 
#line 92
template<> struct __nv_isurf_trait< short4>  { typedef void type; }; 
#line 93
template<> struct __nv_isurf_trait< ushort4>  { typedef void type; }; 
#line 94
template<> struct __nv_isurf_trait< int4>  { typedef void type; }; 
#line 95
template<> struct __nv_isurf_trait< uint4>  { typedef void type; }; 
#line 96
template<> struct __nv_isurf_trait< float4>  { typedef void type; }; 
#line 99
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 100
surf1Dread(T *ptr, ::cudaSurfaceObject_t obj, int x, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)mode;::exit(___);}
#if 0
#line 101
{ 
#line 105
} 
#endif
#line 107 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 108
surf1Dread(::cudaSurfaceObject_t surfObject, int x, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surfObject;(void)x;(void)boundaryMode;::exit(___);}
#if 0
#line 109
{ 
#line 115
} 
#endif
#line 117 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 118
surf2Dread(T *ptr, ::cudaSurfaceObject_t obj, int x, int y, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)mode;::exit(___);}
#if 0
#line 119
{ 
#line 123
} 
#endif
#line 125 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 126
surf2Dread(::cudaSurfaceObject_t surfObject, int x, int y, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)boundaryMode;::exit(___);}
#if 0
#line 127
{ 
#line 133
} 
#endif
#line 136 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 137
surf3Dread(T *ptr, ::cudaSurfaceObject_t obj, int x, int y, int z, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)mode;::exit(___);}
#if 0
#line 138
{ 
#line 142
} 
#endif
#line 144 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 145
surf3Dread(::cudaSurfaceObject_t surfObject, int x, int y, int z, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)z;(void)boundaryMode;::exit(___);}
#if 0
#line 146
{ 
#line 152
} 
#endif
#line 154 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 155
surf1DLayeredread(T *ptr, ::cudaSurfaceObject_t obj, int x, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)mode;::exit(___);}
#if 0
#line 156
{ 
#line 160
} 
#endif
#line 162 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 163
surf1DLayeredread(::cudaSurfaceObject_t surfObject, int x, int layer, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surfObject;(void)x;(void)layer;(void)boundaryMode;::exit(___);}
#if 0
#line 164
{ 
#line 170
} 
#endif
#line 172 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 173
surf2DLayeredread(T *ptr, ::cudaSurfaceObject_t obj, int x, int y, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)mode;::exit(___);}
#if 0
#line 174
{ 
#line 178
} 
#endif
#line 180 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 181
surf2DLayeredread(::cudaSurfaceObject_t surfObject, int x, int y, int layer, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)layer;(void)boundaryMode;::exit(___);}
#if 0
#line 182
{ 
#line 188
} 
#endif
#line 190 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 191
surfCubemapread(T *ptr, ::cudaSurfaceObject_t obj, int x, int y, int face, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)face;(void)mode;::exit(___);}
#if 0
#line 192
{ 
#line 196
} 
#endif
#line 198 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 199
surfCubemapread(::cudaSurfaceObject_t surfObject, int x, int y, int face, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)face;(void)boundaryMode;::exit(___);}
#if 0
#line 200
{ 
#line 206
} 
#endif
#line 208 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 209
surfCubemapLayeredread(T *ptr, ::cudaSurfaceObject_t obj, int x, int y, int layerface, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layerface;(void)mode;::exit(___);}
#if 0
#line 210
{ 
#line 214
} 
#endif
#line 216 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_indirect_functions.h"
template< class T> static T 
#line 217
surfCubemapLayeredread(::cudaSurfaceObject_t surfObject, int x, int y, int layerface, ::cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)layerface;(void)boundaryMode;::exit(___);}
#if 0
#line 218
{ 
#line 224
} 
#endif
#line 226 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 227
surf1Dwrite(T val, ::cudaSurfaceObject_t obj, int x, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)mode;::exit(___);}
#if 0
#line 228
{ 
#line 232
} 
#endif
#line 234 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 235
surf2Dwrite(T val, ::cudaSurfaceObject_t obj, int x, int y, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)mode;::exit(___);}
#if 0
#line 236
{ 
#line 240
} 
#endif
#line 242 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 243
surf3Dwrite(T val, ::cudaSurfaceObject_t obj, int x, int y, int z, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)z;(void)mode;::exit(___);}
#if 0
#line 244
{ 
#line 248
} 
#endif
#line 250 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 251
surf1DLayeredwrite(T val, ::cudaSurfaceObject_t obj, int x, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)layer;(void)mode;::exit(___);}
#if 0
#line 252
{ 
#line 256
} 
#endif
#line 258 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 259
surf2DLayeredwrite(T val, ::cudaSurfaceObject_t obj, int x, int y, int layer, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)layer;(void)mode;::exit(___);}
#if 0
#line 260
{ 
#line 264
} 
#endif
#line 266 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 267
surfCubemapwrite(T val, ::cudaSurfaceObject_t obj, int x, int y, int face, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)face;(void)mode;::exit(___);}
#if 0
#line 268
{ 
#line 272
} 
#endif
#line 274 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\surface_indirect_functions.h"
template< class T> static typename __nv_isurf_trait< T> ::type 
#line 275
surfCubemapLayeredwrite(T val, ::cudaSurfaceObject_t obj, int x, int y, int layerface, ::cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)layerface;(void)mode;::exit(___);}
#if 0
#line 276
{ 
#line 280
} 
#endif
#line 3297 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\crt/device_functions.h"
extern "C" unsigned __stdcall __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, CUstream_st * stream = 0); 
#line 68 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\device_launch_parameters.h"
extern "C" {
#line 71 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.1\\include\\device_launch_parameters.h"
extern const uint3 __device_builtin_variable_threadIdx; 
#line 72
extern const uint3 __device_builtin_variable_blockIdx; 
#line 73
extern const dim3 __device_builtin_variable_blockDim; 
#line 74
extern const dim3 __device_builtin_variable_gridDim; 
#line 75
extern const int __device_builtin_variable_warpSize; 
#line 80
}
#line 200 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1/bin/../include\\cuda_runtime.h"
template< class T> static __inline ::cudaError_t 
#line 201
cudaLaunchKernel(const T *
#line 202
func, ::dim3 
#line 203
gridDim, ::dim3 
#line 204
blockDim, void **
#line 205
args, ::size_t 
#line 206
sharedMem = 0, ::cudaStream_t 
#line 207
stream = 0) 
#line 209
{ 
#line 210
return ::cudaLaunchKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream); 
#line 211
} 
#line 262
template< class T> static __inline ::cudaError_t 
#line 263
cudaLaunchCooperativeKernel(const T *
#line 264
func, ::dim3 
#line 265
gridDim, ::dim3 
#line 266
blockDim, void **
#line 267
args, ::size_t 
#line 268
sharedMem = 0, ::cudaStream_t 
#line 269
stream = 0) 
#line 271
{ 
#line 272
return ::cudaLaunchCooperativeKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream); 
#line 273
} 
#line 306
static __inline cudaError_t cudaEventCreate(cudaEvent_t *
#line 307
event, unsigned 
#line 308
flags) 
#line 310
{ 
#line 311
return ::cudaEventCreateWithFlags(event, flags); 
#line 312
} 
#line 371
static __inline cudaError_t cudaMallocHost(void **
#line 372
ptr, size_t 
#line 373
size, unsigned 
#line 374
flags) 
#line 376
{ 
#line 377
return ::cudaHostAlloc(ptr, size, flags); 
#line 378
} 
#line 380
template< class T> static __inline ::cudaError_t 
#line 381
cudaHostAlloc(T **
#line 382
ptr, ::size_t 
#line 383
size, unsigned 
#line 384
flags) 
#line 386
{ 
#line 387
return ::cudaHostAlloc((void **)((void *)ptr), size, flags); 
#line 388
} 
#line 390
template< class T> static __inline ::cudaError_t 
#line 391
cudaHostGetDevicePointer(T **
#line 392
pDevice, void *
#line 393
pHost, unsigned 
#line 394
flags) 
#line 396
{ 
#line 397
return ::cudaHostGetDevicePointer((void **)((void *)pDevice), pHost, flags); 
#line 398
} 
#line 500
template< class T> static __inline ::cudaError_t 
#line 501
cudaMallocManaged(T **
#line 502
devPtr, ::size_t 
#line 503
size, unsigned 
#line 504
flags = 1) 
#line 506
{ 
#line 507
return ::cudaMallocManaged((void **)((void *)devPtr), size, flags); 
#line 508
} 
#line 590
template< class T> static __inline ::cudaError_t 
#line 591
cudaStreamAttachMemAsync(::cudaStream_t 
#line 592
stream, T *
#line 593
devPtr, ::size_t 
#line 594
length = 0, unsigned 
#line 595
flags = 4) 
#line 597
{ 
#line 598
return ::cudaStreamAttachMemAsync(stream, (void *)devPtr, length, flags); 
#line 599
} 
#line 601
template< class T> __inline ::cudaError_t 
#line 602
cudaMalloc(T **
#line 603
devPtr, ::size_t 
#line 604
size) 
#line 606
{ 
#line 607
return ::cudaMalloc((void **)((void *)devPtr), size); 
#line 608
} 
#line 610
template< class T> static __inline ::cudaError_t 
#line 611
cudaMallocHost(T **
#line 612
ptr, ::size_t 
#line 613
size, unsigned 
#line 614
flags = 0) 
#line 616
{ 
#line 617
return cudaMallocHost((void **)((void *)ptr), size, flags); 
#line 618
} 
#line 620
template< class T> static __inline ::cudaError_t 
#line 621
cudaMallocPitch(T **
#line 622
devPtr, ::size_t *
#line 623
pitch, ::size_t 
#line 624
width, ::size_t 
#line 625
height) 
#line 627
{ 
#line 628
return ::cudaMallocPitch((void **)((void *)devPtr), pitch, width, height); 
#line 629
} 
#line 668
template< class T> static __inline ::cudaError_t 
#line 669
cudaMemcpyToSymbol(const T &
#line 670
symbol, const void *
#line 671
src, ::size_t 
#line 672
count, ::size_t 
#line 673
offset = 0, ::cudaMemcpyKind 
#line 674
kind = cudaMemcpyHostToDevice) 
#line 676
{ 
#line 677
return ::cudaMemcpyToSymbol((const void *)(&symbol), src, count, offset, kind); 
#line 678
} 
#line 722
template< class T> static __inline ::cudaError_t 
#line 723
cudaMemcpyToSymbolAsync(const T &
#line 724
symbol, const void *
#line 725
src, ::size_t 
#line 726
count, ::size_t 
#line 727
offset = 0, ::cudaMemcpyKind 
#line 728
kind = cudaMemcpyHostToDevice, ::cudaStream_t 
#line 729
stream = 0) 
#line 731
{ 
#line 732
return ::cudaMemcpyToSymbolAsync((const void *)(&symbol), src, count, offset, kind, stream); 
#line 733
} 
#line 770
template< class T> static __inline ::cudaError_t 
#line 771
cudaMemcpyFromSymbol(void *
#line 772
dst, const T &
#line 773
symbol, ::size_t 
#line 774
count, ::size_t 
#line 775
offset = 0, ::cudaMemcpyKind 
#line 776
kind = cudaMemcpyDeviceToHost) 
#line 778
{ 
#line 779
return ::cudaMemcpyFromSymbol(dst, (const void *)(&symbol), count, offset, kind); 
#line 780
} 
#line 824
template< class T> static __inline ::cudaError_t 
#line 825
cudaMemcpyFromSymbolAsync(void *
#line 826
dst, const T &
#line 827
symbol, ::size_t 
#line 828
count, ::size_t 
#line 829
offset = 0, ::cudaMemcpyKind 
#line 830
kind = cudaMemcpyDeviceToHost, ::cudaStream_t 
#line 831
stream = 0) 
#line 833
{ 
#line 834
return ::cudaMemcpyFromSymbolAsync(dst, (const void *)(&symbol), count, offset, kind, stream); 
#line 835
} 
#line 893
template< class T> static __inline ::cudaError_t 
#line 894
cudaGraphAddMemcpyNodeToSymbol(::cudaGraphNode_t *
#line 895
pGraphNode, ::cudaGraph_t 
#line 896
graph, const ::cudaGraphNode_t *
#line 897
pDependencies, ::size_t 
#line 898
numDependencies, const T &
#line 899
symbol, const void *
#line 900
src, ::size_t 
#line 901
count, ::size_t 
#line 902
offset, ::cudaMemcpyKind 
#line 903
kind) 
#line 904
{ 
#line 905
return ::cudaGraphAddMemcpyNodeToSymbol(pGraphNode, graph, pDependencies, numDependencies, (const void *)(&symbol), src, count, offset, kind); 
#line 906
} 
#line 964
template< class T> static __inline ::cudaError_t 
#line 965
cudaGraphAddMemcpyNodeFromSymbol(::cudaGraphNode_t *
#line 966
pGraphNode, ::cudaGraph_t 
#line 967
graph, const ::cudaGraphNode_t *
#line 968
pDependencies, ::size_t 
#line 969
numDependencies, void *
#line 970
dst, const T &
#line 971
symbol, ::size_t 
#line 972
count, ::size_t 
#line 973
offset, ::cudaMemcpyKind 
#line 974
kind) 
#line 975
{ 
#line 976
return ::cudaGraphAddMemcpyNodeFromSymbol(pGraphNode, graph, pDependencies, numDependencies, dst, (const void *)(&symbol), count, offset, kind); 
#line 977
} 
#line 1015
template< class T> static __inline ::cudaError_t 
#line 1016
cudaGraphMemcpyNodeSetParamsToSymbol(::cudaGraphNode_t 
#line 1017
node, const T &
#line 1018
symbol, const void *
#line 1019
src, ::size_t 
#line 1020
count, ::size_t 
#line 1021
offset, ::cudaMemcpyKind 
#line 1022
kind) 
#line 1023
{ 
#line 1024
return ::cudaGraphMemcpyNodeSetParamsToSymbol(node, (const void *)(&symbol), src, count, offset, kind); 
#line 1025
} 
#line 1063
template< class T> static __inline ::cudaError_t 
#line 1064
cudaGraphMemcpyNodeSetParamsFromSymbol(::cudaGraphNode_t 
#line 1065
node, void *
#line 1066
dst, const T &
#line 1067
symbol, ::size_t 
#line 1068
count, ::size_t 
#line 1069
offset, ::cudaMemcpyKind 
#line 1070
kind) 
#line 1071
{ 
#line 1072
return ::cudaGraphMemcpyNodeSetParamsFromSymbol(node, dst, (const void *)(&symbol), count, offset, kind); 
#line 1073
} 
#line 1121
template< class T> static __inline ::cudaError_t 
#line 1122
cudaGraphExecMemcpyNodeSetParamsToSymbol(::cudaGraphExec_t 
#line 1123
hGraphExec, ::cudaGraphNode_t 
#line 1124
node, const T &
#line 1125
symbol, const void *
#line 1126
src, ::size_t 
#line 1127
count, ::size_t 
#line 1128
offset, ::cudaMemcpyKind 
#line 1129
kind) 
#line 1130
{ 
#line 1131
return ::cudaGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec, node, (const void *)(&symbol), src, count, offset, kind); 
#line 1132
} 
#line 1180
template< class T> static __inline ::cudaError_t 
#line 1181
cudaGraphExecMemcpyNodeSetParamsFromSymbol(::cudaGraphExec_t 
#line 1182
hGraphExec, ::cudaGraphNode_t 
#line 1183
node, void *
#line 1184
dst, const T &
#line 1185
symbol, ::size_t 
#line 1186
count, ::size_t 
#line 1187
offset, ::cudaMemcpyKind 
#line 1188
kind) 
#line 1189
{ 
#line 1190
return ::cudaGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec, node, dst, (const void *)(&symbol), count, offset, kind); 
#line 1191
} 
#line 1216
template< class T> static __inline ::cudaError_t 
#line 1217
cudaGetSymbolAddress(void **
#line 1218
devPtr, const T &
#line 1219
symbol) 
#line 1221
{ 
#line 1222
return ::cudaGetSymbolAddress(devPtr, (const void *)(&symbol)); 
#line 1223
} 
#line 1248
template< class T> static __inline ::cudaError_t 
#line 1249
cudaGetSymbolSize(::size_t *
#line 1250
size, const T &
#line 1251
symbol) 
#line 1253
{ 
#line 1254
return ::cudaGetSymbolSize(size, (const void *)(&symbol)); 
#line 1255
} 
#line 1292
template< class T, int dim, cudaTextureReadMode readMode> 
#line 1293
__declspec(deprecated) static __inline ::cudaError_t cudaBindTexture(::size_t *
#line 1294
offset, const texture< T, dim, readMode>  &
#line 1295
tex, const void *
#line 1296
devPtr, const ::cudaChannelFormatDesc &
#line 1297
desc, ::size_t 
#line 1298
size = 4294967295U) 
#line 1300
{ 
#line 1301
return ::cudaBindTexture(offset, &tex, devPtr, &desc, size); 
#line 1302
} 
#line 1338
template< class T, int dim, cudaTextureReadMode readMode> 
#line 1339
__declspec(deprecated) static __inline ::cudaError_t cudaBindTexture(::size_t *
#line 1340
offset, const texture< T, dim, readMode>  &
#line 1341
tex, const void *
#line 1342
devPtr, ::size_t 
#line 1343
size = 4294967295U) 
#line 1345
{ 
#line 1346
return cudaBindTexture(offset, tex, devPtr, (tex.channelDesc), size); 
#line 1347
} 
#line 1395
template< class T, int dim, cudaTextureReadMode readMode> 
#line 1396
__declspec(deprecated) static __inline ::cudaError_t cudaBindTexture2D(::size_t *
#line 1397
offset, const texture< T, dim, readMode>  &
#line 1398
tex, const void *
#line 1399
devPtr, const ::cudaChannelFormatDesc &
#line 1400
desc, ::size_t 
#line 1401
width, ::size_t 
#line 1402
height, ::size_t 
#line 1403
pitch) 
#line 1405
{ 
#line 1406
return ::cudaBindTexture2D(offset, &tex, devPtr, &desc, width, height, pitch); 
#line 1407
} 
#line 1454
template< class T, int dim, cudaTextureReadMode readMode> 
#line 1455
__declspec(deprecated) static __inline ::cudaError_t cudaBindTexture2D(::size_t *
#line 1456
offset, const texture< T, dim, readMode>  &
#line 1457
tex, const void *
#line 1458
devPtr, ::size_t 
#line 1459
width, ::size_t 
#line 1460
height, ::size_t 
#line 1461
pitch) 
#line 1463
{ 
#line 1464
return ::cudaBindTexture2D(offset, &tex, devPtr, &(tex.channelDesc), width, height, pitch); 
#line 1465
} 
#line 1497
template< class T, int dim, cudaTextureReadMode readMode> 
#line 1498
__declspec(deprecated) static __inline ::cudaError_t cudaBindTextureToArray(const texture< T, dim, readMode>  &
#line 1499
tex, ::cudaArray_const_t 
#line 1500
array, const ::cudaChannelFormatDesc &
#line 1501
desc) 
#line 1503
{ 
#line 1504
return ::cudaBindTextureToArray(&tex, array, &desc); 
#line 1505
} 
#line 1536
template< class T, int dim, cudaTextureReadMode readMode> 
#line 1537
__declspec(deprecated) static __inline ::cudaError_t cudaBindTextureToArray(const texture< T, dim, readMode>  &
#line 1538
tex, ::cudaArray_const_t 
#line 1539
array) 
#line 1541
{ 
#line 1542
::cudaChannelFormatDesc desc; 
#line 1543
::cudaError_t err = ::cudaGetChannelDesc(&desc, array); 
#line 1545
return (err == (cudaSuccess)) ? cudaBindTextureToArray(tex, array, desc) : err; 
#line 1546
} 
#line 1578
template< class T, int dim, cudaTextureReadMode readMode> 
#line 1579
__declspec(deprecated) static __inline ::cudaError_t cudaBindTextureToMipmappedArray(const texture< T, dim, readMode>  &
#line 1580
tex, ::cudaMipmappedArray_const_t 
#line 1581
mipmappedArray, const ::cudaChannelFormatDesc &
#line 1582
desc) 
#line 1584
{ 
#line 1585
return ::cudaBindTextureToMipmappedArray(&tex, mipmappedArray, &desc); 
#line 1586
} 
#line 1617
template< class T, int dim, cudaTextureReadMode readMode> 
#line 1618
__declspec(deprecated) static __inline ::cudaError_t cudaBindTextureToMipmappedArray(const texture< T, dim, readMode>  &
#line 1619
tex, ::cudaMipmappedArray_const_t 
#line 1620
mipmappedArray) 
#line 1622
{ 
#line 1623
::cudaChannelFormatDesc desc; 
#line 1624
::cudaArray_t levelArray; 
#line 1625
::cudaError_t err = ::cudaGetMipmappedArrayLevel(&levelArray, mipmappedArray, 0); 
#line 1627
if (err != (cudaSuccess)) { 
#line 1628
return err; 
#line 1629
}  
#line 1630
err = ::cudaGetChannelDesc(&desc, levelArray); 
#line 1632
return (err == (cudaSuccess)) ? cudaBindTextureToMipmappedArray(tex, mipmappedArray, desc) : err; 
#line 1633
} 
#line 1660
template< class T, int dim, cudaTextureReadMode readMode> 
#line 1661
__declspec(deprecated) static __inline ::cudaError_t cudaUnbindTexture(const texture< T, dim, readMode>  &
#line 1662
tex) 
#line 1664
{ 
#line 1665
return ::cudaUnbindTexture(&tex); 
#line 1666
} 
#line 1696
template< class T, int dim, cudaTextureReadMode readMode> 
#line 1697
__declspec(deprecated) static __inline ::cudaError_t cudaGetTextureAlignmentOffset(::size_t *
#line 1698
offset, const texture< T, dim, readMode>  &
#line 1699
tex) 
#line 1701
{ 
#line 1702
return ::cudaGetTextureAlignmentOffset(offset, &tex); 
#line 1703
} 
#line 1748
template< class T> static __inline ::cudaError_t 
#line 1749
cudaFuncSetCacheConfig(T *
#line 1750
func, ::cudaFuncCache 
#line 1751
cacheConfig) 
#line 1753
{ 
#line 1754
return ::cudaFuncSetCacheConfig((const void *)func, cacheConfig); 
#line 1755
} 
#line 1757
template< class T> static __inline ::cudaError_t 
#line 1758
cudaFuncSetSharedMemConfig(T *
#line 1759
func, ::cudaSharedMemConfig 
#line 1760
config) 
#line 1762
{ 
#line 1763
return ::cudaFuncSetSharedMemConfig((const void *)func, config); 
#line 1764
} 
#line 1796 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1/bin/../include\\cuda_runtime.h"
template< class T> __inline ::cudaError_t 
#line 1797
cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *
#line 1798
numBlocks, T 
#line 1799
func, int 
#line 1800
blockSize, ::size_t 
#line 1801
dynamicSMemSize) 
#line 1802
{ 
#line 1803
return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void *)func, blockSize, dynamicSMemSize, 0); 
#line 1804
} 
#line 1848
template< class T> __inline ::cudaError_t 
#line 1849
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *
#line 1850
numBlocks, T 
#line 1851
func, int 
#line 1852
blockSize, ::size_t 
#line 1853
dynamicSMemSize, unsigned 
#line 1854
flags) 
#line 1855
{ 
#line 1856
return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void *)func, blockSize, dynamicSMemSize, flags); 
#line 1857
} 
#line 1862
class __cudaOccupancyB2DHelper { 
#line 1863
size_t n; 
#line 1865
public: __cudaOccupancyB2DHelper(size_t n_) : n(n_) { } 
#line 1866
size_t operator()(int) 
#line 1867
{ 
#line 1868
return n; 
#line 1869
} 
#line 1870
}; 
#line 1918
template< class UnaryFunction, class T> static __inline ::cudaError_t 
#line 1919
cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(int *
#line 1920
minGridSize, int *
#line 1921
blockSize, T 
#line 1922
func, UnaryFunction 
#line 1923
blockSizeToDynamicSMemSize, int 
#line 1924
blockSizeLimit = 0, unsigned 
#line 1925
flags = 0) 
#line 1926
{ 
#line 1927
::cudaError_t status; 
#line 1930
int device; 
#line 1931
::cudaFuncAttributes attr; 
#line 1934
int maxThreadsPerMultiProcessor; 
#line 1935
int warpSize; 
#line 1936
int devMaxThreadsPerBlock; 
#line 1937
int multiProcessorCount; 
#line 1938
int funcMaxThreadsPerBlock; 
#line 1939
int occupancyLimit; 
#line 1940
int granularity; 
#line 1943
int maxBlockSize = 0; 
#line 1944
int numBlocks = 0; 
#line 1945
int maxOccupancy = 0; 
#line 1948
int blockSizeToTryAligned; 
#line 1949
int blockSizeToTry; 
#line 1950
int blockSizeLimitAligned; 
#line 1951
int occupancyInBlocks; 
#line 1952
int occupancyInThreads; 
#line 1953
::size_t dynamicSMemSize; 
#line 1959
if (((!minGridSize) || (!blockSize)) || (!func)) { 
#line 1960
return cudaErrorInvalidValue; 
#line 1961
}  
#line 1967
status = ::cudaGetDevice(&device); 
#line 1968
if (status != (cudaSuccess)) { 
#line 1969
return status; 
#line 1970
}  
#line 1972
status = cudaDeviceGetAttribute(&maxThreadsPerMultiProcessor, cudaDevAttrMaxThreadsPerMultiProcessor, device); 
#line 1976
if (status != (cudaSuccess)) { 
#line 1977
return status; 
#line 1978
}  
#line 1980
status = cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, device); 
#line 1984
if (status != (cudaSuccess)) { 
#line 1985
return status; 
#line 1986
}  
#line 1988
status = cudaDeviceGetAttribute(&devMaxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, device); 
#line 1992
if (status != (cudaSuccess)) { 
#line 1993
return status; 
#line 1994
}  
#line 1996
status = cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, device); 
#line 2000
if (status != (cudaSuccess)) { 
#line 2001
return status; 
#line 2002
}  
#line 2004
status = cudaFuncGetAttributes(&attr, func); 
#line 2005
if (status != (cudaSuccess)) { 
#line 2006
return status; 
#line 2007
}  
#line 2009
funcMaxThreadsPerBlock = (attr.maxThreadsPerBlock); 
#line 2015
occupancyLimit = maxThreadsPerMultiProcessor; 
#line 2016
granularity = warpSize; 
#line 2018
if (blockSizeLimit == 0) { 
#line 2019
blockSizeLimit = devMaxThreadsPerBlock; 
#line 2020
}  
#line 2022
if (devMaxThreadsPerBlock < blockSizeLimit) { 
#line 2023
blockSizeLimit = devMaxThreadsPerBlock; 
#line 2024
}  
#line 2026
if (funcMaxThreadsPerBlock < blockSizeLimit) { 
#line 2027
blockSizeLimit = funcMaxThreadsPerBlock; 
#line 2028
}  
#line 2030
blockSizeLimitAligned = (((blockSizeLimit + (granularity - 1)) / granularity) * granularity); 
#line 2032
for (blockSizeToTryAligned = blockSizeLimitAligned; blockSizeToTryAligned > 0; blockSizeToTryAligned -= granularity) { 
#line 2036
if (blockSizeLimit < blockSizeToTryAligned) { 
#line 2037
blockSizeToTry = blockSizeLimit; 
#line 2038
} else { 
#line 2039
blockSizeToTry = blockSizeToTryAligned; 
#line 2040
}  
#line 2042
dynamicSMemSize = blockSizeToDynamicSMemSize(blockSizeToTry); 
#line 2044
status = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&occupancyInBlocks, func, blockSizeToTry, dynamicSMemSize, flags); 
#line 2051
if (status != (cudaSuccess)) { 
#line 2052
return status; 
#line 2053
}  
#line 2055
occupancyInThreads = (blockSizeToTry * occupancyInBlocks); 
#line 2057
if (occupancyInThreads > maxOccupancy) { 
#line 2058
maxBlockSize = blockSizeToTry; 
#line 2059
numBlocks = occupancyInBlocks; 
#line 2060
maxOccupancy = occupancyInThreads; 
#line 2061
}  
#line 2065
if (occupancyLimit == maxOccupancy) { 
#line 2066
break; 
#line 2067
}  
#line 2068
}  
#line 2076
(*minGridSize) = (numBlocks * multiProcessorCount); 
#line 2077
(*blockSize) = maxBlockSize; 
#line 2079
return status; 
#line 2080
} 
#line 2114
template< class UnaryFunction, class T> static __inline ::cudaError_t 
#line 2115
cudaOccupancyMaxPotentialBlockSizeVariableSMem(int *
#line 2116
minGridSize, int *
#line 2117
blockSize, T 
#line 2118
func, UnaryFunction 
#line 2119
blockSizeToDynamicSMemSize, int 
#line 2120
blockSizeLimit = 0) 
#line 2121
{ 
#line 2122
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, blockSizeLimit, 0); 
#line 2123
} 
#line 2160
template< class T> static __inline ::cudaError_t 
#line 2161
cudaOccupancyMaxPotentialBlockSize(int *
#line 2162
minGridSize, int *
#line 2163
blockSize, T 
#line 2164
func, ::size_t 
#line 2165
dynamicSMemSize = 0, int 
#line 2166
blockSizeLimit = 0) 
#line 2167
{ 
#line 2168
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, ((::__cudaOccupancyB2DHelper)(dynamicSMemSize)), blockSizeLimit, 0); 
#line 2169
} 
#line 2198
template< class T> static __inline ::cudaError_t 
#line 2199
cudaOccupancyAvailableDynamicSMemPerBlock(::size_t *
#line 2200
dynamicSmemSize, T 
#line 2201
func, int 
#line 2202
numBlocks, int 
#line 2203
blockSize) 
#line 2204
{ 
#line 2205
return ::cudaOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, (const void *)func, numBlocks, blockSize); 
#line 2206
} 
#line 2257
template< class T> static __inline ::cudaError_t 
#line 2258
cudaOccupancyMaxPotentialBlockSizeWithFlags(int *
#line 2259
minGridSize, int *
#line 2260
blockSize, T 
#line 2261
func, ::size_t 
#line 2262
dynamicSMemSize = 0, int 
#line 2263
blockSizeLimit = 0, unsigned 
#line 2264
flags = 0) 
#line 2265
{ 
#line 2266
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, ((::__cudaOccupancyB2DHelper)(dynamicSMemSize)), blockSizeLimit, flags); 
#line 2267
} 
#line 2300
template< class T> __inline ::cudaError_t 
#line 2301
cudaFuncGetAttributes(::cudaFuncAttributes *
#line 2302
attr, T *
#line 2303
entry) 
#line 2305
{ 
#line 2306
return ::cudaFuncGetAttributes(attr, (const void *)entry); 
#line 2307
} 
#line 2345
template< class T> static __inline ::cudaError_t 
#line 2346
cudaFuncSetAttribute(T *
#line 2347
entry, ::cudaFuncAttribute 
#line 2348
attr, int 
#line 2349
value) 
#line 2351
{ 
#line 2352
return ::cudaFuncSetAttribute((const void *)entry, attr, value); 
#line 2353
} 
#line 2377
template< class T, int dim> 
#line 2378
__declspec(deprecated) static __inline ::cudaError_t cudaBindSurfaceToArray(const surface< T, dim>  &
#line 2379
surf, ::cudaArray_const_t 
#line 2380
array, const ::cudaChannelFormatDesc &
#line 2381
desc) 
#line 2383
{ 
#line 2384
return ::cudaBindSurfaceToArray(&surf, array, &desc); 
#line 2385
} 
#line 2408
template< class T, int dim> 
#line 2409
__declspec(deprecated) static __inline ::cudaError_t cudaBindSurfaceToArray(const surface< T, dim>  &
#line 2410
surf, ::cudaArray_const_t 
#line 2411
array) 
#line 2413
{ 
#line 2414
::cudaChannelFormatDesc desc; 
#line 2415
::cudaError_t err = ::cudaGetChannelDesc(&desc, array); 
#line 2417
return (err == (cudaSuccess)) ? cudaBindSurfaceToArray(surf, array, desc) : err; 
#line 2418
} 
#line 2432 "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1/bin/../include\\cuda_runtime.h"
#pragma warning(pop)
#line 40 "CMakeCUDACompilerId.cu"
const char *info_compiler = ("INFO:compiler[NVIDIA]"); 
#line 42
const char *info_simulate = ("INFO:simulate[MSVC]"); 
#line 278 "CMakeCUDACompilerId.cu"
const char info_version[] = {'I', 'N', 'F', 'O', ':', 'c', 'o', 'm', 'p', 'i', 'l', 'e', 'r', '_', 'v', 'e', 'r', 's', 'i', 'o', 'n', '[', (('0') + ((11 / 10000000) % 10)), (('0') + ((11 / 1000000) % 10)), (('0') + ((11 / 100000) % 10)), (('0') + ((11 / 10000) % 10)), (('0') + ((11 / 1000) % 10)), (('0') + ((11 / 100) % 10)), (('0') + ((11 / 10) % 10)), (('0') + (11 % 10)), '.', (('0') + ((1 / 10000000) % 10)), (('0') + ((1 / 1000000) % 10)), (('0') + ((1 / 100000) % 10)), (('0') + ((1 / 10000) % 10)), (('0') + ((1 / 1000) % 10)), (('0') + ((1 / 100) % 10)), (('0') + ((1 / 10) % 10)), (('0') + (1 % 10)), '.', (('0') + ((105 / 10000000) % 10)), (('0') + ((105 / 1000000) % 10)), (('0') + ((105 / 100000) % 10)), (('0') + ((105 / 10000) % 10)), (('0') + ((105 / 1000) % 10)), (('0') + ((105 / 100) % 10)), (('0') + ((105 / 10) % 10)), (('0') + (105 % 10)), ']', '\000'}; 
#line 305 "CMakeCUDACompilerId.cu"
const char info_simulate_version[] = {'I', 'N', 'F', 'O', ':', 's', 'i', 'm', 'u', 'l', 'a', 't', 'e', '_', 'v', 'e', 'r', 's', 'i', 'o', 'n', '[', (('0') + (((1922 / 100) / 10000000) % 10)), (('0') + (((1922 / 100) / 1000000) % 10)), (('0') + (((1922 / 100) / 100000) % 10)), (('0') + (((1922 / 100) / 10000) % 10)), (('0') + (((1922 / 100) / 1000) % 10)), (('0') + (((1922 / 100) / 100) % 10)), (('0') + (((1922 / 100) / 10) % 10)), (('0') + ((1922 / 100) % 10)), '.', (('0') + (((1922 % 100) / 10000000) % 10)), (('0') + (((1922 % 100) / 1000000) % 10)), (('0') + (((1922 % 100) / 100000) % 10)), (('0') + (((1922 % 100) / 10000) % 10)), (('0') + (((1922 % 100) / 1000) % 10)), (('0') + (((1922 % 100) / 100) % 10)), (('0') + (((1922 % 100) / 10) % 10)), (('0') + ((1922 % 100) % 10)), ']', '\000'}; 
#line 325 "CMakeCUDACompilerId.cu"
const char *info_platform = ("INFO:platform[Windows]"); 
#line 326
const char *info_arch = ("INFO:arch[x64]"); 
#line 331
const char *info_language_dialect_default = ("INFO:dialect_default[03]"); 
#line 347 "CMakeCUDACompilerId.cu"
int main(int argc, char *argv[]) 
#line 348
{ 
#line 349
int require = 0; 
#line 350
require += (info_compiler[argc]); 
#line 351
require += (info_platform[argc]); 
#line 353
require += (info_version[argc]); 
#line 356 "CMakeCUDACompilerId.cu"
require += (info_simulate[argc]); 
#line 359 "CMakeCUDACompilerId.cu"
require += (info_simulate_version[argc]); 
#line 361 "CMakeCUDACompilerId.cu"
require += (info_language_dialect_default[argc]); 
#line 362
(void)argv; 
#line 363
return require; 
#line 364
} 
#line 1 "CMakeCUDACompilerId.cudafe1.stub.c"
#define _NV_ANON_NAMESPACE _GLOBAL__N__27_CMakeCUDACompilerId_cpp1_ii_bd57c623
#ifdef _NV_ANON_NAMESPACE
#endif
#pragma pack()
#line 1 "CMakeCUDACompilerId.cudafe1.stub.c"
#include "CMakeCUDACompilerId.cudafe1.stub.c"
#line 1 "CMakeCUDACompilerId.cudafe1.stub.c"
#undef _NV_ANON_NAMESPACE
