// NVRTC predefines CUDA vector types and size_t; no toolkit headers are needed.
// Emit only one store-callback signature in each LTO IR compilation.
#ifndef CUPY_FFT_CALLBACK_TYPE
#error "CUPY_FFT_CALLBACK_TYPE must select one supported output dtype"
#endif

#if CUPY_FFT_CALLBACK_TYPE == 0
using CallbackOutput = float;
using CallbackReal = float;
#define CUPY_FFT_CALLBACK_NAME cufftJITCallbackStoreReal
#define CUPY_FFT_CALLBACK_COMPLEX 0
#elif CUPY_FFT_CALLBACK_TYPE == 1
using CallbackOutput = double;
using CallbackReal = double;
#define CUPY_FFT_CALLBACK_NAME cufftJITCallbackStoreDoubleReal
#define CUPY_FFT_CALLBACK_COMPLEX 0
#elif CUPY_FFT_CALLBACK_TYPE == 2
using CallbackOutput = float2;
using CallbackReal = float;
#define CUPY_FFT_CALLBACK_NAME cufftJITCallbackStoreComplex
#define CUPY_FFT_CALLBACK_COMPLEX 1
#elif CUPY_FFT_CALLBACK_TYPE == 3
using CallbackOutput = double2;
using CallbackReal = double;
#define CUPY_FFT_CALLBACK_NAME cufftJITCallbackStoreDoubleComplex
#define CUPY_FFT_CALLBACK_COMPLEX 1
#else
#error "Unsupported callback output dtype"
#endif

using CallbackOffset = unsigned long long;

// Preserve C++ linkage: nvmath uses cuFFT's default mangled callback symbol.
__device__ void CUPY_FFT_CALLBACK_NAME(
    void* data_out, CallbackOffset offset, CallbackOutput value,
    void* user_info, void* unused)
{
    const CallbackReal divisor = *static_cast<const CallbackReal*>(user_info);
#if CUPY_FFT_CALLBACK_COMPLEX
    value.x /= divisor;
    value.y /= divisor;
#else
    value /= divisor;
#endif
    static_cast<CallbackOutput*>(data_out)[offset] = value;
}
