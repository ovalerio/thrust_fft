#include <cufft.h>
#include <cufftXt.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define checkCudaErrors(val)  __checkCudaErrors__ ( (val), #val, __FILE__, __LINE__ )
 
template <typename T>
inline void __checkCudaErrors__(T code, const char *func, const char *file, int line) 
{
    if (code) {
        fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n",
                file, line, (unsigned int)code, func);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

/********************************/
/* SCALE USING A CUFFT CALLBACK */
/********************************/
__device__ void scale_cufft_callback(
    void *dataOut,
    size_t offset,
    float2 element,
    void *callerInfo,
    void *sharedPtr)
{
    float2 output;

    output.x = element.x / 2;
    output.y = element.y / 2;

    ((float2*)dataOut)[offset] = output;
}

__device__
cufftCallbackStoreC d_storeCallbackPtr = scale_cufft_callback;

int main(void){

    const int N=2;

    // --- Setting up input device vector
    thrust::device_vector<float2> d_vec(N,make_cuComplex(1.0f,2.0f));

    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);

    // --- Preparing the callback
    cufftCallbackStoreC h_storeCallbackPtr;
    checkCudaErrors(cudaMemcpyFromSymbol(&h_storeCallbackPtr,
					d_storeCallbackPtr,
					sizeof(h_storeCallbackPtr)));

    // --- Associating the callback with the plan.
    cufftResult status = cufftXtSetCallback(plan,
				(void **)&h_storeCallbackPtr,
				CUFFT_CB_ST_COMPLEX,
				0);
    if (status == CUFFT_LICENSE_ERROR) {
	printf("License file was not found, out of date, or invalid.\n");
	exit(EXIT_FAILURE);
    } else {
	checkCudaErrors(status);
    }

    // --- Perform in-place direct Fourier transform
    checkCudaErrors(cufftExecC2C(plan, thrust::raw_pointer_cast(d_vec.data()),thrust::raw_pointer_cast(d_vec.data()), CUFFT_FORWARD));
    //thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(), scale_result((float)(2)));

    // --- Setting up output host vector
    thrust::host_vector<float2> h_vec(d_vec);

    for (int i=0; i<N; i++) printf("Element #%i: \t (%f, %f)\n",i,h_vec[i].x,h_vec[i].y);

    //Clean up
    checkCudaErrors(cufftDestroy(plan));
}
