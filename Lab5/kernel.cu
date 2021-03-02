#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <wb.h>

#define NUM_BINS 4096

#define CUDA_CHECK(ans)                                                   \
	    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
	bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
			file, line);
		if (abort)
			exit(code);
	}
}

__global__ void binning(unsigned int* input_array, unsigned int* bin_array, int input_length, int bin_num) {
	
	//int i = blockDim.x*blockIdx.x + threadIdx.x;
	//int stride = blockDim.x*gridDim.x;

	////__shared__ unsigned int share_mem[NUM_BINS];
	////share_mem[threadIdx.x] = 0;
	////__syncthreads();


	//while (i < bin_num) {
	//	unsigned int position = input_array[i];
	//	if (position < NUM_BINS) {
	//		atomicAdd(&bin_array[position], 1);
	//	}
	//	i += stride;
	//}
	//__syncthreads();

	//
	////bin_array[threadIdx.x] = share_mem[threadIdx.x];
	//

	__shared__ unsigned int temp[NUM_BINS];
	temp[threadIdx.x] = 0;
	__syncthreads();

	int i = threadIdx.x;
	int stride = blockDim.x;

	for (i; i < input_length; i+= stride) {
		atomicAdd(&temp[input_array[i]], 1);
	}
	__syncthreads();

	int k = threadIdx.x;
	for (k; k < NUM_BINS; k += stride) {
		bin_array[k] = temp[k];
		//if (temp[k] <= 127)
		//	bin_array[k] = temp[k];
		//else
		//	bin_array[k] = 127;
	}
	//bin_array[threadIdx.x] = temp[threadIdx.x];

}

__global__ void limmiting(unsigned int* bin_array, int bin_num) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	if (i < bin_num) {
		if (bin_array[i] > 127) {
			bin_array[i] = 127;
		}
	}
}

int main(int argc, char *argv[]) {
	wbArg_t args;
	int inputLength;
	unsigned int *hostInput;
	unsigned int *hostBins;
	unsigned int *deviceInput;
	unsigned int *deviceBins;

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 0),
		&inputLength, "Integer");
	hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The input length is ", inputLength);
	wbLog(TRACE, "The number of bins is ", NUM_BINS);

	for (int i = 0; i < NUM_BINS; i++) {
		hostBins[i] = 0;
	}

	wbTime_start(GPU, "Allocating GPU memory.");
	// TODO: Allocate GPU memory here
	cudaMalloc((void**)&deviceInput, inputLength*sizeof(unsigned int));
	cudaMalloc((void**)&deviceBins, NUM_BINS*sizeof(unsigned int));

	CUDA_CHECK(cudaDeviceSynchronize());
	wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Copying input memory to the GPU.");
	// TODO: Copy memory to the GPU here
	cudaMemcpy(deviceInput, hostInput, inputLength*sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceBins, hostBins, NUM_BINS*sizeof(unsigned int), cudaMemcpyHostToDevice);

	CUDA_CHECK(cudaDeviceSynchronize());
	wbTime_stop(GPU, "Copying input memory to the GPU.");

	// Launch kernel
	// ----------------------------------------------------------
	wbLog(TRACE, "Launching kernel");
	wbTime_start(Compute, "Performing CUDA computation");

	// TODO: Perform kernel computation here
	int thread_num = 256;
	int block_num = (NUM_BINS*2 + inputLength + thread_num - 1) / thread_num;

	binning<<<block_num, thread_num>>>(deviceInput, deviceBins, inputLength, NUM_BINS);

	limmiting << <block_num, thread_num >> >(deviceBins, NUM_BINS);


	// You should call the following lines after you call the kernel.
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output memory to the CPU");
	// TODO: Copy the GPU memory back to the CPU here

	cudaMemcpy(hostBins, deviceBins, NUM_BINS*sizeof(unsigned int), cudaMemcpyDeviceToHost);

	CUDA_CHECK(cudaDeviceSynchronize());
	wbTime_stop(Copy, "Copying output memory to the CPU");

	wbTime_start(GPU, "Freeing GPU Memory");
	// TODO: Free the GPU memory here
	cudaFree(deviceBins);
	cudaFree(deviceInput);

	wbTime_stop(GPU, "Freeing GPU Memory");

	// Verify correctness
	// -----------------------------------------------------
	wbSolution(args, hostBins, NUM_BINS);

	free(hostBins);
	free(hostInput);

#if LAB_DEBUG
	system("pause");
#endif

	return 0;
}
