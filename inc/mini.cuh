#ifndef MINI_CUH_
#define MINI_CUH_
#include <cuda.h>
#include <cuda_runtime.h>

#include "gpu_utils.hpp"

#include "mini_type.hpp"

extern void mini_AddXandY0(cuKrnArgPtr_t ka_ptr, void *stream_ptr);
extern void mini_AddXandY1(cuKrnArgPtr_t ka_ptr, void *stream_ptr, int, int);
extern void resetNoiseFlag();
extern void launchNoiseTest(cudaDeviceProp prop, void *noiseStream, void *workStream, bool, float *);
extern void mini_AddXandY1_Affinity(void *ka_ptr, void *stream, int blocksize , int numblocks );
extern void mini_Init();
extern void mini_PostFunc0();
extern void mini_PostFunc1();
extern int mini_GetResult(void);
extern void mini_Deinit(void);
extern void startNoiseKernel(int, int, void *);


#endif