#include "mini_type.hpp"
#include "util.cu"
#include "gpu_mgr/gpu_mgr.cuh"

#define MINI_N 10

uint32_t *h_sum;
uint32_t *d_sum;
//ROS2 需要gpu执行的操作
__device__ volatile int g_stopFlag = 0;
int stopFlag = 0;

__global__ void noiseKernel(uint32_t smID){
  if(smID == getSMID()){
    asm("exit;")
  }
  while(!g_stopFlag){
    MySleep(1); // sleep for 1 millisecond
  }

}

void syncNoiseKernel(){
  cudaError_t cudaStatus = cudaMemcpyToSymbol(g_stopFlag, &stopFlag, 0, cudaMemcpyHostToDevice);

}

void resetNoiseFlag(){
  stopFlag = 0;
  syncNoiseKernel();
  cudaDeviceSynchronize();
}

void stopNoiseFlag(){
  stopFlag = 1;
  syncNoiseKernel();
  cudaDeviceSynchronize();
}

__global__ void kAddXandY0(uint32_t *sum, uint32_t x, uint32_t y)
{
  sum[0] = x + y;
  // busySleep(50000000); // busy sleep about 40ms
  // MySleep(500);//sleep 500ms
  return;
}

__global__ void kAddXandY1(uint32_t *sum, uint32_t x, uint32_t y)
{
  sum[1] = x + y;
  // busySleep(50000000); // busy sleep about 40ms
  // MySleep(500);//sleep 500ms
  return;
}

void mini_AddXandY0(void *ka_ptr, void *stream)
{
  cudaStream_t *stream_ptr = (cudaStream_t *)stream;
  uint32_t x = ((mini_t *)ka_ptr)->x;
  uint32_t y = ((mini_t *)ka_ptr)->y;

  kAddXandY0<<<1, 1, 0, *stream_ptr>>>(d_sum, x, y);

  cuMemCpyAsyncDtoH(h_sum, d_sum,sizeof(uint32_t)*MINI_N,*stream_ptr);
  // printf("mini_AddXandY0: gpu mgr AsyncDtoH end!\n");
}

void mini_AddXandY1(void *ka_ptr, void *stream)
{
  cudaStream_t *stream_ptr = (cudaStream_t *)stream;
  uint32_t x = ((mini_t *)ka_ptr)->x;
  uint32_t y = ((mini_t *)ka_ptr)->y;

  kAddXandY1<<<1, 1, 0, *stream_ptr>>>(d_sum, x, y);

  cuMemCpyAsyncDtoH(h_sum, d_sum,sizeof(uint32_t)*MINI_N,*stream_ptr);
  // printf("mini_AddXandY1: gpu mgr AsyncDtoH end!\n");
}

int mini_GetResult(void)
{
  return *h_sum;
}

const char *MyGetRuntimeError(cudaError_t error)
{
    if (error != cudaSuccess)
    {
        return cudaGetErrorString(error);
    }
    else
        return NULL;
}

void mini_PostFunc0()
{
  //此处会报CUDA错误：CUDA API不允许在host func中执行
  // cudaError_t error;
  // printf("PostFunc run DtoH!\n");
  // error = cuMemCpyAsyncDtoH(h_sum, d_sum,sizeof(uint32_t)*MINI_N,stm);
  // if (error != cudaSuccess){
  //    printf("thread cudaOccupancyMaxActiveBlocksPerMultiprocessor Error:%s\n", MyGetRuntimeError(error));
  // }
  printf("h_sum is ");
  for (int i = 0; i < MINI_N; i++)
  {
    printf("%d ",h_sum[i]);
  }

  printf("\n");
}

void mini_PostFunc1()
{
  //此处会报CUDA错误：CUDA API不允许在host func中执行
  // cudaError_t error;
  // printf("PostFunc run DtoH!\n");
  // error = cuMemCpyAsyncDtoH(h_sum, d_sum,sizeof(uint32_t)*MINI_N,stm);
  // if (error != cudaSuccess){
  //    printf("thread cudaOccupancyMaxActiveBlocksPerMultiprocessor Error:%s\n", MyGetRuntimeError(error));
  // }
  printf("h_sum is ");
  for (int i = 0; i < MINI_N; i++)
  {
    printf("%d ",h_sum[i]);
  }

  printf("\n");
}

void mini_Init(void)
{

  // Allocate host memory
  cudaMallocHost((void **)&h_sum, sizeof(uint32_t) * MINI_N);

  // Allocate GPU memory
  cudaMalloc((void **)&d_sum, sizeof(uint32_t) * MINI_N);

  // Initialize var on host and GPU memory
  h_sum[9] = 666;
  cudaMemcpy(d_sum, h_sum, sizeof(uint32_t) *MINI_N, cudaMemcpyHostToDevice); // copy to gpu
}

void mini_Deinit(void){
  cudaFreeHost(h_sum);
}