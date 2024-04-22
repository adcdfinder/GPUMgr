#include "../../inc/mini_type.hpp"
#include "util.cu"
#include "../../inc/gpu_mgr/gpu_mgr.cuh"
#include <stdio.h>
// #define TEST_DEBUG
#define MINI_N 10


uint32_t *h_sum;
uint32_t *d_sum;
//ROS2 需要gpu执行的操作
__device__ volatile int g_stopFlag = 0;
int stopFlag = 0;

__global__ void noiseKernel(uint32_t smID){
  if(smID == getSMID()){
    printf("exit on %d .\n", smID);
    asm("exit;");
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
  printf("success ");
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
  printf("Current task is running on %d sm .", getSMID());
  // busySleep(50000000); // busy sleep about 40ms
  // MySleep(500);//sleep 500ms
  return;
}

__global__ void testKernel()
{
  printf("Current task is running on %d sm .\n", getSMID());

}

__global__ void testAffinityKernel()
{
  if(blockIdx.x == (gridDim.x - 1)){
    g_stopFlag = 1;
  }

  printf("Affinity task is running on %d sm .\n", getSMID());

  MySleep(500);//sleep 500ms
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

void mini_AddXandY1(void *ka_ptr, void *stream, int blocksize = 1 , int numblocks = 1)
{
  // cudaStream_t *stream_ptr = (cudaStream_t *)stream;


  printf("testkernel is executing. /n");

  testKernel<<< numblocks, blocksize, 0 ,*((cudaStream_t *)stream)>>>();
  // cudaDeviceSynchronize();
  // *d_sum = 10;
  return;


  // uint32_t x = ((mini_t *)ka_ptr)->x;
  // uint32_t y = ((mini_t *)ka_ptr)->y;

  // kAddXandY1<<<blocksize, numblocks, 0, *stream_ptr>>>(d_sum, x, y);

  // cuMemCpyAsyncDtoH(h_sum, d_sum,sizeof(uint32_t)*MINI_N,*stream_ptr);
  // // printf("mini_AddXandY1: gpu mgr AsyncDtoH end!\n");
}

void mini_AddXandY1_Affinity(void *ka_ptr, void *stream, int blocksize = 1 , int numblocks = 1)
{
  // cudaStream_t *stream_ptr = (cudaStream_t *)stream;

  printf("testkernel is executing. /n");

  testAffinityKernel<<< numblocks, blocksize,  0 ,*((cudaStream_t *)stream)>>>();
  // cudaDeviceSynchronize();
  // *d_sum = 10;
  return;
  
}

void startNoiseKernel(int blocksize, int numblocks, void *stream){
  cudaStream_t *stream_ptr = (cudaStream_t *)stream;
  noiseKernel<<<numblocks, blocksize,  0, (*(cudaStream_t *)stream_ptr)>>>(5);
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

void launchNoiseTest(cudaDeviceProp prop, void *noiseStream, void *workStream){
  int m = 9;
  int smNum = prop.multiProcessorCount;
  int threadsPerBlock = (prop.maxThreadsPerBlock - 256)/m;
  int threadsPerSM = prop.maxThreadsPerMultiProcessor;

  printf("Executing kernel");
  startNoiseKernel(threadsPerBlock, smNum, noiseStream);
  mini_AddXandY1_Affinity(nullptr, workStream, 512, 8);
  mini_AddXandY1(nullptr, workStream, 86, 14);
  mini_AddXandY1(nullptr, workStream, 256, 1);
  
  // cudaDeviceSynchronize();
  resetNoiseFlag();



}
#ifdef TEST_DEBUG
int main(){
  
  cudaStream_t *m_stream = (cudaStream_t *)malloc(  sizeof(cudaStream_t *));
  cudaStream_t *m_stream_noise = (cudaStream_t *)malloc(  sizeof(cudaStream_t *));
  int priority_high, priority_low;

  cudaDeviceProp prop;
  int device;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, 0);

  
  cudaDeviceGetStreamPriorityRange(&priority_low,&priority_high);
  // create streams with highest and lowest available priorities
  // cudaStream_t st_high, st_low;

  cudaStreamCreateWithPriority(&(m_stream[0]),cudaStreamDefault, priority_low);
  cudaStreamCreateWithPriority(&(m_stream_noise[0]),cudaStreamDefault, priority_high);

  printf("the highest priority is %d . \n", priority_high);
  printf("the lowest priority is %d . \n", priority_low);
  
  int smNum = prop.multiProcessorCount;
  int threadsPerBlock = prop.maxThreadsPerBlock - 500;
  int threadsPerSM = prop.maxThreadsPerMultiProcessor;
  printf("Compute capability: %d.%d\n", prop.major, prop.minor);
  printf("SM num: %d.\n", smNum);

  // return 0;
  printf("the max threads perblock is %d . \n", threadsPerBlock);
  printf("the max threads per sm is %d . \n", threadsPerSM);
  // startNoiseKernel(threadsPerBlock, smNum, m_stream_noise );

  // mini_AddXandY1_Affinity(nullptr, m_stream , 512, 8);
  mini_AddXandY1(nullptr, m_stream , 86, 14);
  // mini_AddXandY1(nullptr, m_stream , 256, 1);
  
  // cudaDeviceSynchronize();
  resetNoiseFlag();


  return 0;


}
#endif