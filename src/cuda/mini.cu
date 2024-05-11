#include "../../inc/mini_type.hpp"
#include "util.cu"
#include "../../inc/gpu_mgr/gpu_mgr.cuh"
#include <stdio.h>
#include <sstream>
#include <string>
#include <cuda_runtime.h>
// #define TEST_DEBUG
#define MINI_N 10


uint32_t *h_sum;
uint32_t *d_sum;
int m = 1;
float occupy = 0.0;

cudaError_t cudaRet = cudaSuccess;
//ROS2 需要gpu执行的操作
__device__ volatile int g_stopFlag = 0;
int stopFlag = 0;

__global__ void noiseKernel(uint32_t smID){
  if(smID == getSMID()){
    // printf("exit on %d .\n", smID);
    asm("exit;");
  }
  while(!g_stopFlag){
    //  MySleep(1); // sleep for 1 millisecond
    continue;
  }

}

// extern __shared__ int shared[];
__global__ void vectorAddPartial(const float* A, const float* B, float* C, int offset, int sliceSize)
{
   int i = blockDim.x * blockIdx.x + threadIdx.x + offset;
    // shared[i] = i;
    // __syncthreads();
   if (i < offset + sliceSize)
   {
       C[i] = A[i] + B[i];
   }else{
       C[i] = A[i] - B[i];
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
  // MySleep(50000);
  for(int i  = 0; i < 500;i++){
    continue;
  }
  // printf("Current task is running on %d sm .", getSMID());
  // busySleep(50000000); // busy sleep about 40ms
  // MySleep(500);//sleep 500ms
  return;
}

// __global__ void normalTask(uint32_t *block, uint32_t* timestamp, uint32_t *smId){
//   int index = blockIdx.x * blockDim.x + threadIdx.x;
//   block[index] = blockIdx.x;
//   timestamp[index] = clock();
//   smId[index] = getSMID();

// }

__global__ void testKernel(float *data, int size)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // printf("Normal task is running on %d sm .\n", getSMID());

  for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            data[tid] += i * j;
        }
  }



  return;


}

__global__ void testAffinityKernel(float *data, int size)
{
  if(blockIdx.x == (gridDim.x - 1)){
    g_stopFlag = 1;
  }

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // printf("Affinity task is running on %d sm .\n", getSMID());

  for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            data[tid] += i * j;
        }
  }

 
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

  float *hostData = new float[blocksize * numblocks];
  float* deviceData;
  cudaMalloc(&deviceData, blocksize * numblocks * sizeof(float));

  testKernel<<< numblocks, blocksize, 0 ,*((cudaStream_t *)stream)>>>(deviceData, 10000);
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

  float *hostData = new float[blocksize * numblocks];
  float* deviceData;
  cudaMalloc(&deviceData, blocksize * numblocks * sizeof(float));

  testAffinityKernel<<< numblocks, blocksize,  0 ,*((cudaStream_t *)stream)>>>(deviceData, 10000);
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
float get_gpu_utilization() {
    FILE* pipe = popen("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits", "r");
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }

    char buffer[128];
    float gpu_utilization = -1.0f;

    if (fgets(buffer, 128, pipe) != nullptr) {
       printf("the result of nvidia-smi : %s", buffer);
        // Read the output and parse GPU utilization
        std::istringstream ss(buffer);
        std::string token;
        std::getline(ss, token, ',');  // Ignore the index
        std::getline(ss, token, ',');  // Get utilization
        gpu_utilization = std::stof(token);
    }

    pclose(pipe);
    return gpu_utilization;
}

void checkError(){
  cudaError_t cudaE = cudaGetLastError();
  if(cudaE != cudaSuccess){
    printf("Error while do cuda related: %s", cudaGetErrorString(cudaE));
  }

 
}

void launchNoiseTest(bool bPolicyApplied , float * elapse){
  // stream prepare
  cudaStream_t *noiseStream = (cudaStream_t *)malloc(  sizeof(cudaStream_t *));
  cudaStream_t *workStream = (cudaStream_t *)malloc(  sizeof(cudaStream_t *));
  int priority_high, priority_low;

  cudaDeviceProp prop;
  int device;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, 0);

  
  cudaDeviceGetStreamPriorityRange(&priority_low,&priority_high);
  // create streams with highest and lowest available priorities
  // cudaStream_t st_high, st_low;

  cudaStreamCreateWithPriority(&(workStream[0]),cudaStreamDefault, priority_low);
  cudaStreamCreateWithPriority(&(noiseStream[0]),cudaStreamDefault, priority_high);

  // stream preparation done;
  // int m = 9;
  int smNum = prop.multiProcessorCount;
  int threadsPerBlock = (prop.maxThreadsPerBlock - 256)/m;
  float currentOcc = 0.0;

  //for record the elapse
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  cudaEventQuery(start);
  // printf("Executing kernel");

  // Kernerl operations
  startNoiseKernel(threadsPerBlock, smNum * m , noiseStream);
  mini_AddXandY1_Affinity(nullptr, workStream, 1024, 8);
  if(!bPolicyApplied){
    cudaDeviceSynchronize();
  }

  mini_AddXandY1(nullptr, workStream, 256, 12);
  if(!bPolicyApplied){

    cudaDeviceSynchronize();
  }

  mini_AddXandY1(nullptr, workStream, 256, 12);
  
  currentOcc = get_gpu_utilization();
  if(bPolicyApplied){
    if(occupy == 0.0) occupy = currentOcc;
    if(currentOcc > occupy) {
      occupy = currentOcc;
      m+= 1;
    }
  }

  cudaDeviceSynchronize();

  resetNoiseFlag();


  // //for record the elapse


  // //for test

  // startNoiseKernel(1024, smNum, noiseStream);
  // // startNoiseKernel(threadsPerBlock, smNum, m_stream);

  // mini_AddXandY1_Affinity(nullptr, workStream, 1024, 1);
  // mini_AddXandY1(nullptr, workStream, 86, 14);
  // // mini_AddXandY1(nullptr, m_stream , 256, 1);
  
  // cudaDeviceSynchronize();
  // resetNoiseFlag();



  //end test

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(elapse, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // *elapse = occupy;

}

void launchSliceTest(float * elapse){


  cudaStream_t *workStream = (cudaStream_t *)malloc(  sizeof(cudaStream_t *));
  cudaEvent_t start, stop;
  int numSlices = 4;
  int totalSize = 1024;
 
    // 查询流的优先级范围
   int leastPriority, greatestPriority;
   cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
 
  cudaStreamCreateWithPriority(&(workStream[0]),cudaStreamDefault, greatestPriority);
  
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  cudaEventQuery(start);
 
 
  mini_AddXandY1_Affinity(nullptr, workStream ,totalSize / numSlices, numSlices);

  cudaDeviceSynchronize();
 
   // 复制结果从设备到主机内存

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(elapse, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

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
  float elapse = 0.0;
  launchNoiseTest(prop, m_stream_noise, m_stream, false, &elapse  );
  // return 0;
  printf("the max threads perblock is %d . \n", threadsPerBlock);
  printf("the max threads per sm is %d . \n", threadsPerSM);
  printf("Elapse time is %g . \n",elapse );
  // startNoiseKernel(1024, smNum, m_stream_noise );
  // startNoiseKernel(threadsPerBlock, smNum, m_stream);

  // mini_AddXandY1_Affinity(nullptr, m_stream , 1024, 1);
  // mini_AddXandY1(nullptr, m_stream , 86, 14);
  // mini_AddXandY1(nullptr, m_stream , 256, 1);
  
  // cudaDeviceSynchronize();
  resetNoiseFlag();


  return 0;


}
#endif