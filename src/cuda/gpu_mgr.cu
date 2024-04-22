#ifndef GPU_MGR_CU__
#define GPU_MGR_CU__
#endif

#define  GPU_GPU

#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <iostream>
#include <unistd.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#include "gpu_mgr/gpu_mgr.cuh"
#include "gpu_utils.hpp"
#include "mini.cuh"
#include "gpu_operation.hpp"

#ifdef GPU_GPU
using namespace std;

std::vector<GPU_Operation *> waitQ ;

struct HostFuncData
{
  bool IsNotify;
  std::function<void(void)> *Post_Function;
  std::mutex *gopt_cv_mutex;
  std::condition_variable *gopt_cv;
  cudaStream_t *gopt_stm;
  void *gpu_opt;
};


cudaStream_t *m_stream;                // Only initialized by GPU_Init at gpu_mgr.cu
cudaStream_t *m_streamWork;
cudaStream_t *m_streamNoise;
cudaStream_t *m_streamWithlow;
cudaStream_t *m_streamWithhigh;

std::function<void (std::mutex*,std::condition_variable*)> notifyGpuMgr; // Only assigned by GPU_Init at gpu_mgr.cu
cudaDeviceProp prop;
bool bExecuting = false; 

void GPU_Init(int policy_type,int stream_num,std::function<void (std::mutex*,std::condition_variable*)> notify_callback)
{
  stream_num = 2;

  // Create CUDA stream array dynamically

  m_stream = (cudaStream_t *)malloc(stream_num * sizeof(cudaStream_t *));
  m_streamWork = (cudaStream_t *)malloc(stream_num * sizeof(cudaStream_t *));
  m_streamNoise = (cudaStream_t *)malloc(sizeof(cudaStream_t *));

  // Create CUDA streams no priority
  for (int i = 0; i < stream_num; i++)
    cudaStreamCreate(&(m_stream[i]));

  // get the range of stream priorities for this device
  int priority_high, priority_low;
  cudaDeviceGetStreamPriorityRange(&priority_low,&priority_high);
  // create streams with highest and lowest available priorities
  // cudaStream_t st_high, st_low;
  cudaStreamCreateWithPriority(&(m_streamNoise[0]),cudaStreamDefault, priority_high);

  cudaStreamCreateWithPriority(&(m_streamWork[0]),cudaStreamDefault, priority_low);

  int device;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  notifyGpuMgr = notify_callback;

  bExecuting = false;
  #if 0
  // For debug
  printf("Size of cudaStream_t is %ld\n", sizeof(cudaStream_t));
  # endif
}

void GPU_Deinit(void)
{
  cudaDeviceReset();
}

void CUDART_CB launchFuncCallback(void *data){
  pid_t pid = getpid();
  printf("launchFuncCallback %lu: Run launchFuncCallback!\n",pid);
  HostFuncData tran_data = *((HostFuncData *)data);

  // Free the memory space
  free(data);

  // Unique entry to notify GPU Manager GPU operation done on stream_idx
  // 通知gpu mgr gpu opt已完成 解锁阻塞的线程
  //只有同步方式需要执行
  if(tran_data.IsNotify){
    //sleep一会，以保证gpu_mgr 已经处于wait
    usleep(3000);//3ms
    notifyGpuMgr(tran_data.gopt_cv_mutex,tran_data.gopt_cv);
  }
  else{//只有异步方式执行
    //run post Func
    if(tran_data.Post_Function){
      
      (*(tran_data.Post_Function))();
    }
    //执行到此处时，表示一个GPU_Operation全部被执行完毕
    //delete new的GPU_Operation对象
    printf("launchFuncCallback: run delete gpu opt!\n");
    delete tran_data.gpu_opt;
  }
  
  printf("launchFuncCallback %lu: launchFuncCallback end!\n",pid);
}

cuStreamPtr_t getWorkStream(){
  return (void *)&(m_streamWork);
}

cuStreamPtr_t getNoiseStream(){
  return (void *)&(m_streamNoise);
}

cuStreamPtr_t getCuStreamPtr(int stream_idx)
{
  // TODO: Range Check
  return (void *)&(m_stream[stream_idx]);
}

cuStreamPtr_t getCuLowStreamPtr(int stream_idx)
{
  // TODO: Range Checkm_stream
  return (void *)&(m_streamWithlow[stream_idx]);
}

cuStreamPtr_t getCuHighStreamPtr()
{
  // TODO: Range Checkm_stream
  return (void *)&(m_streamWithhigh[0]);
}

void LaunchHostFunc(bool IsSync,
                    GPU_Post_Function *post_func,
                    std::mutex *gopt_CvMetux,
                    std::condition_variable *gopt_Cv,
                    void *stream_ptr,
                    void *g_op){

  // printf("LaunchHostFunc: run launch Host Func!\n");
  //准备需要传递给launchFuncCallback函数的数据
  HostFuncData *hostfuncdata = (HostFuncData *)malloc(sizeof(HostFuncData));
  hostfuncdata->IsNotify = IsSync;
  hostfuncdata->Post_Function = post_func;
  hostfuncdata->gopt_cv_mutex = gopt_CvMetux;
  hostfuncdata->gopt_cv = gopt_Cv;
  hostfuncdata->gpu_opt = g_op;
  cudaStream_t *s_ptr;
  cudaError_t error;
  // printf("LaunchHostFunc: run cudaLaunchHostFunc!\n");
  if(stream_ptr == NULL){
    hostfuncdata->gopt_stm = NULL;
    error = cudaLaunchHostFunc(0, launchFuncCallback, hostfuncdata);//default stream
    if (error != cudaSuccess){
      printf("LaunchHostFunc: cudaLaunchHostFunc Error:%s\n", GetRuntimeError(error));
    }
  }
  else{
    s_ptr = (cudaStream_t *)stream_ptr;
    hostfuncdata->gopt_stm = s_ptr;
    error = cudaLaunchHostFunc(*s_ptr, launchFuncCallback, hostfuncdata);
    if (error != cudaSuccess){
      printf("LaunchHostFunc: cudaLaunchHostFunc Error:%s\n", GetRuntimeError(error));
    }
  }
  // printf("LaunchHostFunc: launch Host Func end!\n");
}

const char *GetRuntimeError(cudaError_t error)
{
    if (error != cudaSuccess)
    {
        return cudaGetErrorString(error);
    }
    else
        return NULL;
}

void submitTowaitQe(GPU_Operation *g_op)
{
  if(!bExecuting){
    waitQ.push_back(g_op);
  }else{
    g_op->run(multi_strm_por, m_streamWork);
  }
}

void launchNoiseWorkLoad(){
  // get noise kernel threads
  int maxBlockSizeInWait = 0;
  int smNum = prop.multiProcessorCount;
  int threadsPerSM = (prop.maxThreadsPerMultiProcessor - maxBlockSizeInWait) / 2;

  // startNoiseKernel(threadsPerSM, smNum, (void *)getNoiseStream());
}

void executeAffinityTask(void *g_op, int sm_id){
  GPU_Operation * ga_op = (GPU_Operation *)g_op;
  launchNoiseWorkLoad();

}

// void launchNoiseTest(){
//   int smNum = prop.multiProcessorCount;
//   int threadsPerBlock = prop.maxThreadsPerBlock - 500;
//   int threadsPerSM = prop.maxThreadsPerMultiProcessor;

//   startNoiseKernel(threadsPerBlock, smNum, getNoiseStream());
//   mini_AddXandY1_Affinity(nullptr, getWorkStream(), 512, 8);
//   mini_AddXandY1(nullptr, getWorkStream(), 86, 14);
//   mini_AddXandY1(nullptr, getWorkStream(), 256, 1);
  
//   // cudaDeviceSynchronize();
//   resetNoiseFlag();


  
// }

#else


cudaStream_t *m_stream;                // Only initialized by GPU_Init at gpu_mgr.cu
cudaStream_t *m_streamWithlow;
cudaStream_t *m_streamWithhigh;
std::function<void (std::mutex*,std::condition_variable*)> notifyGpuMgr; // Only assigned by GPU_Init at gpu_mgr.cu

void GPU_Init(int policy_type,int stream_num,std::function<void (std::mutex*,std::condition_variable*)> notify_callback)
{
  #ifdef GPU_GPU
  stream_num = 2;
  #endif
  // Create CUDA stream array dynamically
  m_stream = (cudaStream_t *)malloc(stream_num * sizeof(cudaStream_t *));
  m_streamWithlow = (cudaStream_t *)malloc(stream_num * sizeof(cudaStream_t *));
  m_streamWithhigh = (cudaStream_t *)malloc(sizeof(cudaStream_t *));

  // Create CUDA streams no priority
  for (int i = 0; i < stream_num; i++)
    cudaStreamCreate(&(m_stream[i]));

  // get the range of stream priorities for this device
  int priority_high, priority_low;
  cudaDeviceGetStreamPriorityRange(&priority_low,&priority_high);
  // create streams with highest and lowest available priorities
  cudaStream_t st_high, st_low;
  cudaStreamCreateWithPriority(&(m_streamWithhigh[0]),cudaStreamDefault, priority_high);

  for (int i = 0; i < stream_num; i++)
    cudaStreamCreateWithPriority(&(m_streamWithlow[i]),cudaStreamDefault, priority_low);

  notifyGpuMgr = notify_callback;
  #if 0
  // For debug
  printf("Size of cudaStream_t is %ld\n", sizeof(cudaStream_t));
  # endif
}

void GPU_Deinit(void)
{
  cudaDeviceReset();
}

void CUDART_CB launchFuncCallback(void *data){
  pid_t pid = getpid();
  printf("launchFuncCallback %lu: Run launchFuncCallback!\n",pid);
  printf("launchFuncCallback %lu: Run launchFuncCallback!\n",pid);
  HostFuncData tran_data = *((HostFuncData *)data);

  // Free the memory space
  free(data);

  // Unique entry to notify GPU Manager GPU operation done on stream_idx
  // 通知gpu mgr gpu opt已完成 解锁阻塞的线程
  //只有同步方式需要执行
  if(tran_data.IsNotify){
    //sleep一会，以保证gpu_mgr 已经处于wait
    usleep(3000);//3ms
    notifyGpuMgr(tran_data.gopt_cv_mutex,tran_data.gopt_cv);
  }
  else{//只有异步方式执行
    //run post Func
    (*(tran_data.Post_Function))();
    //执行到此处时，表示一个GPU_Operation全部被执行完毕
    //delete new的GPU_Operation对象
    printf("launchFuncCallback: run delete gpu opt!\n");
    delete tran_data.gpu_opt;
  }
  
  printf("launchFuncCallback %lu: launchFuncCallback end!\n",pid);
}

cuStreamPtr_t getCuStreamPtr(int stream_idx)
{
  // TODO: Range Check
  return (void *)&(m_stream[stream_idx]);
}

cuStreamPtr_t getCuLowStreamPtr(int stream_idx)
{
  // TODO: Range Checkm_stream
  return (void *)&(m_streamWithlow[stream_idx]);
}

cuStreamPtr_t getCuHighStreamPtr()
{
  // TODO: Range Checkm_stream
  return (void *)&(m_streamWithhigh[0]);
}

void LaunchHostFunc(bool IsSync,
                    GPU_Post_Function *post_func,
                    std::mutex *gopt_CvMetux,
                    std::condition_variable *gopt_Cv,
                    void *stream_ptr,
                    void *g_op){

  // printf("LaunchHostFunc: run launch Host Func!\n");
  //准备需要传递给launchFuncCallback函数的数据
  HostFuncData *hostfuncdata = (HostFuncData *)malloc(sizeof(HostFuncData));
  hostfuncdata->IsNotify = IsSync;
  hostfuncdata->Post_Function = post_func;
  hostfuncdata->gopt_cv_mutex = gopt_CvMetux;
  hostfuncdata->gopt_cv = gopt_Cv;
  hostfuncdata->gpu_opt = g_op;
  cudaStream_t *s_ptr;
  cudaError_t error;
  // printf("LaunchHostFunc: run cudaLaunchHostFunc!\n");
  if(stream_ptr == NULL){
    hostfuncdata->gopt_stm = NULL;
    error = cudaLaunchHostFunc(0, launchFuncCallback, hostfuncdata);//default stream
    if (error != cudaSuccess){
      printf("LaunchHostFunc: cudaLaunchHostFunc Error:%s\n", GetRuntimeError(error));
    }
  }
  else{
    s_ptr = (cudaStream_t *)stream_ptr;
    hostfuncdata->gopt_stm = s_ptr;
    error = cudaLaunchHostFunc(*s_ptr, launchFuncCallback, hostfuncdata);
    if (error != cudaSuccess){
      printf("LaunchHostFunc: cudaLaunchHostFunc Error:%s\n", GetRuntimeError(error));
    }
  }
  // printf("LaunchHostFunc: launch Host Func end!\n");
}

const char *GetRuntimeError(cudaError_t error)
{
    if (error != cudaSuccess)
    {
        return cudaGetErrorString(error);
    }
    else
        return NULL;
}



#endif
