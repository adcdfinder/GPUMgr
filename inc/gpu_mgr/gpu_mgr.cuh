#ifndef GPU_MGR_CUH__
#define GPU_MGR_CUH__
#ifndef define GPU_GPU
#define GPU_GPU
#endif

#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <functional> // for std::function

#include "gpu_utils.hpp"

cudaDeviceProp prop;
extern cudaStream_t *m_stream;
extern cudaStream_t *m_streamWithlow;
extern cudaStream_t *m_streamWithhigh;
extern std::function<void (std::mutex*,std::condition_variable*)> notifyGpuMgr;

extern void GPU_Init(int policy_type,int stream_num,std::function<void (std::mutex*,std::condition_variable*)> notify_callback);
extern void GPU_Deinit(void);
void CUDART_CB launchFuncCallback(void *data);
extern cuStreamPtr_t getCuStreamPtr(int stream_idx);
extern cuStreamPtr_t getCuLowStreamPtr(int stream_idx);
extern cuStreamPtr_t getCuHighStreamPtr();
extern void LaunchHostFunc(bool IsSync,
                    GPU_Post_Function *post_func,
                    std::mutex *host_CvMetux,
                    std::condition_variable *host_Cv,
                    void *stream_ptr,
                    void *g_op);
const char *GetRuntimeError(cudaError_t error);
#ifdef GPU_GPU
extern void executeAffinityTask(GPU_Operation *g_op, int sm_id);
#endif
#endif