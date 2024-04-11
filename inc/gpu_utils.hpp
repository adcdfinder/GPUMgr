#ifndef GPU_UTILS_HPP_
#define GPU_UTILS_HPP_

#include <functional> // for std::function

//stream id
#define gmgrStream int

//ploicy_type
#define single_strm 1
#define multi_strm 2
#define multi_strm_por 3

//callback type
#define cb_rt_type 1
#define cb_beORgeneral_type 2//策略1 2以及策略3的Best-effort Kernels都应是此种类型

//sync or Async
#define Sync true
#define Async false

typedef void *cuStreamPtr_t;
#define CUSTREAM_TYPE_SIZE 8

// Args for kernel
typedef void *cuKrnArgPtr_t;

typedef std::function<void(cuKrnArgPtr_t, void *, int blocksize = 1 , int blocknum = 1)> GPU_Kernel_Function; // TODO: add used_resource as args of GPU_Kernel_Function
typedef std::function<void(void)> GPU_Post_Function;
typedef void *GPU_Krn_Func_DataPtr;

#endif