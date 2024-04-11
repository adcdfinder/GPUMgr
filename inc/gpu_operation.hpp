#ifndef GPU_OPERATION_HPP_
#define GPU_OPERATION_HPP_

#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <string>
#include <iostream>

#include "gpu_utils.hpp"

class GPU_Operation
{
private:
  gmgrStream stream_idx_;//launch stream的索引
  GPU_Kernel_Function g_krn_func_;
  GPU_Post_Function g_post_func_;
  GPU_Krn_Func_DataPtr g_data_ptr_;
  //每一个GPU_Operation都有一个条件变量，从而实现同步的方式
  std::mutex cv_mutex_;
  std::condition_variable cv_;
  int cb_type;//用户提交的kernel类型，RT BE
  int blocksize = 0 ;
  int numBlocks = 0 ;

public:
  GPU_Operation();
  GPU_Operation(
      int cb_type,
      GPU_Kernel_Function g_krn_func,
      GPU_Post_Function g_post_func,
      GPU_Krn_Func_DataPtr data_ptr,
      gmgrStream stream_id);
  ~GPU_Operation();
  void setGpuKernelFunction(GPU_Kernel_Function g_krn_func);
  void setGpuPostFunction(GPU_Post_Function g_post_func);
  void setGpuKernelFunctionDataPtr(GPU_Krn_Func_DataPtr data_ptr);
  void run(int ploicy_type,void *stm_ptr);
  void runPostfunc();
  GPU_Post_Function *getPostFunc();
  void setstreamid(int stm_id);
  gmgrStream getstreamid();
  std::mutex *getcvMutex();
  std::condition_variable *getcv();
  int getcbType();
};

GPU_Operation::GPU_Operation(
    int cb_type,
    GPU_Kernel_Function g_krn_func,
    GPU_Post_Function g_post_func,
    GPU_Krn_Func_DataPtr data_ptr,
    gmgrStream stream_id)
    : cb_type(cb_type),
      g_krn_func_(g_krn_func),
      g_post_func_(g_post_func),
      g_data_ptr_(data_ptr),
      stream_idx_(stream_id)
{
  if(cb_type != cb_rt_type && cb_type != cb_beORgeneral_type){
    printf("cb type error! please choose 'cb_rt_type' or 'cb_beORgeneral_type'\n");
  }
}

GPU_Operation::~GPU_Operation()
{
}

void GPU_Operation::setGpuKernelFunction(GPU_Kernel_Function g_krn_func)
{
  this->g_krn_func_ = g_krn_func;
}

void GPU_Operation::setGpuPostFunction(GPU_Post_Function g_post_func)
{
  this->g_post_func_ = g_post_func;
}

void GPU_Operation::setGpuKernelFunctionDataPtr(GPU_Krn_Func_DataPtr data_ptr)
{
  this->g_data_ptr_ = data_ptr;
}

void GPU_Operation::run(int ploicy_type,void *stm_ptr)
{
  if(ploicy_type == single_strm){
    this->g_krn_func_(this->g_data_ptr_, 0);//使用default stream
  }
  else{
    this->g_krn_func_(this->g_data_ptr_, stm_ptr, this->blocksize, this->numBlocks);
  }
}

void GPU_Operation::runPostfunc(){
  this->g_post_func_();
}

GPU_Post_Function *GPU_Operation::getPostFunc(){
  return &g_post_func_;
}

int GPU_Operation::getcbType(){
  return cb_type;
}

std::mutex *GPU_Operation::getcvMutex(){
  return &cv_mutex_;
}
std::condition_variable *GPU_Operation::getcv(){
  return &cv_;
}

void GPU_Operation::setstreamid(int stm_id){
  stream_idx_ = stm_id;
}

gmgrStream GPU_Operation::getstreamid(){
  return stream_idx_;
}

void GPU_Operation::setBlockSize(int e_block_size){
  this.blocksize = e_block_size;
}

void GPU_Operation::setBlockNum(int e_block_num){
  this.numBlocks = e_block_num;
}

#endif