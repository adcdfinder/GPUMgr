#ifndef GPU_MGR_HPP_
#define GPU_MGR_HPP_
#define GPU_GPU

#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <map>
#include <iostream>
#include <pthread.h>

#include "gpu_operation.hpp"
#include "gpu_utils.hpp"
#include "gpu_mgr/gpu_mgr.cuh"

using namespace std;

class GpuMgr
{
  //这里应该将一些属性设置为 protected,但是为了测试方便，就全部public
public:
  //同步方式要使用的条件变量
  std::mutex cv_mutex_;
  std::condition_variable cv_;//阻塞mgr thread
  int stream_num = 0;//stream数量
  int policy_type;//gmgr所使用的策略
  bool IsSync = Sync;//sync or async
  std::mutex submitTowait_mutex_;
  std::vector<GPU_Operation *> waiting_gpu_ops_;//策略1 2和3的Best-effort Kernels都放入此
  std::mutex submitTowait_Rt_mutex_;
  std::vector<GPU_Operation *> waiting_gpu_Rtops_;// RT队列
  std::mutex currentStmid_mutex_;
  int currentStmid=0;//策略2的stream id
  std::mutex currentLowStmid_mutex_;
  int currentLowStmid=0;//策略3的stream id
  pthread_t tid;//运行mgr主体的thread id
  bool isAllowToLoop_ = true;

public:
  GpuMgr(int p_type,bool IsSync,int s_num);
  ~GpuMgr();

  void* getWorkStreamA();
  void* getNoiseStreamA();

  //对用户开放的接口
  //获取一个stream id  对用户而言的“create”
  gmgrStream gmgrCreateStream(int cb_type)
  {
    if(cb_type != cb_rt_type && cb_type != cb_beORgeneral_type){
      printf("cb type error! please choose 'cb_rt_type' or 'cb_beORgeneral_type'\n");
      return -1;
    }

    if(cb_type == cb_rt_type && this->policy_type != multi_strm_por){
      printf("The current policy does not support rt callback type!\n");
      return -1;
    }
    gmgrStream stream;
    switch (this->policy_type)
    {
    case single_strm:
      stream = 0;
      break;

    case multi_strm:
      stream = currentStmid;
      currentStmid_mutex_.lock();
      currentStmid++;
      if(currentStmid == stream_num)
        currentStmid = 0;
      currentStmid_mutex_.unlock();
      break;

    case multi_strm_por:
      if(cb_type == cb_rt_type){
        stream = 0;
      }
      else{
        stream = currentLowStmid;
        currentLowStmid_mutex_.lock();
        currentLowStmid++;
        if(currentLowStmid == stream_num)
          currentLowStmid = 0;
        currentLowStmid_mutex_.unlock();
        break;
      }
      break;

    default:
      break;
    }
    return stream;
  }

  //user thread调用该函数，提交Operation到等待队列
  void submitTowaitQe(GPU_Operation *g_op)
  {
    switch (this->policy_type)
    {
    case single_strm:
      // printf("submitTowaitQe: start submitTowaitQe!\n");
      submitTowait_mutex_.lock();
      waiting_gpu_ops_.push_back(g_op);
      submitTowait_mutex_.unlock();
      // printf("submitTowaitQe: submitTowaitQe end!\n");
      break;

    case multi_strm:
      // printf("submitTowaitQe multi_strm: start submitTowaitQe!\n");
      submitTowait_mutex_.lock();
      waiting_gpu_ops_.push_back(g_op);
      submitTowait_mutex_.unlock();
      // printf("submitTowaitQe multi_strm: submitTowaitQe end!\n");
      break;

    case multi_strm_por:
      if(g_op->getcbType() == cb_rt_type){
        // printf("submitTowaitQe multi_strm_por cb_rt_type: start submitTowaitQe!\n");
        submitTowait_Rt_mutex_.lock();
        waiting_gpu_Rtops_.push_back(g_op);
        submitTowait_Rt_mutex_.unlock();
        // printf("submitTowaitQe multi_strm_por cb_rt_type: submitTowaitQe end!\n");
      }
      else{
        // printf("submitTowaitQe multi_strm_por cb_beORgeneral_type: start submitTowaitQe!\n");
        submitTowait_mutex_.lock();
        waiting_gpu_ops_.push_back(g_op);
        submitTowait_mutex_.unlock();
        // printf("submitTowaitQe multi_strm_por cb_beORgeneral_type: submitTowaitQe end!\n");
      }
      break;

    default:
      break;
    }
    
    //只有同步方式运行
    if(IsSync){
      // User thread Sleeping wait for complete
      printf("User thread Waiting!\n");
      std::mutex* cm_ptr = g_op->getcvMutex();
      std::condition_variable* cv_ptr = g_op->getcv();
      std::unique_lock<std::mutex> result_lck(*cm_ptr);
      cv_ptr->wait(result_lck);
      cout << "User thread is notifyed can Run!" << endl;
      //run post Func
      g_op->runPostfunc();
      //执行到此处时，表示一个GPU_Operation全部被执行完毕
      //delete new的GPU_Operation对象
      printf("submitTowaitQe: run delete gpu opt!\n");
      delete g_op;
    }

    //usre thread launch Operation 后，暂停一会，防止多个thread过快的竞争锁（1ms左右）
    // usleep(10000);//sleep 10ms
  }

  //使mgr thread wait
  void gpu_mgr_wait(){
    cout << "gpu_mgr thread waiting!" << endl;
    std::unique_lock<std::mutex> cv_lock(cv_mutex_);
    cv_.wait(cv_lock); //阻塞当前线程
    cout << "gpu_mgr thread is notifyed can Run!" << endl;
  }

  //gpu mgr thread从wait队列中取操作并launch
  void findAndLaunchWaitGpuOps()
  {
    vector<GPU_Operation *>::iterator g_op_it;
    GPU_Operation *g_op = NULL;
    switch (this->policy_type)
    {
    case single_strm:
      if(waiting_gpu_ops_.empty())//等待队列为空，直接返回
        return;
      printf("mgr thread get a opt int single_strm policy!\n");
      g_op_it = waiting_gpu_ops_.begin();//从wait队列头部取操作
      g_op = (GPU_Operation *)(*g_op_it);
      //launch到default stream
      // printf("findAndLaunchWaitGpuOps: run KFunc!\n");
      g_op->run(single_strm,NULL);//launch kernel之后，就要launch hostFunc
      // printf("findAndLaunchWaitGpuOps: runed KFunc!\n");
      //launch hostFunc
      // printf("findAndLaunchWaitGpuOps: launch HostFunc!\n");
      LaunchHostFunc(IsSync,g_op->getPostFunc(),g_op->getcvMutex(),g_op->getcv(),NULL,(void *)g_op);
      // printf("findAndLaunchWaitGpuOps: launched HostFunc!\n");
      //只有同步方式时执行
      if(IsSync){
        //mgr thread launch host后，立马阻塞，防止因为等待submitTowait_mutex_而错过notify
        gpu_mgr_wait();
      }
      submitTowait_mutex_.lock();
      waiting_gpu_ops_.erase(g_op_it);
      submitTowait_mutex_.unlock();
      break;

    case multi_strm:
      if(waiting_gpu_ops_.empty())//等待队列为空，直接返回
        return;
      printf("mgr thread get a opt int multi_strm policy!\n");
      g_op_it = waiting_gpu_ops_.begin();//从wait队列头部取操作
      g_op = (GPU_Operation *)(*g_op_it);
      //launch to stream
      g_op->run(multi_strm,getCuStreamPtr(g_op->getstreamid()));
      // printf("Launch Kernel stream is %d\n",g_op->getstreamid());
      //launch hostFunc
      LaunchHostFunc(IsSync,g_op->getPostFunc(),g_op->getcvMutex(),g_op->getcv(),
                              getCuStreamPtr(g_op->getstreamid()),(void *)g_op);

      //只有同步方式时执行
      if(IsSync){
        //mgr thread launch host后，立马阻塞，防止因为等待submitTowait_mutex_而错过notify
        gpu_mgr_wait();
      }
      submitTowait_mutex_.lock();
      waiting_gpu_ops_.erase(g_op_it);
      submitTowait_mutex_.unlock();
      break;

    case multi_strm_por:
      if(!waiting_gpu_Rtops_.empty())//RT等待队列不为空，那么就取rt kernel
      {
        printf("gpu mgr thread get a RT opt!\n");
        g_op_it = waiting_gpu_Rtops_.begin();//从wait队列头部取操作
        g_op = (GPU_Operation *)(*g_op_it);
        //launch到stream
        #ifdef GPU_GPU 
          if(g_op->getIsAffinity()){
            executeAffinityTask((void *)g_op, 1); 
          }else{

            g_op->run(multi_strm_por,getWorkStream());
          }
          
        #else
          g_op->run(multi_strm_por,getCuHighStreamPtr());
        #endif
        //launch hostFunc
        LaunchHostFunc(IsSync,g_op->getPostFunc(),g_op->getcvMutex(),g_op->getcv(),
                              getCuHighStreamPtr(),(void *)g_op);
        //只有同步方式时执行
        if(IsSync){
          //mgr thread launch host后，立马阻塞，防止因为等待submitTowait_Rt_mutex_而错过notify
          gpu_mgr_wait();
        }
        submitTowait_Rt_mutex_.lock();
        waiting_gpu_Rtops_.erase(g_op_it);
        submitTowait_Rt_mutex_.unlock();
        break;
      }
      else{//RT队列为空
        if(!waiting_gpu_ops_.empty())//RT队列为空，但BE队列不为空
        {
          printf("RT opt is empty, gpu mgr thread get a BE opt!\n");
          g_op_it = waiting_gpu_ops_.begin();//从wait队列头部取操作
          g_op = (GPU_Operation *)(*g_op_it);
          //launch to stream
          g_op->run(multi_strm_por,getCuLowStreamPtr(g_op->getstreamid()));
          // printf("Launch Kernel Low stream is %d\n",g_op->getstreamid());
          //launch hostFunc
          LaunchHostFunc(IsSync,g_op->getPostFunc(),g_op->getcvMutex(),g_op->getcv(),
                            getCuLowStreamPtr(g_op->getstreamid()),(void *)g_op);
          
          //只有同步方式时执行
          if(IsSync){
            //mgr thread launch host后，立马阻塞，防止因为等待submitTowait_mutex_而错过notify
            gpu_mgr_wait();
          }
    
          submitTowait_mutex_.lock();
          waiting_gpu_ops_.erase(g_op_it);
          submitTowait_mutex_.unlock();
          // printf("Once gpu_mgr thread end!\n");
          break;
        }
        else{//两个队列都是空
          // printf("RT opt and BE opt are empty!\n");
          return;
        }
      }
      break;

    default:
      break;
    }
    printf("findAndLaunchWaitGpuOps: end!\n");
  }

  //对用户开放的接口
  //user thread调用的接口，作用：launch Operation到gpu_mgr
  void gmgrLaunchKAndPFunc(int cb_type,GPU_Kernel_Function g_krn_func,GPU_Post_Function g_post_func,
                GPU_Krn_Func_DataPtr data_ptr, bool isAffinity = false , int blocksize = 1, int numblocks = 1 , int affinitySMid = 0){
    if(cb_type != cb_rt_type && cb_type != cb_beORgeneral_type){
      printf("cb type error! please choose 'cb_rt_type' or 'cb_beORgeneral_type'\n");
      return;
    }

    if(cb_type == cb_rt_type && this->policy_type != multi_strm_por){
      printf("The current policy does not support rt callback type!\n");
      return;
    }

    if(affinitySMid < 0 || affinitySMid >= stream_num){
      printf("stream Error! please choose correct stream!\n");
      return;
    }
    
    //使用new，保证user thread退出后，Operation依旧可用
    GPU_Operation *gpu_opt = new GPU_Operation(cb_type, g_krn_func, g_post_func, (void *)data_ptr, isAffinity, blocksize, numblocks);
    // g_op(cb_type, g_krn_func, g_post_func, (void *)data_ptr);

    //user thread只管提交操作到wait队列，然后就阻塞
    submitTowaitQe(gpu_opt);
    // findAndExecuteReadyPostFunc();//还可改进，思路：不需要每次都循环去寻找可执行的post func
  }

  //停止 gpu mgr Thread
  void gmgr_end(){
    isAllowToLoop_ = false;//结束loop thread
    int res;
    //创建一个空指针
    void * thread_result;
    //等待 tid 线程执行完成，并用 thread_result 指针接收该线程的返回值
    res = pthread_join(tid, &thread_result);
    if (res != 0) {
        printf("等待线程失败!\n");
    }
    cout << "gpu mgr thread end success!" << endl;
    printf("%s\n", (char*)thread_result);
  }
};

// gpu mgr Thread运行的函数
void* gmgr_start(void *data){
  GpuMgr *gmgr = (GpuMgr *)data;
  cout << "gpu mgr thread Start!" << endl;
  // int loop_num = 0;
  while (gmgr->isAllowToLoop_)//mgr一直运行
  {
    // cout << "gpu mgr thread loop at " << loop_num << endl;

    // Find and Launch GPU operations from wait queue
    gmgr->findAndLaunchWaitGpuOps();

    // cout << "gpu mgr thread End loop at " << loop_num++ << endl;
  }
}

//gpu mgr初始化：策略、同步或异步、stream数目
GpuMgr::GpuMgr(int p_type,bool IsSync,int s_num) : policy_type(p_type), IsSync(IsSync), stream_num(s_num)
{
  if(p_type != single_strm && p_type != multi_strm && p_type != multi_strm_por){
    printf("ploicy type error! please choose 'single_strm' or 'multi_strm' or 'multi_strm_por'\n");
    return;
  }
  if(p_type != single_strm && s_num <= 0){
    printf("stream num error!\n");
    return ;
  }
  if(p_type == single_strm && s_num < 0){
    printf("stream num error!\n");
    return;
  }

  // 同步方式会使用的notify函数
  std::function<void (std::mutex*,std::condition_variable*)> notifyWhileLoop = [&](std::mutex* cv_mutex,std::condition_variable* cv){

    //notify gpu_mgr thread
    std::lock_guard<std::mutex> cv_lock_(cv_mutex_);
    cv_.notify_one();
    cout << "notify gpu_mgr thread" << endl;

    //notify user thread
    std::lock_guard<std::mutex> cv_lock(*cv_mutex);
    (*cv).notify_one();
    cout << "notify user thread" << endl;
  };
  
  //初始化指定数量的stream
  GPU_Init(p_type,stream_num,notifyWhileLoop);

  //create and run thread
  //参数依次是：创建的线程id，线程参数，调用的函数，传入的函数参数
  int ret = pthread_create(&tid, NULL, gmgr_start, this);
  if (ret != 0){
    cout << "pthread_create error: error_code=" << ret << endl;
  }
  cout << "GPU MGR Init Success!" << endl;
}

GpuMgr::~GpuMgr()
{
  // Destroy CUDA context
  GPU_Deinit();
}

void* GpuMgr::getWorkStreamA(){
  return (void*)getWorkStream();
}

void* GpuMgr::getNoiseStreamA(){
  return (void*)getNoiseStream();
}


#endif