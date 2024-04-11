#ifndef GPU_MGR_NODE_HPP_
#define GPU_MGR_NODE_HPP_

#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <iostream> 
#include <pthread.h>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int32.hpp"

#include "gpu_operation.hpp"
#include "gpu_utils.hpp"
#include "gpu_mgr/gpu_mgr.hpp"

#define MSG_TYPE__ std_msgs::msg::Int32

using namespace std::chrono_literals;

class GpuMgrNode : public rclcpp::Node
{
public:
  GpuMgr *gpumgr;
public:
  GpuMgrNode(void)
      : Node("GPU_Mgr_Node")
  {
    //同步 策略3
    // gpumgr = new GpuMgr(multi_strm_por,Sync,10);
    //同步 策略2
    // gpumgr = new GpuMgr(multi_strm,Sync,10);
    //同步 策略1
    // gpumgr = new GpuMgr(single_strm,Sync,10);
    //异步 策略3
    gpumgr = new GpuMgr(multi_strm_por,Async,10);
    //异步 策略2
    // gpumgr = new GpuMgr(multi_strm,Async,10);
    //异步 策略1
    // gpumgr = new GpuMgr(single_strm,Async,10);
    
  }

  ~GpuMgrNode(void){
    delete gpumgr;
  }
  
};

#endif