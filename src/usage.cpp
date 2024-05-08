// Copyright 2020 Open Source Robotics Foundation, Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <unistd.h>
#include <iostream> 
#include "rclcpp/rclcpp.hpp"//ROS Client Library for C++
#include "std_msgs/msg/int32.hpp"

// #include "pub_timer_int32_node.hpp"
// #include "pub_notimer_int32_node.hpp"
// #include "gpu_mgr_node.hpp"
#include "pub_timer_usage_node.hpp"


using namespace std::chrono_literals;//C++时间库

#ifndef DATA_TYPE
#define DATA_TYPE uint32_t
#endif
//函数位于cuda/selfadd.cu
// extern void init_data(void);
// extern void deinit_data(void);
// extern int readResult(int index);
// extern void copyToGpu(int index, int num);
// extern void PlusOnGpu(int index, int num);
// extern void PlusOnGpu1(int index, int num);

#define MSG_TYPE_INT32 std_msgs::msg::Int32

// typedef std::function<void(void*)> GPU_Wrapper_Function;
// typedef void* GPU_Wrapper_Arg_Data;
// typedef std::pair<GPU_Wrapper_Function,GPU_Wrapper_Arg_Data> GPU_Operation;
// typedef std::vector<GPU_Operation> GPU_Operation_Queue;
// GPU_Operation_Queue gpu_ops_;

// class dig{
//   public:
//     uint32_t data;
// };

// void test(void* data){

//   printf("%d\n",((dig*)data)->data );
// }

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  // init_data();//Initialize host data

  // You MUST use the MultiThreadedExecutor to use, well, multiple threads
  rclcpp::executors::MultiThreadedExecutor executor;
  // auto pubnode = std::make_shared<PublisherNode>();

  //触发simple node
  auto pub_timer_node = std::make_shared<PubTimerUsageNode>("s0","PubTimer_mini_node");//创建并返回1个指定类型的智能指针
  //触发gmr的while loop
  // auto pub_notimer_node = std::make_shared<PubNoTimerInt32Node>("s0","PubNoTimer_whileloop_node");
  // auto pub_notimer_node = std::make_shared<PubTimerInt32Node>("while_loop","PubNoTimer_whileloop_node");
  // auto gpu_mgr_node = std::make_shared<GpuMgrNode>();
  // auto subnode = std::make_shared<DualThreadedNode>();  // This contains BOTH subscriber callbacks.
                                                        // They will still run on different threads
                                                        // One Node. Two callbacks. Two Threads
  executor.add_node(pub_timer_node);
  // executor.add_node(pub_notimer_node);
  // executor.add_node(subnode);
  executor.spin();//Create a default single-threaded executor and spin the specified node
  rclcpp::shutdown();
  return 0;
}