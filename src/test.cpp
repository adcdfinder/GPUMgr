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
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int32.hpp"

// #include "quick_test/test_gpu_mgr_node.hpp"
// #include "quick_test/test_gpu_operation.hpp"
#include "quick_test/test_simple_node.hpp"
// #include "quick_test/test_gpu_mgr.hpp"

using namespace std::chrono_literals;



int main(int argc, char * argv[])
// int main()
{
  // test_gpu_operation(); // passed

  // 测试了gpu opt提交，以及mgr的大致工作
  // test_gpu_mgr_node(argc, argv);

  // 只创建了一个gpu mgr对象
  // test_gpu_mgr();

  //测试了整个gmr的功能，包含真正的kernel
  test_simple_node(argc, argv);
  return 0;

}