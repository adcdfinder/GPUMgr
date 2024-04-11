#ifndef QUICKTEST_GPU_MGR_NODE_HPP_
#define QUICKTEST_GPU_MGR_NODE_HPP_

#include <mutex>
#include <condition_variable>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int32.hpp"
#include "gpu_mgr/gpu_mgr.hpp"
#include "gpu_mgr_node.hpp"
#include "simple_node.hpp"

void test_simple_node(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::executors::MultiThreadedExecutor executor;

  // 这里为了测试，直接将GpuMgr对象传入了测试node：simplenode中。但是真正在第三方项目引入动态连接库的时候，不会直接暴露GpuMgr
  // 可以通过虚基类的方式：在第三方项目中获取虚基类的对象，通过连接库暴露的接口获取一个GpuMgr对象，并将此赋予虚基类的对象
  // 第三方项目使用虚基类的对象
  auto gpu_mgr_node = std::make_shared<GpuMgrNode>();
  auto simplenode = std::make_shared<SimpleNode>();
  simplenode->setGpuMgr(gpu_mgr_node->gpumgr);//设置gpumgr

  executor.add_node(gpu_mgr_node);
  executor.add_node(simplenode);
  executor.spin();
  rclcpp::shutdown();
}

#endif