#include <unistd.h>
#include <string>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int32.hpp"
#include "mini.cuh"
#include "gpu_mgr/gpu_mgr.hpp"
#include "gpu_utils.hpp"
class SimpleNode : public rclcpp::Node
{
public:
  // rclcpp::CallbackGroup::SharedPtr cbg_sub_trig_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr sub_trig0_;
  
  GpuMgr *g_mgr = NULL;

  SimpleNode()
      : Node("SimpleNode")
  {
    mini_Init();//准备内存

    auto cbg_sub_trig0_ = this->create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    auto sub_trig0_opt = rclcpp::SubscriptionOptions();
    sub_trig0_opt.callback_group = cbg_sub_trig0_;

    sub_trig0_ = this->create_subscription<std_msgs::msg::Int32>(
        "s0",
        rclcpp::QoS(10),
        std::bind(
            &SimpleNode::trig_cb0,
            this,
            std::placeholders::_1),
        sub_trig0_opt);
  }
  
  void trig_cb0(const std_msgs::msg::Int32::ConstSharedPtr msg)
  {
    RCLCPP_INFO(
        this->get_logger(), "Trig s0\n");
    //获取一个stream
    gmgrStream stream = g_mgr->gmgrCreateStream(cb_beORgeneral_type);

    mini_t m_mini0 = {10, msg->data};
    //选择callback type 并传入kunc PostFunc data
    g_mgr->gmgrLaunchKAndPFunc(cb_beORgeneral_type,mini_AddXandY0,mini_PostFunc0,(void *)&m_mini0,stream);
    
    //选择callback type 并传入kunc PostFunc data
    g_mgr->gmgrLaunchKAndPFunc(cb_rt_type,mini_AddXandY1,mini_PostFunc1,(void *)&m_mini0,stream);

    RCLCPP_INFO(
        this->get_logger(), "Mini0 %d + %d \nUsre Thread END!\n", m_mini0.x, m_mini0.y);
  }

  void setGpuMgr(GpuMgr *gm){
    g_mgr = gm;
  }

  ~SimpleNode(void)
  {
    mini_Deinit();
  }
};