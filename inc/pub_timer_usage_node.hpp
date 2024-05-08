
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int32.hpp"
#include "mini.cuh"
#define MSG_TYPE__ std_msgs::msg::Int32

using namespace std::chrono_literals;

class PubTimerUsageNode : public rclcpp::Node
{
public:
  PubTimerUsageNode(const std::string topic,const std::string node_name)
  :Node(node_name),count_(0)
  {

    // publisher_ = this->create_publisher<MSG_TYPE__>(topic, 10);

    auto timer_callback =
      [this]() -> void {

      float currentOcc = 0.0 ;
      currentOcc = get_gpu_utilization();
        RCLCPP_INFO(this->get_logger(), 
            "\nGPU Utilizization is %f\n", currentOcc);//打印信息
      };
    timer_ = this->create_wall_timer(500ms, timer_callback);//500ms触发一次定时的callback
  }
private:
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<MSG_TYPE__>::SharedPtr publisher_;
  size_t count_=0;
};