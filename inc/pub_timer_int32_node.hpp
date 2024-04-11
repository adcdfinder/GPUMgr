
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int32.hpp"

#define MSG_TYPE__ std_msgs::msg::Int32

using namespace std::chrono_literals;

class PubTimerInt32Node : public rclcpp::Node
{
public:
  PubTimerInt32Node(const std::string topic,const std::string node_name)
  :Node(node_name),count_(0)
  {

    publisher_ = this->create_publisher<MSG_TYPE__>(topic, 10);

    auto timer_callback =
      [this]() -> void {
        auto message = MSG_TYPE__();
        // message.data = "Hello World! " + std::to_string(this->count_++);
        message.data = this->count_++;
        this->publisher_->publish(message);//发布消息
        RCLCPP_INFO(this->get_logger(), 
            "\nTimer Publishing %d\n",message.data);//打印信息
      };
    timer_ = this->create_wall_timer(50ms, timer_callback);//500ms触发一次定时的callback
  }
private:
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<MSG_TYPE__>::SharedPtr publisher_;
  size_t count_=0;
};