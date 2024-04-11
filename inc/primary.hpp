
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int32.hpp"
class PublisherNode : public rclcpp::Node
{
public:
  PublisherNode()
  : Node("PublisherNode"), count_(0)
  {
    publisher_ = this->create_publisher<MSG_TYPE_INT32>("topic", 10);
    auto timer_callback =
      [this]() -> void {
        auto message = MSG_TYPE_INT32();
        // message.data = "Hello World! " + std::to_string(this->count_++);
        message.data = this->count_++;
        if (this->count_ == 4)
        {
          // rclcpp::shutdown();
          deinit_data();
        }
        else if (this->count_ > 4)
        {
          RCLCPP_INFO(this->get_logger(), 
            "\nTimer Publishing %d\n",message.data);
        }
        
        else
        {
          // copyToGpu(message.data, message.data);
          // copyToGpu(message.data+10, message.data);
          RCLCPP_INFO(this->get_logger(), 
            "\nTimer Publishing %d\n a[%d] is %d \na[%d] is %d\n", 
            message.data, 
            message.data,
            readResult(message.data),
            message.data+10,
            readResult(message.data+10)
          );
          
          this->publisher_->publish(message);
        }
      };
    timer_ = this->create_wall_timer(500ms, timer_callback);
  }

private:
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<MSG_TYPE_INT32>::SharedPtr publisher_;
  size_t count_;
};

class DualThreadedNode : public rclcpp::Node
{
public:
  DualThreadedNode()
  : Node("DualThreadedNode")
  {
    /* These define the callback groups
     * They don't really do much on their own, but they have to exist in order to
     * assign callbacks to them. They're also what the executor looks for when trying to run multiple threads
     */
    callback_group_subscriber1_ = this->create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive);
    callback_group_subscriber2_ = this->create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive);
    callback_group_subscriber3_ = this->create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive);
    callback_group_subscriber4_ = this->create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive);


    // Each of these callback groups is basically a thread
    // Everything assigned to one of them gets bundled into the same thread
    auto sub1_opt = rclcpp::SubscriptionOptions();
    sub1_opt.callback_group = callback_group_subscriber1_;
    auto sub2_opt = rclcpp::SubscriptionOptions();
    sub2_opt.callback_group = callback_group_subscriber2_;
    auto sub3_opt = rclcpp::SubscriptionOptions();
    sub3_opt.callback_group = callback_group_subscriber3_;
    auto sub4_opt = rclcpp::SubscriptionOptions();
    sub4_opt.callback_group = callback_group_subscriber4_;

    subscription1_ = this->create_subscription<MSG_TYPE_INT32>(
      "topic",
      rclcpp::QoS(10),
      // std::bind is sort of C++'s way of passing a function
      // If you're used to function-passing, skip these comments
      std::bind(
        &DualThreadedNode::subscriber1_cb,  // First parameter is a reference to the function
        this,                               // What the function should be bound to
        std::placeholders::_1),             // At this point we're not positive of all the
                                            // parameters being passed
                                            // So we just put a generic placeholder
                                            // into the binder
                                            // (since we know we need ONE parameter)
      sub1_opt);                  // This is where we set the callback group.
                                  // This subscription will run with callback group subscriber1

    subscription2_ = this->create_subscription< MSG_TYPE_INT32 >(
      "topic",
      rclcpp::QoS(10),
      std::bind(
        &DualThreadedNode::subscriber2_cb,
        this,
        std::placeholders::_1),
      sub2_opt);


      subscription3_ = this->create_subscription< MSG_TYPE_INT32 >(
      "loop",
      rclcpp::QoS(10),
      std::bind(
        &DualThreadedNode::subscriber3_cb,
        this,
        std::placeholders::_1),
      sub3_opt);

      subscription4_ = this->create_subscription< MSG_TYPE_INT32 >(
      "trig",
      rclcpp::QoS(10),
      std::bind(
        &DualThreadedNode::subscriber4_cb,
        this,
        std::placeholders::_1),
      sub4_opt);
  }

private:
  /**
   * Simple function for generating a timestamp
   * Used for somewhat ineffectually demonstrating that the multithreading doesn't cripple performace
   */
  std::string timing_string()
  {
    rclcpp::Time time = this->now();
    return std::to_string(time.nanoseconds());
  }

  /**
   * Every time the Publisher publishes something, all subscribers to the topic get poked
   * This function gets called when Subscriber1 is poked (due to the std::bind we used when defining it)
   */
  void subscriber1_cb(const MSG_TYPE_INT32::ConstSharedPtr msg)
  {
    auto message_received_at = timing_string();
    auto recv_int = msg->data;
    PlusOnGpu(recv_int,recv_int);
    // Extract current thread
    RCLCPP_INFO(
      this->get_logger(), "Sub1 Recv %d a[%d] is %d", recv_int, recv_int, readResult(recv_int));

  }

  /**
   * This function gets called when Subscriber2 is poked
   * Since it's running on a separate thread than Subscriber 1, it will run at (more-or-less) the same time!
   */
  void subscriber2_cb(const MSG_TYPE_INT32::ConstSharedPtr msg)
  {
    auto message_received_at = timing_string();
    auto recv_int = msg->data;
    PlusOnGpu1(recv_int+10,recv_int);
    // Extract current thread
    RCLCPP_INFO(
      this->get_logger(), "Sub2 Recv %d a[%d] is %d", recv_int, recv_int+10, readResult(recv_int+10));
  }

  void subscriber3_cb(const MSG_TYPE_INT32::ConstSharedPtr msg)
  {
    auto message_received_at = timing_string();
    auto recv_int = msg->data;
    cnt_ ++;
    RCLCPP_INFO(
      this->get_logger(), "Sub3 Recv %d", recv_int++);
    while (rclcpp::ok(nullptr))
    {
      RCLCPP_INFO(
      this->get_logger(), "Loop Wait at %d", recv_int++);
      std::unique_lock<std::mutex> cv_lock(cv_mutex);
      cv.wait(cv_lock);
      RCLCPP_INFO(
      this->get_logger(), "Triggered and run");
    }
    
  }
  void subscriber4_cb(const MSG_TYPE_INT32::ConstSharedPtr msg)
  {
    auto message_received_at = timing_string();
    auto recv_int = msg->data;

    RCLCPP_INFO(
      this->get_logger(), "Sub4 Recv %d", recv_int++);
    
    cv.notify_one();

  }
  rclcpp::CallbackGroup::SharedPtr callback_group_subscriber1_;
  rclcpp::CallbackGroup::SharedPtr callback_group_subscriber2_;
  rclcpp::CallbackGroup::SharedPtr callback_group_subscriber3_;
  rclcpp::CallbackGroup::SharedPtr callback_group_subscriber4_;
  rclcpp::Subscription<MSG_TYPE_INT32>::SharedPtr subscription1_;
  rclcpp::Subscription<MSG_TYPE_INT32>::SharedPtr subscription2_;
  rclcpp::Subscription<MSG_TYPE_INT32>::SharedPtr subscription3_;
  rclcpp::Subscription<MSG_TYPE_INT32>::SharedPtr subscription4_;
  rclcpp::Publisher<MSG_TYPE_INT32>::SharedPtr publisher_;
  std::mutex cv_mutex;
  std::condition_variable cv;
  uint32_t cnt_=0;
};
