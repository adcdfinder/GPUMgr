#ifndef QUICKTEST_GPU_MGR_NODE_HPP_
#define QUICKTEST_GPU_MGR_NODE_HPP_

#include <mutex>
#include <condition_variable>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int32.hpp"
#include "gpu_mgr/gpu_mgr.hpp"
#include "gpu_mgr_node.hpp"

#define test_mutex_cond_var 1
#define test_complete_flags 0
#define test_set_Used_Resource 0
#define test_submit 0

int bbbbb = 666;
int *cf_ptr = nullptr;
std::mutex *cm_ptr;
std::condition_variable *cv_ptr;
std::mutex m;
std::condition_variable cv;//条件变量
void krn_func0(void *data, int idx)
{
  printf("Krn Func data is %d idx is %d\n", *((int *)data), idx);

// test pass mutex cond_var
#if test_mutex_cond_var
  std::lock_guard<std::mutex> cv_lock(*cm_ptr);
  // std::lock_guard<std::mutex> cv_lock(m);
  printf("Notifying\n");
  cv_ptr->notify_one();//唤醒一个wait thread
  // cv.notify_one(&cv_lock);
  printf("Notified\n");
#endif
}
//该测试并没有测试post_func
void post_func0(void)
{
  printf("Post Func global var is %d\n", bbbbb);
}

class simplesub : public rclcpp::Node
{
public:
  rclcpp::CallbackGroup::SharedPtr cbg_sub_trig_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr sub_trig_;
  std::mutex* cvm_ptr;
  std::condition_variable* cv__ptr;
  std::shared_ptr<GpuMgr> g_mgr;

  simplesub(std::mutex* m_ptr,std::condition_variable* c_ptr,std::shared_ptr<GpuMgr> gm)
      : Node("simplesub"),cvm_ptr(m_ptr),cv__ptr(c_ptr),g_mgr(gm)
  {
    cbg_sub_trig_ = this->create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);

    auto sub_trig_loop_opt = rclcpp::SubscriptionOptions();
    sub_trig_loop_opt.callback_group = cbg_sub_trig_;
    //订阅 trig
    //为了测试，订阅“while_loop”
    sub_trig_ = this->create_subscription<std_msgs::msg::Int32>(
        "trig",
        rclcpp::QoS(10),
        std::bind(
            &simplesub::trig_cb,
            this,
            std::placeholders::_1),
        sub_trig_loop_opt);
  }
  //接收到订阅的topic，激活该函数
  void trig_cb(const std_msgs::msg::Int32::ConstSharedPtr msg){
    // std::lock_guard<std::mutex> lck(*cvm_ptr);
    // cv__ptr->notify_one();
    int a = 2333;
    // int stream_id = -1;
    GPU_Operation g_op(0, krn_func0, post_func0, &a);
    RCLCPP_INFO(
        this->get_logger(), "Trig\n");
    // (*g_mgr).submit(&g_op);
    GPU_Operation g_op1(0, krn_func0, post_func0, &a);
    // g_mgr->submit(&g_op, &stream_id);
    g_mgr->submit(&g_op);
    RCLCPP_INFO(
        this->get_logger(), "submit g_op \n");
    g_mgr->setComplete(0);
    // g_mgr->submit(&g_op1, &stream_id);//为了测试，临时改变了submit函数
    g_mgr->submit(&g_op1);
    RCLCPP_INFO(
        this->get_logger(), "submit g_op1 \n");
    g_mgr->setComplete(1);
  }


};

void
test_gpu_mgr_node(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::executors::MultiThreadedExecutor executor;//创建一个ROS2上的执行器

  auto gpu_mgr_node = std::make_shared<GpuMgrNode>();//创建一个gpumgr node
  int a = 2333;
  GPU_Operation g_op(0, krn_func0, post_func0, &a);

// TOTEST: w/o CUDA kernel
// submit [ + ]
// pass mutex cond_var  [ + ]
// pass complete_flags_ [ + ]
// set Used Resource [ + ]
// postfunc run [ + ]

// test pass complete_flags
#if test_complete_flags
    cf_ptr = gpu_mgr_node->getCompFlagsPtr();//return complete_flags_
    *(cf_ptr + 1) = 233;//cf_ptr[1]
    printf("cf is %d\n", gpu_mgr_node->getCompFlagAt(1));//return complete_flags_[1]
#endif

// test set Used Resource
#if test_set_Used_Resource
    g_op.setUsedResource(0, 1, 1, 0);
#endif

// test submit
#if test_submit
    // int stream_id = -1;
    gpu_mgr_node->submit(&g_op);
#endif

// test pass mutex cond_var
#if test_mutex_cond_var
  cm_ptr = gpu_mgr_node->getGpuCvMutex();
  cv_ptr = gpu_mgr_node->getGpuCondVar();//传递全局条件变量
  std::shared_ptr<GpuMgr> gm =  gpu_mgr_node;//将子类复制给父类对象
  auto simplesub_node = std::make_shared<simplesub>(cm_ptr,cv_ptr,gm);
  executor.add_node(simplesub_node);
  // gpu_mgr_node->submit(&g_op);
#endif

// test cv vec and mutex vec
#if 1
  std::mutex* cm5 = gpu_mgr_node->getStrmCvMutex(5);
  std::mutex* cv5 = gpu_mgr_node->getStrmCvMutex(5);
  (void)cm5;
  (void)cv5;
#endif 
  // printf("now exec add gpu_mgr_node!\n");
  executor.add_node(gpu_mgr_node);
  // printf("now exec spin!\n");
  executor.spin();
  // printf("shutdown!\n");
  rclcpp::shutdown();
}

#endif