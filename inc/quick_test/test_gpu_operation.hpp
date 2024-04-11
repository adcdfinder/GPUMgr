#ifndef QUICKTEST_GPU_OPERATION_HPP_
#define QUICKTEST_GPU_OPERATION_HPP_
#include "gpu_operation.hpp"

#define GPU_OP_UNDER_TEST
//这里只进行了简单的测试，没有真正的Kernel
int bbbb = 666;

void krn_func0(void* data,int idx){
    printf("Krn Func data is %d idx is %d\n",*((int*)data),idx);
}

void post_func0(void){
    // TODO： Combined test with gpu_mgr_node
    printf("Post Func global var is %d\n",bbbb);
}

void test_gpu_operation(void){
    int a = 2333;
    GPU_Operation g_op(0,krn_func0,post_func0,&a);
    a = 777;
    g_op.run(6);//执行kernel函数
    g_op.postrun();//执行host函数
    a = 2333;
    g_op.run(7);
    g_op.postrun();

    g_op.setPriority(10);
    printf("prio is %d\n",g_op.getPriority());
    g_op.setPriority(60);
    printf("prio is %d\n",g_op.getPriority());

    printf("reg_num is %d\n",g_op.getUsedResource().reg_num);
    printf("block_num is %d\n",g_op.getUsedResource().block_num);
    printf("thread_num is %d\n",g_op.getUsedResource().thread_num);
    printf("shared_mem_per_blk is %d\n",g_op.getUsedResource().shared_mem_per_blk);
    printf("shared_mem_total is %d\n",g_op.getUsedResource().shared_mem_total);

}

#endif