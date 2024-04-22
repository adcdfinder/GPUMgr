cmake_minimum_required(VERSION 3.5)
project(ros2_gpu)
# add_compile_options(-g -finstrument-functions)



# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -fPIC -I /usr/local/cuda/include/ -lpthread)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)


add_executable(main src/main.cpp)
target_include_directories(main PUBLIC inc)
ament_target_dependencies(main rclcpp std_msgs)

add_executable(qtest src/test.cpp)
target_include_directories(qtest PUBLIC inc)
ament_target_dependencies(qtest rclcpp std_msgs)



# add CUDA

FIND_PACKAGE(CUDA REQUIRED)
# SET(CUDA_NVCC_FLAGS "-arch=sm_75 --resource-usage " CACHE STRING "nvcc flags" FORCE)
SET(CUDA_NVCC_FLAGS "-arch=sm_75 --resource-usage " CACHE STRING "nvcc flags" FORCE)
#SET(CUDA_NVCC_FLAGS -arch=sm_75 --resource-usage -Xcompiler -fPIC)
SET (CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
include_directories(${CUDA_INCLUDE_DIRS} inc)
link_directories(${CUDA_LIBRARY_DIRS} inc)


# add CUDA library cuGpuMgrIntf
# 此处的 CUDA_SRC_cuGpuMgrIntf 只是为了测试，并不正确
SET(CUDA_LIB_cuGpuMgrIntf cuGpuMgrIntf)
SET(LIB_cuGpuMgrIntf_TYPE SHARED)
SET(CUDA_SRC_cuGpuMgrIntf
  src/cuda/gpu_mgr.cu
)
CUDA_ADD_LIBRARY(${CUDA_LIB_cuGpuMgrIntf} ${LIB_cuGpuMgrIntf_TYPE} ${CUDA_SRC_cuGpuMgrIntf})


# add CUDA library cuSelfAdd
SET(CUDA_LIB_cuMini cuMini)
SET(LIB_cuMini_TYPE STATIC)
SET(CUDA_SRC_cuMini
  src/cuda/mini.cu
  src/cuda/util.cu
)
CUDA_ADD_LIBRARY(${CUDA_LIB_cuMini} ${LIB_cuMini_TYPE} ${CUDA_SRC_cuMini})
ADD_DEPENDENCIES(${CUDA_LIB_cuMini} ${CUDA_LIB_cuGpuMgrIntf})
TARGET_LINK_LIBRARIES(${CUDA_LIB_cuMini} ${CUDA_LIB_cuGpuMgrIntf})

# add CUDA library cuSelfAdd
#SET(CUDA_LIB_cuSelfAdd cuSelfAdd)
#SET(LIB_cuSelfAdd_TYPE STATIC)
#SET(CUDA_SRC_cuSelfAdd
#  src/cuda/selfadd.cu
#  src/cuda/util.cu
#)
#CUDA_ADD_LIBRARY(${CUDA_LIB_cuSelfAdd} ${LIB_cuSelfAdd_TYPE} ${CUDA_SRC_cuSelfAdd})
#ADD_DEPENDENCIES(${CUDA_LIB_cuSelfAdd} ${CUDA_LIB_cuGpuMgrIntf})
#TARGET_LINK_LIBRARIES(${CUDA_LIB_cuSelfAdd} ${CUDA_LIB_cuGpuMgrIntf})


# add CUDA library cuGpuMgr
#SET(CUDA_LIB_GpuMgr cuGpuMgr)
#SET(LIB_GpuMgr_TYPE STATIC)
#SET(CUDA_SRC_GpuMgr
#  src/cuda/gpu_mgr.cu
#)

#CUDA_ADD_LIBRARY(${CUDA_LIB_GpuMgr} ${LIB_GpuMgr_TYPE} ${CUDA_SRC_GpuMgr})
#ADD_DEPENDENCIES(${CUDA_LIB_GpuMgr} ${CUDA_LIB_cuGpuMgrIntf})
#TARGET_LINK_LIBRARIES(${CUDA_LIB_GpuMgr} ${CUDA_LIB_cuGpuMgrIntf})

# add dependencies and libs

ADD_DEPENDENCIES(
  main 
  ${CUDA_LIB_cuGpuMgrIntf}
  ${CUDA_LIB_cuMini}
)

TARGET_LINK_LIBRARIES(
  main
  ${CUDA_LIB_cuGpuMgrIntf}
  ${CUDA_LIB_cuMini}
)

ADD_DEPENDENCIES(
  qtest 
  ${CUDA_LIB_cuGpuMgrIntf}
  ${CUDA_LIB_cuMini}
)

TARGET_LINK_LIBRARIES(
  qtest 
  ${CUDA_LIB_cuGpuMgrIntf}
  ${CUDA_LIB_cuMini}
)



install(TARGETS
  main
  qtest
  ${CUDA_LIB_cuGpuMgrIntf}
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
