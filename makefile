# This makefile is not invovled the organization of ROS2 build
# It's a simple script to help running cmd lines
PATH_TO_LIBGPUMGRINTF=/home/hwang/GPUMgr/install/ros2_gpu/lib/ros2_gpu/libcuGpuMgrIntf.so

build-colcon:
	colcon build --parallel-workers 12

run:
	LD_PRELOAD=${PATH_TO_LIBGPUMGRINTF} \
	ros2 run ros2_gpu main

qtest:
	LD_PRELOAD=${PATH_TO_LIBGPUMGRINTF} \
	ros2 run ros2_gpu qtest

wake_loop:
	ros2 topic pub /while_loop std_msgs/msg/Int32 "{data: 1}" --once

end_loop:
	ros2 topic pub /loop_end std_msgs/msg/Int32 "{data: 1}" --once

trig:
	ros2 topic pub /trig std_msgs/msg/Int32 "{data: 1}" --once

MINI_IN ?= 1

mini:
	ros2 topic pub /mini std_msgs/msg/Int32 "{data: ${MINI_IN}}" --once

affinity:
	ros2 topic pub /affinity std_msgs/msg/Int32 "{data: ${MINI_IN}}" --once

normal:
	ros2 topic pub /normal std_msgs/msg/Int32 "{data: ${MINI_IN}}" --once

slice:
	ros2 topic pub /slice std_msgs/msg/Int32 "{data: ${MINI_IN}}" --once

s0:
	ros2 topic pub /s0 std_msgs/msg/Int32 "{data: ${MINI_IN}}" --once

s1:
	ros2 topic pub /s1 std_msgs/msg/Int32 "{data: ${MINI_IN}}" --once

profile:
	nvprof \
	--print-gpu-trace \
	--device-buffer-size 256 \
	--profile-child-processes \
	ros2 run ros2_gpu main
	# ./install/ros2_gpu/lib/ros2_gpu/main
	
ptest:
	LD_PRELOAD=${PATH_TO_LIBGPUMGRINTF} \
	nvprof \
	--print-gpu-trace \
	--device-buffer-size 256 \
	--profile-child-processes \
	ros2 run ros2_gpu qtest


# ./install/ros2_cuda_demo/lib/ros2_cuda_demo/main