[toc]
# How to build and run

> 参考：https://zhuanlan.zhihu.com/p/354346029

## 基础知识

ROS工作空间是具有特定结构的目录。工作空间中通常有一个src子目录。在该子目录中是ROS软件包源代码所在的位置。通常，该子目录开始时是空的。

Colcon进行源代码构建。默认情况下，它会创建与src目录平级的下列目录：

- build目录：是存储中间文件的目录。会为每个软件包在build目录中创建一个子目录，在该子目录中，例如，CMake会被调用。

- install目录：这是每个软件包将被安装的目录。默认情况下，每个软件包都将会被安装到一个单独的子目录中。

- log目录：该目录包含有关每个colcon调用的各种日志信息。

## Build

```shell
make build-colcon
```

## 设置环境

当colcon成功地build完之后，packages会安装在”install”目录中。为了使用其中的可执行文件和库，我们需要将其目录添加到搜索路径中。Colcon会在”install”目录中生成所需脚本来完成环境的设置。命令如下：

```shell
source install/setup.bash
```

## Run

> 注意：一个新打开的终端，需要先设置环境，才能run。所以每个新打开的终端需要先执行 source install/setup.bash

```
source install/setup.bash
make run
```

In one terminal

```shell
source install/setup.bash
make qtest # Setup GPU Manager node and one simple test node
```

In 2nd terminal
```shell
make wake_loop # Publish one msg to activate GPU Manager monitor loop
make end_loop # Publish one msg to deactivate GPU Manager monitor loop
```

In 2nd and other terminals
```shell
make mini # Publish one digit to test node and let GPU compute
make mini MINI_IN=5 # Publish 5 to test node and let GPU compute
```