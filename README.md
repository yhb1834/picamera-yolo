# picamera-yolo

### How to load YOLO v5 and YOLO v7 model on the ROS2

### 1. Create your ROS2 workspace and package 

### 2. [Install PiCamera onto Raspberry Pi 4b+](https://index.ros.org/r/v4l2_camera/)
    $ sudo apt-get install ros-${ROS_DISTRO}-v4l2-camera
    $ git clone --branch ${ROS_DISTRO} https://gitlab.com/boldhearts/ros2_v4l2_camera.git src/v4l2_camera
    $ rosdep install --from-paths src/v4l2_camera --ignore-src -r -y
    $ colcon build

**How to know picamera is connected on to the Raspberry Pi** 
    $ vcgencmd get_camera 
    
### 3. run PiCamera
    $ ros2 run v4l2_camera v4l2_camera_node

### 4. download [yolo.py]() to your workspace's node
    $ colcon build
    $ . install/setup.bash
    $ ros2 run ____  _____
