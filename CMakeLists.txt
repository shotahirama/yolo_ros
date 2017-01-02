cmake_minimum_required(VERSION 2.8.3)
project(yolo_ros)

add_definitions("-std=c++11")

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  image_transport
  roscpp
  rospy
  sensor_msgs
  std_msgs
  message_generation
)

add_message_files(
  FILES
  DetectObject.msg
  DetectObjectArray.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  INCLUDE_DIRS include
#  LIBRARIES yolo_ros
#  CATKIN_DEPENDS cv_bridge geometry_msgs image_transport roscpp rospy sensor_msgs std_msgs
#  DEPENDS system_lib
)

include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/darknet)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/darknet/src)

include(cmake/cuda_darknet.cmake)
cudacheck()

add_executable(yolo_node src/yolo_node.cpp)
target_link_libraries(yolo_node ${catkin_LIBRARIES} darknet)
