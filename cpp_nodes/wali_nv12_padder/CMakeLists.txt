cmake_minimum_required(VERSION 3.8)
project(wali_nv12_padder)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(nv12_padder_node src/nv12_padder_node.cpp)
ament_target_dependencies(nv12_padder_node rclcpp sensor_msgs)

target_compile_options(nv12_padder_node PRIVATE -O3 -Wall -Wextra -Wpedantic)

install(TARGETS nv12_padder_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
