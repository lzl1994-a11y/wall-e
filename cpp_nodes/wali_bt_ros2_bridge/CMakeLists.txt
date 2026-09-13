cmake_minimum_required(VERSION 3.16)
project(wali_bt_ros2_bridge)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(behaviortree_ros2 REQUIRED)
find_package(btcpp_ros2_interfaces REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(wali_bt_ros2_bridge src/wali_bt_ros2_bridge.cpp)
ament_target_dependencies(wali_bt_ros2_bridge
  behaviortree_ros2
  btcpp_ros2_interfaces
  rclcpp
  std_msgs)
target_link_libraries(wali_bt_ros2_bridge nlohmann_json::nlohmann_json)
target_compile_options(wali_bt_ros2_bridge PRIVATE -Wall -Wextra -Wpedantic)

install(TARGETS wali_bt_ros2_bridge
  DESTINATION lib/${PROJECT_NAME})

ament_package()
