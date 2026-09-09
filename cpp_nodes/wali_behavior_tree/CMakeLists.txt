cmake_minimum_required(VERSION 3.8)
project(wali_behavior_tree)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(behaviortree_cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(behavior_tree_node src/behavior_tree_node.cpp)
ament_target_dependencies(behavior_tree_node behaviortree_cpp rclcpp std_msgs)
target_link_libraries(behavior_tree_node nlohmann_json::nlohmann_json)
target_compile_options(behavior_tree_node PRIVATE -Wall -Wextra -Wpedantic)

install(TARGETS behavior_tree_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
